"""OpenAssistant OASST1 dataset utilities and provider."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datasets import load_dataset
from tqdm.auto import tqdm

from data.base import DatasetExample, DatasetProvider


def _value(row: Dict[str, Any], *keys: str) -> Any:
    """Return the first non-null field among candidate keys.

    Args:
        row: Input row dictionary.
        *keys: Ordered keys to search.

    Returns:
        Value for the first present, non-null key, or None.
    """
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _message_id(row: Dict[str, Any]) -> Optional[str]:
    """Extract a message id from a row.

    Args:
        row: Input row dictionary.

    Returns:
        Message id as a string, or None if missing.
    """
    mid = _value(row, "message_id", "id")
    return str(mid) if mid is not None else None


def _parent_id(row: Dict[str, Any]) -> Optional[str]:
    """Extract a parent id from a row.

    Args:
        row: Input row dictionary.

    Returns:
        Parent id as a string, or None if missing.
    """
    pid = _value(row, "parent_id")
    return str(pid) if pid is not None else None


def _tree_id(row: Dict[str, Any]) -> str:
    """Extract the conversation/tree id from a row.

    Args:
        row: Input row dictionary.

    Returns:
        Tree id string, or empty string if missing.
    """
    tid = _value(row, "message_tree_id", "conversation_id")
    return str(tid) if tid is not None else ""


def _rank(row: Dict[str, Any]) -> float:
    """Parse a sortable rank value from a row.

    Args:
        row: Input row dictionary.

    Returns:
        Parsed float rank, or +inf on parse failure.
    """
    rank = row.get("rank")
    try:
        return float(rank)
    except Exception:
        return float("inf")


def _chatml(messages: Sequence[dict[str, Any]]) -> str:
    """Render messages into ChatML for length checks (output not persisted).

    Args:
        messages: Sequence of message dicts with role/text fields.

    Returns:
        ChatML-formatted string, or empty string if inputs are invalid.
    """
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role")
        role = "user" if role == "prompter" else role
        content = (msg.get("text") or "").strip()
        if not role or not content:
            return ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _gather_chain(msg: dict[str, Any], index: dict[str, dict[str, Any]]) -> Optional[List[dict[str, Any]]]:
    """Return root->...->msg chain; None if any parent is missing.

    Args:
        msg: Leaf message row.
        index: Mapping of message_id -> row.

    Returns:
        Ordered list of rows from root to leaf, or None on missing parents.
    """
    chain: List[dict[str, Any]] = []
    cur = msg
    while cur:
        chain.append(cur)
        pid = _parent_id(cur)
        if pid is None:
            break
        cur = index.get(pid)
        if cur is None:
            return None
    chain.reverse()
    return chain


def convert_oasst1(
    split: str,
    output_path: Path,
    limit: Optional[int],
    lang: Optional[str],
    min_review_count: int,
    min_chars: int,
    max_prompt_chars: int,
    drop_metadata: bool,
) -> int:
    """Flatten OASST1 into ChatML message lists (context only, no assistant reply).

    Args:
        split: Dataset split to load.
        output_path: Destination JSONL path.
        limit: Optional cap on number of prompts to emit.
        lang: Optional language filter (e.g., "en").
        min_review_count: Minimum assistant review count to keep.
        min_chars: Minimum character count for assistant responses.
        max_prompt_chars: Maximum ChatML prompt length; longer contexts are dropped.
        drop_metadata: If True, omit metadata fields from output rows.

    Returns:
        Number of prompts written to the output file.
    """
    dataset = load_dataset("OpenAssistant/oasst1", split=split)

    index: dict[str, dict[str, Any]] = {}
    assistants: List[dict[str, Any]] = []
    for row in dataset:
        if lang and row.get("lang") != lang:
            continue
        if row.get("deleted"):
            continue
        role = row.get("role")
        if role not in {"prompter", "assistant"}:
            continue
        mid = _message_id(row)
        if not mid:
            continue
        index[mid] = row
        if role == "assistant":
            assistants.append(row)

    assistants.sort(key=lambda r: (_tree_id(r), _rank(r), _message_id(r) or ""))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    emitted_parents: set[str] = set()

    with output_path.open("w", encoding="utf-8") as handle:
        iterator = tqdm(assistants, desc="Writing OASST1 chatml prompts") if len(assistants) > 1000 else assistants
        for reply in iterator:
            if limit is not None and written >= limit:
                break

            response_text = (reply.get("text") or "").strip()
            if len(response_text) < min_chars or reply.get("review_count", 0) < min_review_count:
                continue

            chain = _gather_chain(reply, index)
            if not chain or not chain[-1] is reply:
                continue
            if chain[0].get("role") != "prompter":
                continue

            context_messages = chain[:-1]
            if not context_messages:
                continue

            pid = _parent_id(reply)
            if pid and pid in emitted_parents:
                continue

            prompt_text = _chatml(context_messages)
            if not prompt_text or len(prompt_text) > max_prompt_chars:
                continue

            msg_payload = [
                {
                    "role": ("user" if m.get("role") == "prompter" else "assistant"),
                    "content": (m.get("text") or "").strip(),
                }
                for m in context_messages
            ]
            payload: dict[str, Any] = {"messages": msg_payload}
            if not drop_metadata:
                payload.update(
                    {
                        "lang": reply.get("lang"),
                        "assistant_message_id": _message_id(reply),
                        "parent_id": _parent_id(reply),
                        "message_tree_id": _tree_id(reply),
                        "depth": len(chain),
                        "review_count": int(reply.get("review_count") or 0),
                        "rank": reply.get("rank"),
                    }
                )
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            written += 1
            if pid:
                emitted_parents.add(pid)

    return written


@dataclass(slots=True)
class OASST1Provider(DatasetProvider):
    name: str = "oasst1"
    data_path: Path | None = None

    def load(self, split: str, limit: int | None = None) -> Iterable[DatasetExample]:
        """Load OASST1 message contexts from a prebuilt JSONL file.

        Args:
            split: Dataset split name (ignored; JSONL is always read as "train").
            limit: Optional cap on the number of examples.

        Returns:
            Iterable of DatasetExample with messages populated.

        Raises:
            FileNotFoundError: If the JSONL file cannot be found.
        """
        path = self.data_path or Path(os.environ.get("OASST1_PATH", "data/oasst1_chatml_messages.jsonl"))
        if not path.exists():
            raise FileNotFoundError(
                f"OASST1 JSONL not found at {path}. "
                "Generate it with data/oasst1.py --output <path> or set OASST1_PATH."
            )
        dataset = load_dataset("json", data_files=str(path), split="train")
        total = len(dataset)
        max_rows = min(total, limit) if limit is not None else total
        for idx in range(max_rows):
            row = dataset[int(idx)]
            messages = row.get("messages")
            if not messages:
                continue
            yield DatasetExample(prompt=None, messages=messages, solution=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for OASST1 conversion.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Convert OpenAssistant/oasst1 to ChatML message JSONL.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load.")
    parser.add_argument("--lang", type=str, default="en", help="Language filter; set empty to keep all.")
    parser.add_argument("--min-review-count", type=int, default=1, help="Drop assistant replies with fewer reviews.")
    parser.add_argument("--min-chars", type=int, default=16, help="Minimum character count for prompt/response.")
    parser.add_argument("--max-prompt-chars", type=int, default=8000, help="Maximum ChatML prompt length; drop longer contexts.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of rows to write.")
    parser.add_argument("--drop-metadata", action="store_true", help="If set, emit only messages without metadata fields.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/oasst1_chatml_messages.jsonl"),
        help="Destination JSONL file.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for OASST1 conversion.

    Args:
        argv: Optional list of CLI arguments (defaults to sys.argv).

    Returns:
        None.
    """
    args = build_parser().parse_args(argv)
    lang = args.lang or None
    total = convert_oasst1(
        split=args.split,
        output_path=args.output,
        limit=args.limit,
        lang=lang,
        min_review_count=args.min_review_count,
        min_chars=args.min_chars,
        max_prompt_chars=args.max_prompt_chars,
        drop_metadata=args.drop_metadata,
    )
    print(f"Wrote {total} chatml prompts to {args.output}")


if __name__ == "__main__":
    main()
