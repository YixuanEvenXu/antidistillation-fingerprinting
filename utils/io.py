"""Simple JSON and JSONL helpers with explicit failure behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence


def read_json(path: Path) -> dict:
    """Read a JSON file into a dictionary.

    Args:
        path: Path to a JSON file.

    Returns:
        Parsed JSON object as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    """Write a dictionary to a JSON file.

    Args:
        path: Destination path for the JSON file.
        payload: Dictionary to serialize.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file line by line.

    Args:
        path: Path to a JSON Lines file.

    Yields:
        One dict per non-empty line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dictionaries to JSON Lines.

    Args:
        path: Destination path for the JSONL file.
        rows: Iterable of dictionaries to write.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def read_jsonl_rows(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    return list(iter_jsonl(path))
