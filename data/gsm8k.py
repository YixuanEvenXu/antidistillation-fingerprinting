"""gsm8k dataset provider."""

from __future__ import annotations

from typing import Iterable, Iterator

from datasets import load_dataset

from data.base import DatasetExample, DatasetProvider

_SYSTEM_PROMPT = (
    "You are a careful math tutor who explains reasoning before answering."
)


class GSM8KProvider(DatasetProvider):
    name = "gsm8k"

    def load(self, split: str, limit: int | None = None) -> Iterable[DatasetExample]:
        """Load GSM8K question/answer pairs.

        Args:
            split: Dataset split name (e.g., "train", "test").
            limit: Optional cap on the number of examples.

        Returns:
            Iterable of DatasetExample with prompt and solution fields.
        """
        dataset = load_dataset("gsm8k", "main", split=split)
        total = len(dataset)
        max_rows = min(total, limit) if limit is not None else total
        for idx in range(max_rows):
            row = dataset[int(idx)]
            yield DatasetExample(prompt=row["question"].strip(), solution=row["answer"].strip())


__all__ = ["GSM8KProvider", "_SYSTEM_PROMPT"]
