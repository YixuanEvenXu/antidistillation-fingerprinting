"""Dataset abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(slots=True)
class DatasetExample:
    prompt: str | None = None
    solution: str | None = None
    messages: list[dict[str, str]] | None = None


class DatasetProvider(Protocol):
    name: str

    def load(self, split: str, limit: int | None = None) -> Iterable[DatasetExample]:
        ...
