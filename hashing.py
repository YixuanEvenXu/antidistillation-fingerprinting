"""Hash configuration + deterministic Bernoulli mask generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch

from utils.io import read_json, write_json

Bigram = Tuple[int, int]


@dataclass(slots=True)
class HashConfig:
    seed: int
    gamma: float

    def __post_init__(self) -> None:
        """Normalize seed and validate gamma bounds.

        Args:
            None (uses dataclass fields on self).

        Returns:
            None. Raises ValueError if gamma is outside (0, 1).
        """
        # clamp to signed 63-bit to avoid torch int64 overflow
        object.__setattr__(self, "seed", int(self.seed) % (2**63))
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must lie in (0, 1)")


def load_hash_config(path: Path) -> HashConfig:
    """Load a hash configuration from JSON.

    Args:
        path: Path to a JSON file with "seed" and "gamma".

    Returns:
        Parsed HashConfig instance.
    """
    data = read_json(path)
    return HashConfig(seed=int(data["seed"]), gamma=float(data["gamma"]))


def write_hash_config(path: Path, config: HashConfig) -> None:
    """Write a hash configuration to JSON.

    Args:
        path: Destination path for the JSON file.
        config: HashConfig to serialize.

    Returns:
        None.
    """
    write_json(path, {"seed": config.seed, "gamma": config.gamma})


class BigramHash:
    """Deterministic bigram -> mask mapping."""

    def __init__(
        self,
        config: HashConfig,
        vocab_size: int,
        *,
        excluded_token_ids: Iterable[int] | None = None,
    ) -> None:
        """Create a deterministic bigram hash-to-mask generator.

        Args:
            config: HashConfig with seed and gamma.
            vocab_size: Size of the vocabulary to mask.
            excluded_token_ids: Optional token ids to always mark as False.

        Returns:
            None.
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.vocab_size = vocab_size
        self.excluded = torch.tensor(
            sorted({int(i) for i in (excluded_token_ids or [])}),
            dtype=torch.long,
        )

    def _derive_seed(self, bigram: Bigram) -> int:
        """Derive a deterministic seed from a bigram.

        Args:
            bigram: Tuple of (prev_id, next_id).

        Returns:
            64-bit integer seed derived from the base seed and bigram.
        """
        hasher = hashlib.sha256()
        hasher.update(str(self.config.seed).encode("utf-8"))
        hasher.update(b"::")
        hasher.update(str(int(bigram[0])).encode("utf-8"))
        hasher.update(b"::")
        hasher.update(str(int(bigram[1])).encode("utf-8"))
        return int.from_bytes(hasher.digest()[:8], "big")

    def _sample_mask_vec(
        self,
        bigrams: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Vectorized deterministic Bernoulli masks for a batch of bigrams.
        Uses integer hashing (splitmix64-style) to generate uniform draws.

        Args:
            bigrams: Tensor of shape [B, 2] with bigram token ids.
            device: Optional device to perform computation on.
            dtype: Optional dtype for the returned mask.

        Returns:
            Boolean or dtype-cast mask of shape [B, vocab_size].
        """
        dev = device or torch.device("cpu")
        bigrams = bigrams.to(torch.long)
        vocab_ids = torch.arange(self.vocab_size, device=dev, dtype=torch.long)
        # Constants for hashing (splitmix64-like) using int64 with wraparound.
        mask63 = torch.tensor((1 << 63) - 1, device=dev, dtype=torch.int64)
        # 64-bit friendly mixing constants (fit in signed int64).
        mul1 = torch.tensor(6364136223846793005, device=dev, dtype=torch.int64)
        mul2 = torch.tensor(1442695040888963407, device=dev, dtype=torch.int64)
        mul3 = torch.tensor(22695477, device=dev, dtype=torch.int64)
        safe_seed = int(self.config.seed) & ((1 << 63) - 1)
        seed = torch.tensor(safe_seed, device=dev, dtype=torch.int64)
        token_ids = vocab_ids.to(torch.int64)
        # Shape: [B, vocab]
        x = (token_ids.unsqueeze(0) * mul1) ^ (bigrams[:, :1].to(torch.int64) * mul2)
        x = (x ^ (bigrams[:, 1:].to(torch.int64) * mul3) ^ seed)
        x = (x ^ (x >> 30)) * mul2
        x = (x ^ (x >> 27)) * mul3
        x = x ^ (x >> 31)
        x = x & mask63
        draws = x.to(torch.float64) / float(2**63)
        mask = draws < float(self.config.gamma)
        if self.excluded.numel() > 0:
            excluded = self.excluded.to(dev)
            mask.index_fill_(1, excluded, False)
        if dtype:
            mask = mask.to(dtype)
        return mask

    def mask(
        self,
        bigram: Bigram,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute a mask vector for a single bigram.

        Args:
            bigram: Tuple of (prev_id, next_id).
            device: Optional device for computation.
            dtype: Optional dtype for the returned mask.

        Returns:
            Tensor of shape [vocab_size] with True at green-list positions.
        """
        bigrams = torch.tensor([[int(bigram[0]), int(bigram[1])]], device=device or "cpu")
        mask = self._sample_mask_vec(bigrams, device=device, dtype=dtype)
        return mask[0]

    def mask_batch(
        self,
        bigrams: Iterable[Bigram],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute mask vectors for a batch of bigrams.

        Args:
            bigrams: Iterable of (prev_id, next_id) pairs.
            device: Optional device for computation.
            dtype: Optional dtype for the returned mask.

        Returns:
            Tensor of shape [B, vocab_size] of masks; empty if no bigrams.
        """
        bigram_list = list(bigrams)
        if not bigram_list:
            return torch.empty(0, self.vocab_size, device=device or "cpu", dtype=dtype or torch.bool)
        tensor = torch.tensor(bigram_list, device=device or "cpu", dtype=torch.long)
        return self._sample_mask_vec(tensor, device=device, dtype=dtype or torch.bool)
