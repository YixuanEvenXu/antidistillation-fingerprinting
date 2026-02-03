"""Tokenizer loading and alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import ModelSpec


@dataclass(slots=True)
class TokenOverlap:
    """Mapping data between two tokenizers."""

    source_to_target: torch.Tensor  # shape: [source_vocab]
    shared_token_strings: set[str]
    string_to_target_ids: Dict[str, List[int]]


def load_tokenizer(spec: ModelSpec, *, padding_side: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and ensure a valid pad token is configured.

    Args:
        spec: ModelSpec describing the tokenizer to load.
        padding_side: Padding side ("left" or "right").

    Returns:
        Initialized Hugging Face tokenizer instance.

    Raises:
        ValueError: If no pad token is available after fallback logic.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        spec.name,
        padding_side=padding_side,
        use_fast=True,
        trust_remote_code=True,
    )
    if spec.pad_token:
        tokenizer.pad_token = spec.pad_token
    lowered = spec.name.lower()
    if tokenizer.pad_token_id is None:
        # comparison behavior: set pad for LLaMA/Qwen if missing
        if "llama" in lowered:
            tokenizer.pad_token_id = tokenizer.pad_token_id or 128004
            tokenizer.eos_token_id = tokenizer.eos_token_id or 128001
        elif "qwen" in lowered:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None:
        raise ValueError(
            f"Tokenizer {spec.name} does not define pad_token_id and fallback failed. Pass an explicit pad token."
        )
    return tokenizer


def build_vocab_lookup(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Build a token string -> id lookup with de-duplication.

    Args:
        tokenizer: Tokenizer whose vocabulary should be indexed.

    Returns:
        Dictionary mapping token string to the smallest observed id.
    """
    vocab = tokenizer.get_vocab()
    # Some tokenizers return duplicate entries; keep the first (smallest id) per token.
    dedup: Dict[str, int] = {}
    for token, idx in vocab.items():
        if token not in dedup or idx < dedup[token]:
            dedup[token] = int(idx)
    return dedup


def tokenizer_id_span(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the maximum token id span for a tokenizer.

    Args:
        tokenizer: Tokenizer to inspect.

    Returns:
        Integer span covering max token id + 1 and tokenizer length.
    """
    vocab = tokenizer.get_vocab()
    span = int(max(vocab.values())) + 1 if vocab else 0
    return max(span, len(tokenizer))


def compute_overlap(
    source: PreTrainedTokenizerBase,
    target: PreTrainedTokenizerBase,
) -> TokenOverlap:
    """Compute shared tokens and id mapping from source to target tokenizer.

    Args:
        source: Source tokenizer whose ids are to be mapped.
        target: Target tokenizer providing destination ids.

    Returns:
        TokenOverlap with a source->target id tensor and shared token strings.
    """
    source_vocab = source.get_vocab()
    target_lookup = build_vocab_lookup(target)
    source_span = tokenizer_id_span(source)
    target_span = len(target)
    mapping = torch.full((source_span,), -1, dtype=torch.long)
    shared: set[str] = set()
    for token, src_id in source_vocab.items():
        tgt_id = target_lookup.get(token)
        if tgt_id is None or tgt_id >= target_span:
            continue
        mapping[src_id] = tgt_id
        shared.add(token)
    string_to_target: Dict[str, List[int]] = {}
    for token, tgt_id in target_lookup.items():
        if token in shared and tgt_id < target_span:
            string_to_target.setdefault(token, []).append(tgt_id)
    return TokenOverlap(mapping, shared, string_to_target)
