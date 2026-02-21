"""Model loading helpers."""

from __future__ import annotations

from transformers import AutoModelForCausalLM
import torch

from config import ModelSpec

_DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {name}")
    return _DTYPE_ALIASES[key]


def load_causal_lm(spec: ModelSpec) -> AutoModelForCausalLM:
    dtype = resolve_dtype(spec.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name,
        dtype=dtype,
        trust_remote_code=True,
        use_cache=True,
    )
    model.eval()
    return model
