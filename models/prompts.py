"""Prompt construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from transformers import PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
    "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
)

OASST1_SYSTEM_PROMPT = (
    "You are a helpful, honest, and courteous assistant. "
    "Follow the user's instructions, ask clarifying questions when needed, "
    "and avoid harmful or unsafe content."
)


@dataclass(slots=True)
class PromptBuilder:
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT

    def build(self, tokenizer: PreTrainedTokenizerBase, user_prompt: str) -> str:
        """
        Return a prompt using the tokenizer's chat template (DeepSeek R1 compatible).
        """
        user_prompt = user_prompt.rstrip() + "\n"
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fallback for models without chat templates.
            parts = []
            for msg in messages:
                role = msg.get("role", "").strip().title() or "User"
                parts.append(f"{role}: {msg.get('content','').strip()}")
            parts.append("Assistant:")
            return "\n".join(parts)

    def build_from_messages(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: Sequence[dict[str, str]],
        *,
        add_system: bool = False,
    ) -> str:
        formatted: list[dict[str, str]] = []
        if add_system and self.system_prompt:
            formatted.append({"role": "system", "content": self.system_prompt})
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").rstrip()
            if not role or not content:
                continue
            formatted.append({"role": role, "content": content + "\n"})
        try:
            return tokenizer.apply_chat_template(
                formatted,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            # Fallback for tokenizers without chat templates: simple role-prefixed text.
            parts = []
            for msg in formatted:
                role = msg.get("role", "").strip().title() or "User"
                parts.append(f"{role}: {msg.get('content','').strip()}")
            parts.append("Assistant:")
            return "\n".join(parts)
