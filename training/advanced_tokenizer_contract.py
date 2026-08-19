"""Tokenizer invariants required by advanced RAG span/collation semantics."""
from __future__ import annotations

from typing import Any


def assert_advanced_training_tokenizer(tokenizer: Any) -> None:
    """Fail before collation when span offsets or padding semantics are incompatible."""
    if tokenizer is None or not callable(tokenizer):
        raise ValueError("advanced RAG tokenizer must be callable")
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("advanced RAG span supervision requires a fast tokenizer with offset mappings")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("advanced RAG training requires tokenizer.padding_side='right'")
    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ValueError("advanced RAG tokenizer requires pad_token_id")
    side = getattr(tokenizer, "truncation_side", "right")
    if side not in {"left", "right"}:
        raise ValueError("tokenizer.truncation_side must be left or right")


__all__ = ["assert_advanced_training_tokenizer"]
