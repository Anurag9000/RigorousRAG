"""Shared contracts for sparse, late-interaction, and multimodal retrieval models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

_MODES = {"sparse", "late-interaction", "multimodal"}


def finite_vector(values: Sequence[Any]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not values or len(values) > 1_000_000:
        raise RuntimeError("invalid model vector")
    result = []
    for value in values:
        if isinstance(value, bool):
            raise RuntimeError("model vectors must contain finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("model vectors must contain finite numbers") from exc
        if not math.isfinite(number):
            raise RuntimeError("model vectors must contain finite numbers")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class RetrievalModelSpec:
    mode: str
    model_name: str
    revision: str
    checksum_sha256: str
    allow_download: bool = False
    trust_remote_code: bool = False
    max_terms: int = 2048
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("unsupported retrieval mode")
        for name in ("model_name", "revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 300 or "\x00" in value:
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, value.strip())
        checksum = self.checksum_sha256.lower() if isinstance(self.checksum_sha256, str) else ""
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError("checksum_sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "checksum_sha256", checksum)
        if not isinstance(self.allow_download, bool) or not isinstance(self.trust_remote_code, bool):
            raise ValueError("download/trust controls must be booleans")
        if isinstance(self.max_terms, bool) or not isinstance(self.max_terms, int) or not 1 <= self.max_terms <= 100_000:
            raise ValueError("max_terms is invalid")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or not 1 <= self.max_tokens <= 32_768:
            raise ValueError("max_tokens is invalid")


@dataclass(frozen=True)
class MultimodalInput:
    text: str | None = None
    image_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if self.text is None and self.image_bytes is None:
            raise ValueError("text or image_bytes is required")
        if self.text is not None and (not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 1_000_000):
            raise ValueError("text is invalid")
        if self.image_bytes is not None and (not isinstance(self.image_bytes, bytes) or not 1 <= len(self.image_bytes) <= 50_000_000):
            raise ValueError("image_bytes is invalid")


class SparseBackend(Protocol):
    def encode_sparse(self, texts: Sequence[str]) -> Sequence[Mapping[str, float]]: ...


class LateInteractionBackend(Protocol):
    def encode_tokens(self, texts: Sequence[str]) -> Sequence[Sequence[Sequence[float]]]: ...


class MultimodalBackend(Protocol):
    def encode_multimodal(self, items: Sequence[MultimodalInput]) -> Sequence[Sequence[float]]: ...


__all__ = ["LateInteractionBackend", "MultimodalBackend", "MultimodalInput", "RetrievalModelSpec", "SparseBackend", "finite_vector"]
