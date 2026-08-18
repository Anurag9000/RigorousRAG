"""Verified last-mile materialization of selected evidence into generator context.

The packing layer selects immutable evidence references without reading source text. This
module is the authority boundary where text is fetched. Every selected item must have a
binding to an exact text SHA-256, tokenizer identity and token count. Provider output is
rehashed and retokenized before it can become a materialized context artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from tools.evidence_context_packing import ContextPackingReceipt


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


@dataclass(frozen=True)
class ContextContentBinding:
    evidence_sha256: str
    text_sha256: str
    tokenizer_sha256: str
    token_count: int

    def __post_init__(self) -> None:
        for name in ("evidence_sha256", "text_sha256", "tokenizer_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 1:
            raise ValueError("token_count must be a positive integer")

    @property
    def binding_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-context-content-binding/v1", **asdict(self)})


class EvidenceContentProvider(Protocol):
    def fetch_text(self, *, evidence_sha256: str) -> str: ...


class TokenCounter(Protocol):
    @property
    def tokenizer_sha256(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class MaterializedEvidence:
    order: int
    evidence_sha256: str
    text: str
    text_sha256: str
    token_count: int

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order must be positive")
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        if not isinstance(self.text, str) or not self.text or "\x00" in self.text:
            raise ValueError("materialized evidence text is invalid")
        object.__setattr__(self, "text_sha256", _sha(self.text_sha256, "text_sha256"))
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("text_sha256 does not match materialized text")
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 1:
            raise ValueError("token_count must be positive")


@dataclass(frozen=True)
class MaterializedContext:
    packing_receipt_sha256: str
    tokenizer_sha256: str
    binding_set_sha256: str
    evidence: tuple[MaterializedEvidence, ...]
    total_tokens: int
    context_sha256: str

    def __post_init__(self) -> None:
        for name in ("packing_receipt_sha256", "tokenizer_sha256", "binding_set_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        rows = tuple(self.evidence)
        if [row.order for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("materialized evidence order must be contiguous")
        if len({row.evidence_sha256 for row in rows}) != len(rows):
            raise ValueError("materialized context contains duplicate evidence")
        object.__setattr__(self, "evidence", rows)
        if isinstance(self.total_tokens, bool) or not isinstance(self.total_tokens, int) or self.total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if self.total_tokens != sum(row.token_count for row in rows):
            raise ValueError("total_tokens does not match materialized evidence")
        expected = _digest(self._payload())
        provided = _sha(self.context_sha256, "context_sha256")
        if expected != provided:
            raise ValueError("context_sha256 does not match materialized context")
        object.__setattr__(self, "context_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-materialized-context/v1",
            "packing_receipt_sha256": self.packing_receipt_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "binding_set_sha256": self.binding_set_sha256,
            "evidence": [
                {
                    "order": row.order,
                    "evidence_sha256": row.evidence_sha256,
                    "text_sha256": row.text_sha256,
                    "token_count": row.token_count,
                }
                for row in self.evidence
            ],
            "total_tokens": self.total_tokens,
        }

    @property
    def prompt_text(self) -> str:
        return "\n\n".join(row.text for row in self.evidence)


def _binding_set_sha256(bindings: Sequence[ContextContentBinding]) -> str:
    return _digest({"schema": "rigorousrag-context-content-binding-set/v1", "bindings": sorted(row.binding_sha256 for row in bindings)})


def materialize_context(
    packing: ContextPackingReceipt,
    *,
    bindings: Sequence[ContextContentBinding],
    provider: EvidenceContentProvider,
    token_counter: TokenCounter,
) -> MaterializedContext:
    if not isinstance(packing, ContextPackingReceipt):
        raise ValueError("packing must be ContextPackingReceipt")
    rows = tuple(bindings)
    if any(not isinstance(row, ContextContentBinding) for row in rows):
        raise ValueError("bindings contains invalid values")
    if len({row.evidence_sha256 for row in rows}) != len(rows):
        raise ValueError("bindings contains duplicate evidence identities")
    tokenizer_sha = _sha(token_counter.tokenizer_sha256, "tokenizer_sha256")
    if any(row.tokenizer_sha256 != tokenizer_sha for row in rows):
        raise ValueError("all content bindings must use the runtime tokenizer identity")
    binding_by_evidence: Mapping[str, ContextContentBinding] = {row.evidence_sha256: row for row in rows}
    selected_ids = {row.evidence_sha256 for row in packing.selected}
    if set(binding_by_evidence) != selected_ids:
        raise ValueError("content bindings must exactly cover the packed evidence")

    materialized = []
    for packed in packing.selected:
        binding = binding_by_evidence[packed.evidence_sha256]
        if binding.token_count != packed.token_count:
            raise ValueError("content binding token count differs from packing receipt")
        text = provider.fetch_text(evidence_sha256=packed.evidence_sha256)
        if not isinstance(text, str) or not text or "\x00" in text:
            raise RuntimeError("evidence content provider returned invalid text")
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_sha != binding.text_sha256:
            raise RuntimeError("evidence text digest differs from immutable content binding")
        observed_tokens = token_counter.count_tokens(text)
        if isinstance(observed_tokens, bool) or not isinstance(observed_tokens, int) or observed_tokens < 1:
            raise RuntimeError("token counter returned an invalid token count")
        if observed_tokens != binding.token_count:
            raise RuntimeError("runtime token count differs from immutable content binding")
        materialized.append(MaterializedEvidence(packed.order, packed.evidence_sha256, text, text_sha, observed_tokens))

    materialized_rows = tuple(materialized)
    payload = {
        "schema": "rigorousrag-materialized-context/v1",
        "packing_receipt_sha256": packing.receipt_sha256,
        "tokenizer_sha256": tokenizer_sha,
        "binding_set_sha256": _binding_set_sha256(rows),
        "evidence": [
            {"order": row.order, "evidence_sha256": row.evidence_sha256, "text_sha256": row.text_sha256, "token_count": row.token_count}
            for row in materialized_rows
        ],
        "total_tokens": sum(row.token_count for row in materialized_rows),
    }
    return MaterializedContext(
        packing.receipt_sha256,
        tokenizer_sha,
        payload["binding_set_sha256"],
        materialized_rows,
        payload["total_tokens"],
        _digest(payload),
    )


__all__ = [
    "ContextContentBinding",
    "EvidenceContentProvider",
    "MaterializedContext",
    "MaterializedEvidence",
    "TokenCounter",
    "materialize_context",
]
