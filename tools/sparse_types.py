"""Public immutable sparse-index field, snapshot and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tools.sparse_utils import (
    _MAX_FIELD_CHARS,
    _SCHEMA_VERSION,
    _exact_int,
    _field_type,
    _identifier,
    _json_text,
    _optional_int,
)

@dataclass(frozen=True)
class SparseField:
    field_id: str
    field_type: str
    text: str
    position: int
    page_number: int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_id", _identifier(self.field_id, "field_id"))
        object.__setattr__(self, "field_type", _field_type(self.field_type))
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Sparse field text is required.")
        if len(self.text) > _MAX_FIELD_CHARS:
            raise ValueError("Sparse field text exceeds the character limit.")
        object.__setattr__(
            self,
            "position",
            _exact_int(self.position, "position", minimum=0, maximum=10_000_000),
        )
        object.__setattr__(
            self,
            "page_number",
            _optional_int(
                self.page_number,
                "page_number",
                minimum=1,
                maximum=1_000_000,
            ),
        )
        if self.section is not None:
            if not isinstance(self.section, str) or len(self.section) > 1_000 or any(
                ord(character) < 32 or ord(character) == 127 for character in self.section
            ):
                raise ValueError("section is invalid.")
        _json_text(self.metadata, "field metadata")


@dataclass(frozen=True)
class SparseFieldSnapshot:
    field_id: str
    field_type: str
    text: str
    position: int
    token_count: int
    page_number: int | None
    section: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SparseDocumentSnapshot:
    owner_id: str
    doc_id: str
    generation: int
    profile_fingerprint: str
    metadata: Mapping[str, Any]
    fields: tuple[SparseFieldSnapshot, ...]
    schema_version: int = _SCHEMA_VERSION


@dataclass(frozen=True)
class SparseMatch:
    field_id: str
    field_type: str
    field_position: int
    page_number: int | None
    section: str | None
    term_frequencies: Mapping[str, int]
    positions: Mapping[str, tuple[int, ...]]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SparseSearchHit:
    doc_id: str
    score: float
    generation: int
    profile_fingerprint: str
    metadata: Mapping[str, Any]
    matches: tuple[SparseMatch, ...]


