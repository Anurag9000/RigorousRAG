"""Validated immutable embedding profile definitions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Sequence

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,299}$")
_ALLOWED_MODES = frozenset({"dense", "sparse", "multi-vector"})
_MAX_TEXT = 4_096


def clean_text(
    value: Any,
    label: str,
    *,
    maximum: int = _MAX_TEXT,
    allow_empty: bool = True,
    require_trimmed: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    if require_trimmed and value != value.strip():
        raise ValueError(f"{label} may not contain leading or trailing whitespace.")
    if len(value) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{label} is invalid or exceeds {maximum} characters.")
    if not allow_empty and not value:
        raise ValueError(f"{label} is required.")
    return value


def profile_alias(value: Any) -> str:
    text = clean_text(value, "profile alias", maximum=128, allow_empty=False)
    if not _ALIAS_RE.fullmatch(text):
        raise ValueError(
            "Profile aliases must use lowercase letters, numbers, '.', '_' or '-'."
        )
    return text


def model_name(value: Any) -> str:
    text = clean_text(value, "model_name", maximum=300, allow_empty=False)
    if not _MODEL_RE.fullmatch(text):
        raise ValueError("model_name contains unsupported characters.")
    return text


def optional_exact_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer or null.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class EmbeddingProfile:
    alias: str
    model_name: str
    dimensions: int | None
    max_sequence_tokens: int | None
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize_embeddings: bool = True
    language: str = "unknown"
    domain: str = "general"
    modes: tuple[str, ...] = ("dense",)
    license: str = "unknown"
    source_url: str = ""
    notes: str = ""
    schema_version: int = 1
    requires_adapter: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "alias", profile_alias(self.alias))
        object.__setattr__(self, "model_name", model_name(self.model_name))
        object.__setattr__(
            self,
            "dimensions",
            optional_exact_int(
                self.dimensions, "dimensions", minimum=1, maximum=1_000_000
            ),
        )
        object.__setattr__(
            self,
            "max_sequence_tokens",
            optional_exact_int(
                self.max_sequence_tokens,
                "max_sequence_tokens",
                minimum=1,
                maximum=10_000_000,
            ),
        )
        for name in (
            "query_prefix",
            "passage_prefix",
            "language",
            "domain",
            "license",
            "source_url",
            "notes",
        ):
            object.__setattr__(
                self,
                name,
                clean_text(
                    getattr(self, name),
                    name,
                    maximum=8_192 if name == "notes" else _MAX_TEXT,
                    require_trimmed=name not in {"query_prefix", "passage_prefix"},
                ),
            )
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be a boolean.")
        if not isinstance(self.requires_adapter, bool):
            raise ValueError("requires_adapter must be a boolean.")
        schema = optional_exact_int(
            self.schema_version,
            "schema_version",
            minimum=1,
            maximum=1_000_000,
        )
        if schema is None:
            raise ValueError("schema_version may not be null.")
        object.__setattr__(self, "schema_version", schema)
        if isinstance(self.modes, (str, bytes, bytearray)) or not isinstance(
            self.modes, Sequence
        ):
            raise ValueError("modes must be a sequence.")
        cleaned_modes: list[str] = []
        for mode in self.modes:
            if not isinstance(mode, str) or mode not in _ALLOWED_MODES:
                raise ValueError(f"Unsupported embedding mode: {mode!r}.")
            if mode in cleaned_modes:
                raise ValueError(f"Duplicate embedding mode: {mode}.")
            cleaned_modes.append(mode)
        if not cleaned_modes:
            raise ValueError("At least one embedding mode is required.")
        object.__setattr__(self, "modes", tuple(cleaned_modes))

    def canonical_definition(self) -> dict[str, Any]:
        value = asdict(self)
        value["modes"] = list(self.modes)
        return value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_definition(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def format_query(self, text: str) -> str:
        return self.query_prefix + clean_text(text, "query text", maximum=100_000)

    def format_passage(self, text: str) -> str:
        return self.passage_prefix + clean_text(
            text, "passage text", maximum=5_000_000
        )


PROFILE_FIELDS = {item.name for item in fields(EmbeddingProfile)} - {"alias"}


__all__ = [
    "EmbeddingProfile",
    "PROFILE_FIELDS",
    "clean_text",
    "model_name",
    "optional_exact_int",
    "profile_alias",
]
