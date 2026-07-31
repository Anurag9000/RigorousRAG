"""Strict embedding-profile registry, compatibility resolution and operator JSON."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

from tools.embedding_models import (
    PROFILE_FIELDS,
    EmbeddingProfile,
    model_name,
    profile_alias,
)
from tools.embedding_profiles import BUILTIN_PROFILES, MODEL_ALIASES

_MAX_CONFIG_BYTES = 1_000_000
_MAX_PROFILES = 256
_REQUIRED_FIELDS = {
    "model_name",
    "dimensions",
    "max_sequence_tokens",
    "query_prefix",
    "passage_prefix",
    "normalize_embeddings",
    "language",
    "domain",
    "modes",
    "license",
    "source_url",
    "notes",
    "schema_version",
    "requires_adapter",
}


def _strict_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_strict_object_pairs,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON constant: {constant}")
        ),
    )


def compatibility_profile(selected_model: str) -> EmbeddingProfile:
    model = model_name(selected_model)
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
    return EmbeddingProfile(
        alias=f"compat-{digest}",
        model_name=model,
        dimensions=None,
        max_sequence_tokens=None,
        normalize_embeddings=True,
        language="unknown",
        domain="operator-defined",
        modes=("dense",),
        license="operator-review-required",
        notes=(
            "Compatibility profile: dimensions, sequence budget and instructions "
            "are unknown."
        ),
    )


def load_operator_profiles(raw_json: str | None) -> dict[str, EmbeddingProfile]:
    if raw_json in (None, ""):
        return {}
    if not isinstance(raw_json, str):
        raise ValueError("EMBEDDING_PROFILES_JSON must be a string.")
    if len(raw_json.encode("utf-8", errors="strict")) > _MAX_CONFIG_BYTES:
        raise ValueError("EMBEDDING_PROFILES_JSON exceeds the byte limit.")
    parsed = _strict_json_loads(raw_json)
    if not isinstance(parsed, dict):
        raise ValueError("EMBEDDING_PROFILES_JSON must contain an object.")
    if len(parsed) > _MAX_PROFILES:
        raise ValueError(f"At most {_MAX_PROFILES} operator profiles are supported.")
    result: dict[str, EmbeddingProfile] = {}
    for raw_alias, definition in parsed.items():
        alias = profile_alias(raw_alias)
        if not isinstance(definition, dict):
            raise ValueError(f"Profile {alias!r} must be an object.")
        unknown = set(definition) - PROFILE_FIELDS
        missing = _REQUIRED_FIELDS - set(definition)
        if unknown:
            raise ValueError(
                f"Profile {alias!r} contains unknown fields: {sorted(unknown)}."
            )
        if missing:
            raise ValueError(
                f"Profile {alias!r} is missing fields: {sorted(missing)}."
            )
        result[alias] = EmbeddingProfile(alias=alias, **definition)
    return result


class EmbeddingProfileRegistry:
    """Profile lookup with explicit, fully validated operator overrides."""

    def __init__(
        self,
        profiles: Mapping[str, EmbeddingProfile] | None = None,
        *,
        operator_json: str | None = None,
    ) -> None:
        merged = dict(BUILTIN_PROFILES if profiles is None else profiles)
        for alias, profile in merged.items():
            if profile_alias(alias) != profile.alias:
                raise ValueError("Profile mapping keys must match profile aliases.")
        merged.update(load_operator_profiles(operator_json))
        self._profiles = merged
        self._model_aliases = dict(MODEL_ALIASES)
        for alias, profile in merged.items():
            existing = self._model_aliases.get(profile.model_name)
            if existing is not None and existing != alias:
                raise ValueError(
                    f"Model name {profile.model_name!r} is assigned to multiple profiles."
                )
            self._model_aliases[profile.model_name] = alias

    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def get(self, alias: str) -> EmbeddingProfile:
        identifier = profile_alias(alias)
        try:
            return self._profiles[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown embedding profile: {identifier}") from exc

    def resolve(
        self,
        value: str,
        *,
        allow_compatibility: bool = True,
    ) -> EmbeddingProfile:
        if not isinstance(value, str):
            raise ValueError("Embedding model/profile selection must be a string.")
        candidate = value.strip()
        if candidate in self._profiles:
            return self._profiles[candidate]
        alias = self._model_aliases.get(candidate)
        if alias and alias in self._profiles:
            return self._profiles[alias]
        if not allow_compatibility:
            raise KeyError(f"Unknown embedding profile or model: {candidate}")
        return compatibility_profile(candidate)


def registry_from_environment() -> EmbeddingProfileRegistry:
    return EmbeddingProfileRegistry(
        operator_json=os.getenv("EMBEDDING_PROFILES_JSON", "")
    )


def resolve_embedding_profile(
    value: str | None = None,
    *,
    registry: EmbeddingProfileRegistry | None = None,
    allow_compatibility: bool = True,
) -> EmbeddingProfile:
    selected = value
    if selected is None:
        selected = os.getenv(
            "EMBEDDING_PROFILE",
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
    return (registry or registry_from_environment()).resolve(
        selected,
        allow_compatibility=allow_compatibility,
    )


__all__ = [
    "BUILTIN_PROFILES",
    "MODEL_ALIASES",
    "EmbeddingProfile",
    "EmbeddingProfileRegistry",
    "compatibility_profile",
    "load_operator_profiles",
    "registry_from_environment",
    "resolve_embedding_profile",
]
