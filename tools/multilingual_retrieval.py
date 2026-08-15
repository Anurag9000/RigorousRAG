"""Multilingual/Indic query normalization and retrieval routing contracts.

Original-language evidence is never overwritten.  Normalized/transliterated/translated
queries are derived variants with explicit lineage; providers are optional and bounded.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

_MAX_QUERY = 20_000
_SCRIPT_RANGES = {
    "latin": ((0x0041, 0x024F),),
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "oriya": ((0x0B00, 0x0B7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF),),
}


def _text(value: Any, label: str, maximum: int = _MAX_QUERY) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFKC", _text(value, "text"))


def detect_scripts(value: str) -> Mapping[str, float]:
    text = normalize_unicode(value)
    counts = {name: 0 for name in _SCRIPT_RANGES}
    eligible = 0
    for character in text:
        codepoint = ord(character)
        if not character.isalpha():
            continue
        eligible += 1
        for name, ranges in _SCRIPT_RANGES.items():
            if any(start <= codepoint <= end for start, end in ranges):
                counts[name] += 1
                break
    if eligible == 0:
        return {"unknown": 1.0}
    result = {name: count / eligible for name, count in counts.items() if count}
    recognized = sum(result.values())
    if recognized < 1.0:
        result["other"] = max(0.0, 1.0 - recognized)
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


class TransliterationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...
    def transliterate(self, text: str, *, source_script: str, target_script: str) -> str: ...


class TranslationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...
    def translate(self, text: str, *, source_language: str, target_language: str) -> str: ...


@dataclass(frozen=True)
class LanguageProfile:
    language: str
    script: str
    sparse_analyzer: str
    embedding_profile: str
    cross_lingual: bool = False

    def __post_init__(self) -> None:
        for name, maximum in (("language", 32), ("script", 32), ("sparse_analyzer", 128), ("embedding_profile", 256)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum).lower())
        if not isinstance(self.cross_lingual, bool):
            raise ValueError("cross_lingual must be boolean")


@dataclass(frozen=True)
class MultilingualVariant:
    text: str
    language: str
    script: str
    derivation: str
    provider_id: str = "deterministic"
    parent_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "variant text"))
        object.__setattr__(self, "language", _text(self.language, "language", 32).lower())
        object.__setattr__(self, "script", _text(self.script, "script", 32).lower())
        derivation = _text(self.derivation, "derivation", 64).lower()
        if derivation not in {"original", "normalized", "transliteration", "translation"}:
            raise ValueError("unsupported multilingual derivation")
        object.__setattr__(self, "derivation", derivation)
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 256))
        if self.parent_sha256:
            digest = self.parent_sha256.lower().strip()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("parent_sha256 is invalid")
            object.__setattr__(self, "parent_sha256", digest)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MultilingualRoutePlan:
    original: MultilingualVariant
    variants: tuple[MultilingualVariant, ...]
    profiles: tuple[LanguageProfile, ...]
    preserve_original_citations: bool
    fingerprint: str


def build_multilingual_plan(
    query: str,
    *,
    source_language: str,
    profiles: Sequence[LanguageProfile],
    transliterator: TransliterationProvider | None = None,
    translation_provider: TranslationProvider | None = None,
    translation_targets: Sequence[str] = (),
    transliterate_to_latin: bool = True,
    max_variants: int = 8,
) -> MultilingualRoutePlan:
    original_text = _text(query, "query")
    if not profiles or len(profiles) > 32 or any(not isinstance(item, LanguageProfile) for item in profiles):
        raise ValueError("profiles are invalid")
    if not 1 <= max_variants <= 16:
        raise ValueError("max_variants is invalid")
    scripts = detect_scripts(original_text)
    primary_script = next(iter(scripts))
    language = _text(source_language, "source_language", 32).lower()
    original = MultilingualVariant(original_text, language, primary_script, "original")
    variants: list[MultilingualVariant] = [original]
    seen = {original_text.casefold()}

    normalized = normalize_unicode(original_text)
    if normalized.casefold() not in seen and len(variants) < max_variants:
        variants.append(MultilingualVariant(normalized, language, primary_script, "normalized", parent_sha256=original.fingerprint))
        seen.add(normalized.casefold())

    if transliterate_to_latin and transliterator is not None and primary_script not in {"latin", "unknown", "other"} and len(variants) < max_variants:
        try:
            rendered = _text(transliterator.transliterate(original_text, source_script=primary_script, target_script="latin"), "transliteration")
        except Exception:
            rendered = ""
        if rendered and rendered.casefold() not in seen:
            variants.append(MultilingualVariant(rendered, language, "latin", "transliteration", _text(transliterator.provider_id, "provider_id", 256), original.fingerprint))
            seen.add(rendered.casefold())

    if translation_provider is not None:
        for target in list(translation_targets)[:8]:
            if len(variants) >= max_variants:
                break
            target_language = _text(target, "target language", 32).lower()
            if target_language == language:
                continue
            try:
                rendered = _text(translation_provider.translate(original_text, source_language=language, target_language=target_language), "translation")
            except Exception:
                continue
            if rendered.casefold() in seen:
                continue
            translated_script = next(iter(detect_scripts(rendered)))
            variants.append(MultilingualVariant(rendered, target_language, translated_script, "translation", _text(translation_provider.provider_id, "provider_id", 256), original.fingerprint))
            seen.add(rendered.casefold())

    payload = {
        "original": original.fingerprint,
        "variants": [item.fingerprint for item in variants],
        "profiles": [asdict(item) for item in profiles],
        "preserve_original_citations": True,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return MultilingualRoutePlan(original, tuple(variants), tuple(profiles), True, fingerprint)


__all__ = [
    "LanguageProfile", "MultilingualRoutePlan", "MultilingualVariant", "TranslationProvider",
    "TransliterationProvider", "build_multilingual_plan", "detect_scripts", "normalize_unicode",
]
