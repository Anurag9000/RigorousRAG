"""Unicode-script signals for multilingual and code-switched retrieval routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SCRIPT_RANGES = (
    ("latin", ((0x0041, 0x024F),)),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("cyrillic", ((0x0400, 0x052F),)),
    ("han", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))),
)


def _script(character: str) -> str | None:
    code = ord(character)
    for name, ranges in _SCRIPT_RANGES:
        if any(start <= code <= end for start, end in ranges):
            return name
    return None


@dataclass(frozen=True)
class MultilingualRouteSignal:
    scripts: tuple[str, ...]
    script_fractions: tuple[tuple[str, float], ...]
    code_switched: bool
    multilingual_model_required: bool
    lexical_fallback_recommended: bool


def analyze_multilingual_query(text: Any, *, minimum_script_fraction: float = 0.10) -> MultilingualRouteSignal:
    """Classify meaningful Unicode scripts without claiming language identification."""

    if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
        raise ValueError("text must be a non-empty bounded string.")
    if isinstance(minimum_script_fraction, bool):
        raise ValueError("minimum_script_fraction must be between 0 and 1.")
    threshold = float(minimum_script_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_script_fraction must be between 0 and 1.")
    counts: dict[str, int] = {}
    total = 0
    for character in text:
        script = _script(character)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
            total += 1
    if total == 0:
        return MultilingualRouteSignal((), (), False, False, True)
    fractions = tuple(
        sorted(
            ((script, count / total) for script, count in counts.items()),
            key=lambda row: (-row[1], row[0]),
        )
    )
    meaningful = tuple(script for script, fraction in fractions if fraction >= threshold)
    code_switched = len(meaningful) >= 2
    non_latin = any(script != "latin" for script in meaningful)
    return MultilingualRouteSignal(
        scripts=meaningful,
        script_fractions=fractions,
        code_switched=code_switched,
        multilingual_model_required=non_latin or code_switched,
        lexical_fallback_recommended=code_switched or len(meaningful) == 0,
    )


__all__ = ["MultilingualRouteSignal", "analyze_multilingual_query"]
