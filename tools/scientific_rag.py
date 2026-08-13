"""Scientific-document normalization helpers for equations, sections, and abbreviations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


_EQUATION = re.compile(r"(?:\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\])", re.DOTALL)
_ABBREVIATION = re.compile(
    r"\b([A-Za-z][A-Za-z\- ]{2,60}?)\s*\(([A-Z][A-Z0-9-]{1,12})\)"
)
_SECTION_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(.+?)\s*$")


@dataclass(frozen=True)
class EquationSpan:
    text: str
    start: int
    end: int


def extract_equations(text: str) -> Tuple[EquationSpan, ...]:
    return tuple(
        EquationSpan(match.group(0), match.start(), match.end())
        for match in _EQUATION.finditer(text or "")
    )


def extract_abbreviations(text: str) -> Mapping[str, str]:
    output: Dict[str, str] = {}
    for long_form, short_form in _ABBREVIATION.findall(text or ""):
        output.setdefault(short_form.strip(), " ".join(long_form.split()))
    return output


def normalize_section_heading(heading: str) -> str:
    match = _SECTION_PREFIX.match(heading or "")
    value = match.group(1) if match else heading
    return " ".join(value.strip().lower().split())


def canonical_section(heading: str) -> str:
    normalized = normalize_section_heading(heading)
    aliases = {
        "abstract": "abstract",
        "introduction": "introduction",
        "background": "background",
        "related work": "related_work",
        "materials and methods": "methods",
        "methods": "methods",
        "methodology": "methods",
        "results": "results",
        "discussion": "discussion",
        "results and discussion": "results_discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "references": "references",
        "bibliography": "references",
    }
    return aliases.get(normalized, normalized.replace(" ", "_"))


def normalize_unit_spacing(text: str) -> str:
    """Normalize common numeric/unit spacing without changing unit semantics."""

    return re.sub(
        r"(?<=\d)\s*(?=(?:kg|g|mg|m|cm|mm|km|s|ms|Hz|kHz|MHz|GHz|Pa|kPa|MPa|K|°C)\b)",
        " ",
        text or "",
    )
