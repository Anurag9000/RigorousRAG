"""Validated BibTeX generation for common scholarly entry types."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List

from tools.privacy import mask_metadata_text

_ALLOWED_ENTRY_TYPES = {
    "article", "book", "incollection", "inproceedings", "mastersthesis",
    "misc", "phdthesis", "techreport",
}
_REQUIRED_FIELDS = {
    "article": {"journal"},
    "book": {"publisher"},
    "incollection": {"booktitle"},
    "inproceedings": {"booktitle"},
    "mastersthesis": {"school"},
    "phdthesis": {"school"},
    "techreport": {"institution"},
}
_SCALAR_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
BIBTEX_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "export_to_bibtex",
        "description": "Convert scholarly metadata into escaped BibTeX entries.",
        "parameters": {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array",
                    "maxItems": 100,
                    "items": {
                        "type": "object",
                        "properties": {
                            "entry_type": {"type": "string", "maxLength": 50},
                            "title": {"type": "string", "maxLength": 1000},
                            "authors": {"type": "string", "maxLength": 3000},
                            "year": _SCALAR_SCHEMA,
                            "journal": {"type": "string", "maxLength": 1000},
                            "booktitle": {"type": "string", "maxLength": 1000},
                            "publisher": {"type": "string", "maxLength": 1000},
                            "institution": {"type": "string", "maxLength": 1000},
                            "school": {"type": "string", "maxLength": 1000},
                            "volume": _SCALAR_SCHEMA,
                            "number": _SCALAR_SCHEMA,
                            "pages": {"type": "string", "maxLength": 200},
                            "doi": {"type": "string", "maxLength": 500},
                            "url": {"type": "string", "maxLength": 4096},
                        },
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["citations"],
            "additionalProperties": False,
        },
    },
}
_FIELD_ORDER = [
    "title", "author", "year", "journal", "booktitle", "publisher",
    "institution", "school", "volume", "number", "pages", "doi", "url",
]
_FIELD_LIMITS = {
    "title": 1000,
    "authors": 3000,
    "author": 3000,
    "year": 100,
    "journal": 1000,
    "booktitle": 1000,
    "publisher": 1000,
    "institution": 1000,
    "school": 1000,
    "volume": 100,
    "number": 100,
    "pages": 200,
    "doi": 500,
    "url": 4096,
    "entry_type": 50,
}
_MAX_OUTPUT_ENTRIES = 100
_MAX_INSPECTED_CANDIDATES = 1000
_MAX_OUTPUT_CHARS = 500_000
_MAX_KEY_YEAR_DIGITS = 8
_BIBTEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "%": r"\%",
    "#": r"\#",
    "$": r"\$",
    "&": r"\&",
    "_": r"\_",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _bounded_scalar(value: Any, field: str) -> str:
    limit = _FIELD_LIMITS.get(field, 1000)
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        rendered = str(value)
    elif isinstance(value, str):
        rendered = value
    else:
        return ""
    # Bound attacker-controlled text before privacy regexes and escaping. Apply
    # the output bound again because redaction placeholders can expand text.
    return mask_metadata_text(rendered[:limit])[:limit]


def _escape_bibtex(value: Any) -> str:
    """Normalize controls and escape each original character exactly once."""

    if not isinstance(value, str):
        return ""
    without_controls = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in value
    )
    text = " ".join(without_controls.split())
    return "".join(_BIBTEX_ESCAPES.get(character, character) for character in text)


def _slug(value: str, limit: int = 28) -> str:
    if not isinstance(value, str):
        return "reference"
    return "".join(re.findall(r"[A-Za-z0-9]+", value))[:limit] or "reference"


def _value(citation: Dict[str, Any], field: str) -> Any:
    try:
        return citation.get(field)
    except Exception:
        return None


def _first_nonempty(citation: Dict[str, Any], *fields: str) -> str:
    for field in fields:
        value = _bounded_scalar(_value(citation, field), field)
        if value.strip():
            return value
    return ""


def _citation_key(citation: Dict[str, Any], index: int) -> str:
    authors = _first_nonempty(citation, "authors", "author")
    first_author = re.split(r"\s+and\s+|,|;", authors, maxsplit=1, flags=re.I)[0]
    surname = first_author.strip().split()[-1] if first_author.strip() else "anon"
    year_value = _bounded_scalar(_value(citation, "year"), "year") or "nd"
    year_digits = re.sub(r"\D", "", year_value)[:_MAX_KEY_YEAR_DIGITS]
    year = year_digits or "nd"
    title = _bounded_scalar(_value(citation, "title"), "title") or "untitled"
    title_slug = _slug(title, 18)
    identity = "|".join(
        _bounded_scalar(_value(citation, field), field)
        for field in ("title", "authors", "year", "doi", "url")
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:6]
    return f"{_slug(surname, 16)}{year}{title_slug}{digest}" or f"reference{index + 1}"


def _normalise_entry(citation: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    raw_entry_type = _bounded_scalar(
        _value(citation, "entry_type"),
        "entry_type",
    )
    entry_type = (raw_entry_type or "article").lower().strip()
    if entry_type not in _ALLOWED_ENTRY_TYPES:
        entry_type = "misc"
    fields: Dict[str, str] = {}
    mappings = {
        "title": "title", "authors": "author", "author": "author", "year": "year",
        "journal": "journal", "booktitle": "booktitle", "publisher": "publisher",
        "institution": "institution", "school": "school", "volume": "volume",
        "number": "number", "pages": "pages", "doi": "doi", "url": "url",
    }
    for source, destination in mappings.items():
        if destination in fields:
            continue
        bounded = _bounded_scalar(_value(citation, source), source)
        if bounded.strip():
            fields[destination] = _escape_bibtex(bounded)
    fields.setdefault("title", "Untitled")
    fields.setdefault("author", "Unknown")
    fields.setdefault("year", "n.d.")
    if not _REQUIRED_FIELDS.get(entry_type, set()).issubset(fields):
        entry_type = "misc"
    return entry_type, fields


def export_to_bibtex(citations: Iterable[Dict[str, Any]]) -> str:
    if isinstance(citations, (str, bytes, bytearray)):
        raise ValueError("citations must be an iterable of metadata objects, not text.")
    try:
        iterator = iter(citations)
    except TypeError as exc:
        raise ValueError("citations must be an iterable of metadata objects.") from exc

    entries: List[str] = []
    used_keys: set[str] = set()
    output_chars = 0
    for _ in range(_MAX_INSPECTED_CANDIDATES):
        if len(entries) >= _MAX_OUTPUT_ENTRIES:
            break
        try:
            citation = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise ValueError("citations iteration failed.") from exc
        if not isinstance(citation, dict):
            continue
        entry_type, fields = _normalise_entry(citation)
        key = _citation_key(citation, len(entries))
        base_key = key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}{suffix}"
            suffix += 1
        used_keys.add(key)
        ordered = [field for field in _FIELD_ORDER if field in fields]
        lines = [f"@{entry_type}{{{key},"]
        for position, field in enumerate(ordered):
            comma = "," if position < len(ordered) - 1 else ""
            lines.append(f"  {field} = {{{fields[field]}}}{comma}")
        lines.append("}")
        entry = "\n".join(lines)
        addition = len(entry) + (2 if entries else 0)
        if output_chars + addition > _MAX_OUTPUT_CHARS:
            break
        entries.append(entry)
        output_chars += addition
    return "\n\n".join(entries)
