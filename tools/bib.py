"""Validated BibTeX generation for common scholarly entry types."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List

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


def _escape_bibtex(value: Any) -> str:
    """Escape each original character once, avoiding cascading replacements."""

    text = " ".join(str(value or "").replace("\x00", "").split())
    return "".join(_BIBTEX_ESCAPES.get(character, character) for character in text)


def _slug(value: str, limit: int = 28) -> str:
    return "".join(re.findall(r"[A-Za-z0-9]+", value or ""))[:limit] or "reference"


def _citation_key(citation: Dict[str, Any], index: int) -> str:
    authors = str(citation.get("authors") or citation.get("author") or "")
    first_author = re.split(r"\s+and\s+|,|;", authors, maxsplit=1, flags=re.I)[0]
    surname = first_author.strip().split()[-1] if first_author.strip() else "anon"
    year = re.sub(r"\D", "", str(citation.get("year") or "nd")) or "nd"
    title_slug = _slug(str(citation.get("title") or "untitled"), 18)
    identity = "|".join(
        str(citation.get(field) or "")
        for field in ("title", "authors", "year", "doi", "url")
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:6]
    return f"{_slug(surname, 16)}{year}{title_slug}{digest}" or f"reference{index + 1}"


def _normalise_entry(citation: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    entry_type = str(citation.get("entry_type") or "article").lower().strip()
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
        value = citation.get(source)
        if value not in (None, "") and destination not in fields:
            fields[destination] = _escape_bibtex(value)
    fields.setdefault("title", "Untitled")
    fields.setdefault("author", "Unknown")
    fields.setdefault("year", "n.d.")
    if not _REQUIRED_FIELDS.get(entry_type, set()).issubset(fields):
        entry_type = "misc"
    return entry_type, fields


def export_to_bibtex(citations: Iterable[Dict[str, Any]]) -> str:
    entries: List[str] = []
    used_keys: set[str] = set()
    for raw in citations:
        if len(entries) >= 100:
            break
        if not isinstance(raw, dict):
            continue
        citation = dict(raw)
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
        entries.append("\n".join(lines))
    return "\n\n".join(entries)
