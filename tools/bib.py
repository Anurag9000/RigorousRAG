"""Validated BibTeX generation for common scholarly entry types."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List

_ALLOWED_ENTRY_TYPES = {
    "article",
    "book",
    "incollection",
    "inproceedings",
    "mastersthesis",
    "misc",
    "phdthesis",
    "techreport",
}

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
                            "entry_type": {"type": "string"},
                            "title": {"type": "string"},
                            "authors": {"type": "string"},
                            "year": {"type": ["string", "integer"]},
                            "journal": {"type": "string"},
                            "booktitle": {"type": "string"},
                            "publisher": {"type": "string"},
                            "institution": {"type": "string"},
                            "school": {"type": "string"},
                            "volume": {"type": ["string", "integer"]},
                            "number": {"type": ["string", "integer"]},
                            "pages": {"type": "string"},
                            "doi": {"type": "string"},
                            "url": {"type": "string"},
                        },
                        "required": ["title"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["citations"],
            "additionalProperties": False,
        },
    },
}

_FIELD_ORDER = [
    "title",
    "author",
    "year",
    "journal",
    "booktitle",
    "publisher",
    "institution",
    "school",
    "volume",
    "number",
    "pages",
    "doi",
    "url",
]


def _escape_bibtex(value: Any) -> str:
    """Escape syntax-significant characters without interpreting LaTeX."""

    text = " ".join(str(value or "").replace("\x00", "").split())
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("{", r"\{").replace("}", r"\}")
    text = text.replace("%", r"\%").replace("#", r"\#")
    text = text.replace("&", r"\&").replace("_", r"\_")
    return text


def _slug(value: str, limit: int = 28) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value or "")
    return "".join(words)[:limit] or "reference"


def _citation_key(citation: Dict[str, Any], index: int) -> str:
    authors = str(citation.get("authors") or citation.get("author") or "")
    first_author = re.split(r"\s+and\s+|,|;", authors, maxsplit=1, flags=re.I)[0]
    surname = first_author.strip().split()[-1] if first_author.strip() else "anon"
    year = re.sub(r"\D", "", str(citation.get("year") or "nd")) or "nd"
    title_slug = _slug(str(citation.get("title") or "untitled"), limit=18)
    identity = "|".join(
        str(citation.get(field) or "")
        for field in ("title", "authors", "year", "doi", "url")
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:6]
    key = f"{_slug(surname, 16)}{year}{title_slug}{digest}"
    return key or f"reference{index + 1}"


def _normalise_entry(citation: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    entry_type = str(citation.get("entry_type") or "article").lower().strip()
    if entry_type not in _ALLOWED_ENTRY_TYPES:
        entry_type = "misc"
    fields: Dict[str, str] = {}
    mappings = {
        "title": "title",
        "authors": "author",
        "author": "author",
        "year": "year",
        "journal": "journal",
        "booktitle": "booktitle",
        "publisher": "publisher",
        "institution": "institution",
        "school": "school",
        "volume": "volume",
        "number": "number",
        "pages": "pages",
        "doi": "doi",
        "url": "url",
    }
    for source, destination in mappings.items():
        value = citation.get(source)
        if value not in (None, "") and destination not in fields:
            fields[destination] = _escape_bibtex(value)
    fields.setdefault("title", "Untitled")
    fields.setdefault("author", "Unknown")
    fields.setdefault("year", "n.d.")
    return entry_type, fields


def export_to_bibtex(citations: Iterable[Dict[str, Any]]) -> str:
    """Generate deterministic, escaped BibTeX entries."""

    entries: List[str] = []
    used_keys: set[str] = set()
    for index, raw in enumerate(citations):
        citation = dict(raw or {})
        entry_type, fields = _normalise_entry(citation)
        key = _citation_key(citation, index)
        base_key = key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}{suffix}"
            suffix += 1
        used_keys.add(key)
        lines = [f"@{entry_type}{{{key},"]
        ordered = [field for field in _FIELD_ORDER if field in fields]
        ordered.extend(sorted(set(fields) - set(ordered)))
        for position, field in enumerate(ordered):
            comma = "," if position < len(ordered) - 1 else ""
            lines.append(f"  {field} = {{{fields[field]}}}{comma}")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries)
