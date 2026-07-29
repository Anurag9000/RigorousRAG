"""Semantic Scholar search adapter with bounded strict provider handling."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

from pydantic import BaseModel, Field

from tools.models import Citation
from tools.security import safe_download

_SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_MAX_RESPONSE_BYTES = 5_000_000
_MAX_PROVIDER_CANDIDATES = 50
_MAX_AUTHORS = 100


class AcademicSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    year_from: Optional[int] = Field(default=None, ge=0, le=9999)
    year_to: Optional[int] = Field(default=None, ge=0, le=9999)


ACADEMIC_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "academic_search",
        "description": "Search scholarly literature metadata through Semantic Scholar.",
        "parameters": AcademicSearchInput.model_json_schema(),
    },
}


class AcademicSearchError(RuntimeError):
    pass


def _bounded_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("Academic-search queries must be strings.")
    value = query.strip()
    if not value:
        return ""
    if len(value) > 2000 or "\x00" in value:
        raise ValueError("Academic-search queries may contain at most 2,000 valid characters.")
    return value


def _bounded_year(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        year = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not 0 <= year <= 9999:
        raise ValueError(f"{label} must be between 0 and 9999.")
    return year


def _bounded_limit(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("limit must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("limit must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("limit must be an integer.")
    if not 1 <= parsed <= 10:
        raise ValueError("limit must be between 1 and 10.")
    return parsed


def _provider_key() -> Optional[str]:
    raw = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    if not raw:
        return None
    if (
        len(raw) > 4096
        or any(character in raw for character in ("\x00", "\r", "\n"))
    ):
        raise AcademicSearchError("The configured scholarly-search provider key is invalid.")
    return raw


def _strict_provider_json(content: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise AcademicSearchError(
            "The scholarly-search provider returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise AcademicSearchError(
            "The scholarly-search provider returned an invalid result structure."
        )
    return payload


def _authors(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    names: List[str] = []
    for author in value[:_MAX_AUTHORS]:
        if not isinstance(author, dict):
            continue
        name = author.get("name")
        if isinstance(name, str):
            bounded = name.strip()[:300]
            if bounded:
                names.append(bounded)
    return ", ".join(names)[:3000]


def _metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    metadata: Dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= 100 or not isinstance(key, str):
            break
        if isinstance(item, bool) or item is None or isinstance(item, int):
            metadata[key[:200]] = item
        elif isinstance(item, str):
            metadata[key[:200]] = item[:1000]
    return metadata


def academic_search(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    *,
    limit: int = 5,
) -> List[Citation]:
    bounded_query = _bounded_query(query)
    if not bounded_query:
        return []
    requested = _bounded_limit(limit)
    start_year = _bounded_year(year_from, "year_from")
    end_year = _bounded_year(year_to, "year_to")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("year_from cannot be greater than year_to.")

    params = {
        "query": bounded_query,
        "limit": str(requested),
        "fields": "title,abstract,authors,year,venue,url,externalIds,paperId",
    }
    if start_year is not None or end_year is not None:
        params["year"] = f"{start_year or ''}-{end_year or ''}"
    headers = {"Accept": "application/json"}
    api_key = _provider_key()
    if api_key:
        headers["x-api-key"] = api_key
    try:
        downloaded = safe_download(
            f"{_SEMANTIC_SCHOLAR_ENDPOINT}?{urlencode(params)}",
            headers=headers,
            timeout=15,
            max_bytes=_MAX_RESPONSE_BYTES,
            allowed_content_types={"application/json"},
        )
        payload = _strict_provider_json(downloaded.content)
    except AcademicSearchError:
        raise
    except Exception as exc:
        raise AcademicSearchError(
            "The scholarly-search provider request failed."
        ) from exc

    papers = payload.get("data", [])
    if not isinstance(papers, list):
        raise AcademicSearchError(
            "The scholarly-search provider returned an invalid result structure."
        )
    citations: List[Citation] = []
    for paper in papers[:_MAX_PROVIDER_CANDIDATES]:
        if not isinstance(paper, dict):
            continue
        title = paper.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        year = paper.get("year")
        if isinstance(year, bool) or not isinstance(year, int) or not 0 <= year <= 9999:
            year = None
        if start_year is not None and (year is None or year < start_year):
            continue
        if end_year is not None and (year is None or year > end_year):
            continue
        paper_id = paper.get("paperId")
        raw_url = paper.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            url = raw_url.strip()[:4096]
        elif isinstance(paper_id, str) and paper_id.strip():
            url = (
                "https://www.semanticscholar.org/paper/"
                + quote(paper_id.strip()[:500], safe="")
            )
        else:
            continue
        abstract = paper.get("abstract")
        snippet = abstract.strip()[:4000] if isinstance(abstract, str) else None
        venue = paper.get("venue")
        venue_text = venue.strip()[:500] if isinstance(venue, str) else None
        metadata = {
            "year": year,
            "venue": venue_text,
            "authors": _authors(paper.get("authors")),
            "external_ids": _metadata(paper.get("externalIds")),
        }
        try:
            citations.append(
                Citation(
                    label=f"[{len(citations) + 1}]",
                    title=title.strip()[:500],
                    url=url,
                    source_type="web_search",
                    snippet=snippet or None,
                    source_id=(paper_id.strip()[:500] if isinstance(paper_id, str) else url),
                    metadata=metadata,
                )
            )
        except Exception:
            continue
        if len(citations) >= requested:
            break
    return citations
