"""Serper-backed web search with strict network and result-host boundaries."""

from __future__ import annotations

import itertools
import json
import operator
import os
import re
from typing import Any, Iterable, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from tools.config import bounded_int_env
from tools.models import Citation
from tools.security import hostname_matches, safe_download, validate_public_url

_SERPER_ENDPOINT = "https://google.serper.dev/search"
_SERPER_MAX_RESPONSE_BYTES = bounded_int_env(
    "SERPER_MAX_RESPONSE_BYTES",
    2_000_000,
    minimum=10_000,
    maximum=20_000_000,
)
_MAX_RESULT_CANDIDATES = bounded_int_env(
    "WEB_SEARCH_MAX_RESULT_CANDIDATES",
    30,
    minimum=10,
    maximum=100,
)
_MAX_ALLOWED_DOMAINS = 50
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class WebSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        max_length=_MAX_ALLOWED_DOMAINS,
        description="Optional exact hostnames or parent domains.",
    )


WEB_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web, optionally restricting results by hostname.",
        "parameters": WebSearchInput.model_json_schema(),
    },
}


class WebSearchError(RuntimeError):
    pass


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _bounded_limit(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("limit must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("limit must be an integer.") from exc
    limit = int(parsed)
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10.")
    return limit


def _bounded_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Web-search queries must be strings.")
    query = value.strip()
    if not query:
        return ""
    if len(query) > 2000 or _contains_ascii_control(query):
        raise ValueError("Web-search queries may contain at most 2,000 valid characters.")
    return query


def _canonical_domain(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Allowed domains must be strings.")
    rendered = value.strip().rstrip(".").lower()
    if (
        not rendered
        or len(rendered) > 253
        or _contains_ascii_control(rendered)
        or any(character.isspace() for character in rendered)
        or "\\" in rendered
    ):
        raise ValueError("Allowed domains must be valid hostnames.")
    try:
        parsed = urlparse(rendered if "://" in rendered else f"https://{rendered}")
        hostname = parsed.hostname or ""
        ascii_host = hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (ValueError, UnicodeError):
        raise ValueError("Allowed domains must be valid hostnames.")
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Allowed domains must contain hostnames only.")
    labels = ascii_host.split(".")
    if not labels or any(not _HOST_LABEL_RE.fullmatch(label) for label in labels):
        raise ValueError("Allowed domains must be valid hostnames.")
    return ascii_host


def _bounded_domains(values: Optional[Iterable[str]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("allowed_domains must be an array of hostnames.")
    try:
        raw_values = list(
            itertools.islice(iter(values), _MAX_ALLOWED_DOMAINS + 1)
        )
    except Exception as exc:
        raise ValueError("allowed_domains must be iterable.") from exc
    if len(raw_values) > _MAX_ALLOWED_DOMAINS:
        raise ValueError(
            f"At most {_MAX_ALLOWED_DOMAINS} allowed domains may be supplied."
        )
    domains: List[str] = []
    for raw_value in raw_values:
        domain = _canonical_domain(raw_value)
        if domain not in domains:
            domains.append(domain)
    return domains


def _provider_key() -> str:
    raw = os.getenv("SERPER_API_KEY", "")
    if not raw:
        raise WebSearchError(
            "Web search is unavailable because SERPER_API_KEY is not configured."
        )
    if (
        not isinstance(raw, str)
        or raw != raw.strip()
        or len(raw) > 4096
        or _contains_ascii_control(raw)
    ):
        raise WebSearchError("The configured web-search provider key is invalid.")
    return raw


def _strict_provider_json(content: bytes) -> dict[str, Any]:
    if not isinstance(content, bytes):
        raise WebSearchError(
            "The web-search provider returned invalid JSON."
        )
    try:
        decoded = content.decode("utf-8")
        payload = json.loads(
            decoded,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-standard JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise WebSearchError(
            "The web-search provider returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise WebSearchError(
            "The web-search provider returned an invalid result structure."
        )
    return payload


def web_search(
    query: str,
    allowed_domains: Optional[List[str]] = None,
    *,
    limit: int = 5,
) -> List[Citation]:
    bounded_query = _bounded_query(query)
    if not bounded_query:
        return []
    domains = _bounded_domains(allowed_domains)
    requested = _bounded_limit(limit)
    api_key = _provider_key()
    try:
        downloaded = safe_download(
            _SERPER_ENDPOINT,
            method="POST",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json_body={"q": bounded_query, "num": requested},
            timeout=15,
            max_bytes=_SERPER_MAX_RESPONSE_BYTES,
            allowed_content_types={"application/json"},
        )
        payload = _strict_provider_json(downloaded.content)
    except WebSearchError:
        raise
    except Exception as exc:
        raise WebSearchError(
            "The configured web-search provider request failed."
        ) from exc

    organic = payload.get("organic", [])
    if not isinstance(organic, list):
        raise WebSearchError(
            "The web-search provider returned an invalid result structure."
        )
    try:
        candidates = itertools.islice(iter(organic), _MAX_RESULT_CANDIDATES)
    except Exception as exc:
        raise WebSearchError(
            "The web-search provider returned an invalid result structure."
        ) from exc
    citations: List[Citation] = []
    try:
        for result in candidates:
            if not isinstance(result, dict):
                continue
            raw_link = result.get("link")
            if not isinstance(raw_link, str):
                continue
            link = raw_link.strip()
            if not link:
                continue
            try:
                parsed = urlparse(link)
                hostname = parsed.hostname or ""
            except ValueError:
                continue
            if domains and (not hostname or not hostname_matches(hostname, domains)):
                continue
            try:
                public_url = validate_public_url(link)
            except Exception:
                continue
            raw_title = result.get("title")
            title = raw_title.strip() if isinstance(raw_title, str) else "Untitled result"
            raw_snippet = result.get("snippet")
            snippet = raw_snippet.strip() if isinstance(raw_snippet, str) else ""
            try:
                citation = Citation(
                    label=f"[{len(citations) + 1}]",
                    title=title[:500] or "Untitled result",
                    url=public_url,
                    source_type="web_search",
                    snippet=snippet[:4000] or None,
                    source_id=public_url,
                )
            except Exception:
                continue
            citations.append(citation)
            if len(citations) >= requested:
                break
    except Exception as exc:
        raise WebSearchError(
            "The web-search provider returned an invalid result structure."
        ) from exc
    return citations