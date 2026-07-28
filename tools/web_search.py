"""Serper-backed web search with strict network and result-host boundaries."""

from __future__ import annotations

import json
import os
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from tools.models import Citation
from tools.security import hostname_matches, safe_download, validate_public_url

_SERPER_ENDPOINT = "https://google.serper.dev/search"
_SERPER_MAX_RESPONSE_BYTES = max(
    10_000,
    min(int(os.getenv("SERPER_MAX_RESPONSE_BYTES", "2000000")), 20_000_000),
)


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    allowed_domains: Optional[List[str]] = Field(
        default=None,
        max_length=50,
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


def web_search(
    query: str,
    allowed_domains: Optional[List[str]] = None,
    *,
    limit: int = 5,
) -> List[Citation]:
    query = (query or "").strip()
    if not query:
        return []
    if len(query) > 2000:
        raise ValueError("Web-search queries may contain at most 2,000 characters.")
    domains = [str(value).strip()[:253] for value in (allowed_domains or []) if str(value).strip()]
    if len(domains) > 50:
        raise ValueError("At most 50 allowed domains may be supplied.")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise WebSearchError("Web search is unavailable because SERPER_API_KEY is not configured.")
    limit = max(1, min(int(limit), 10))
    try:
        downloaded = safe_download(
            _SERPER_ENDPOINT,
            method="POST",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json_body={"q": query, "num": limit},
            timeout=15,
            max_bytes=_SERPER_MAX_RESPONSE_BYTES,
            allowed_content_types={"application/json"},
        )
        payload = json.loads(downloaded.content.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Provider response must be a JSON object.")
    except Exception as exc:
        raise WebSearchError("The configured web-search provider request failed.") from exc

    organic = payload.get("organic", [])
    if not isinstance(organic, list):
        raise WebSearchError("The web-search provider returned an invalid result structure.")
    citations: List[Citation] = []
    for result in organic[:100]:
        if not isinstance(result, dict):
            continue
        link = str(result.get("link") or "").strip()
        if not link:
            continue
        try:
            public_url = validate_public_url(link)
        except Exception:
            continue
        hostname = urlparse(public_url).hostname or ""
        if domains and not hostname_matches(hostname, domains):
            continue
        citations.append(
            Citation(
                label=f"[{len(citations) + 1}]",
                title=str(result.get("title") or "Untitled result")[:500],
                url=public_url,
                source_type="web_search",
                snippet=str(result.get("snippet") or "")[:4000] or None,
                source_id=public_url,
            )
        )
        if len(citations) >= limit:
            break
    return citations
