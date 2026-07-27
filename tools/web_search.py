"""Serper-backed web search with strict result-host filtering."""

from __future__ import annotations

import os
from typing import List, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from tools.models import Citation
from tools.security import hostname_matches, validate_public_url


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    allowed_domains: Optional[List[str]] = Field(
        default=None,
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
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise WebSearchError("Web search is unavailable because SERPER_API_KEY is not configured.")
    limit = max(1, min(int(limit), 10))
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"q": query, "num": limit},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WebSearchError("The configured web-search provider request failed.") from exc
    except ValueError as exc:
        raise WebSearchError("The web-search provider returned invalid JSON.") from exc

    citations: List[Citation] = []
    for result in payload.get("organic", []) or []:
        link = str(result.get("link") or "").strip()
        if not link:
            continue
        try:
            public_url = validate_public_url(link)
        except Exception:
            continue
        hostname = urlparse(public_url).hostname or ""
        if allowed_domains and not hostname_matches(hostname, allowed_domains):
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
