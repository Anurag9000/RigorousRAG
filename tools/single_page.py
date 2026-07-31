"""Bounded public webpage extraction."""

from __future__ import annotations

import operator
import os
from typing import Any, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from tools.privacy import mask_metadata_text
from tools.security import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_REQUEST_TIMEOUT,
    safe_download,
)

_MAX_PAGE_BYTES = DEFAULT_MAX_DOWNLOAD_BYTES
_MAX_USER_AGENT_CHARS = 1200
_DEFAULT_CONTACT_URL = os.getenv(
    "CRAWLER_CONTACT_URL",
    "https://github.com/Anurag9000/RigorousRAG",
)
DEFAULT_USER_AGENT = (
    "RigorousRAGBot/3.0 "
    f"(+{_DEFAULT_CONTACT_URL})"
)[:_MAX_USER_AGENT_CHARS]
ALLOWED_PAGE_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}


class PageContent(BaseModel):
    url: str = Field(max_length=4096)
    title: str = Field(max_length=500)
    text: str = Field(max_length=100_000)
    content_length: int = Field(ge=0)
    error: Optional[str] = Field(default=None, max_length=500)


def _safe_text(value: Any, *, maximum: int, default: str = "") -> str:
    try:
        rendered = str(value if value is not None else default)
    except Exception:
        rendered = default
    return rendered[:maximum]


def _public_display_url(value: Any) -> str:
    rendered = _safe_text(value, maximum=4096).strip()
    return mask_metadata_text(rendered)[:4096]


def _bounded_page_bytes(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("max_bytes must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_bytes must be an integer.") from exc
    limit = int(parsed)
    if limit <= 0:
        raise ValueError("max_bytes must be positive.")
    return min(limit, _MAX_PAGE_BYTES)


def _user_agent(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("user_agent must be a string.")
    without_controls = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in value
    )
    cleaned = " ".join(without_controls.split())
    if not cleaned:
        raise ValueError("user_agent is required.")
    return cleaned[:_MAX_USER_AGENT_CHARS]


def fetch_single_page(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> PageContent:
    """Fetch a public page without permitting internal-network access."""

    page_limit = _bounded_page_bytes(max_bytes)
    safe_display_url = _public_display_url(url)
    try:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")
        agent = _user_agent(user_agent)
        downloaded = safe_download(
            url,
            headers={
                "User-Agent": agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
            },
            timeout=DEFAULT_REQUEST_TIMEOUT,
            max_bytes=page_limit,
            allowed_content_types=ALLOWED_PAGE_CONTENT_TYPES,
        )
        content_type = downloaded.headers.get("Content-Type", "").lower()
        encoding = "utf-8"
        if "charset=" in content_type:
            encoding = content_type.split("charset=", 1)[1].split(";", 1)[0].strip()
        try:
            decoded = downloaded.content.decode(encoding, errors="replace")
        except LookupError:
            decoded = downloaded.content.decode("utf-8", errors="replace")

        if content_type.startswith("text/plain"):
            text = " ".join(decoded.split())
            title = downloaded.final_url
        else:
            soup = BeautifulSoup(decoded, "html.parser")
            for element in soup(
                ["script", "style", "noscript", "header", "footer", "nav", "aside", "svg"]
            ):
                element.decompose()
            title = "Untitled"
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            else:
                heading = soup.find(["h1", "h2"])
                if heading:
                    title = heading.get_text(" ", strip=True)
            text = " ".join(soup.get_text(separator=" ", strip=True).split())

        return PageContent(
            url=_public_display_url(downloaded.final_url),
            title=mask_metadata_text(_safe_text(title, maximum=500).strip()) or "Untitled",
            text=mask_metadata_text(text[:100_000]),
            content_length=len(downloaded.content),
        )
    except Exception as exc:
        return PageContent(
            url=safe_display_url,
            title="Error",
            text="",
            content_length=0,
            error=f"Page fetch failed ({type(exc).__name__}).",
        )
