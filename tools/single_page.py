"""Bounded public webpage extraction."""

from __future__ import annotations

import os
from typing import Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from tools.privacy import mask_metadata_text
from tools.security import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    DEFAULT_REQUEST_TIMEOUT,
    safe_download,
)

DEFAULT_USER_AGENT = (
    "RigorousRAGBot/3.0 "
    f"(+{os.getenv('CRAWLER_CONTACT_URL', 'https://github.com/Anurag9000/RigorousRAG')})"
)
ALLOWED_PAGE_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}


class PageContent(BaseModel):
    url: str = Field(max_length=4096)
    title: str = Field(max_length=500)
    text: str = Field(max_length=100_000)
    content_length: int = Field(ge=0)
    error: Optional[str] = Field(default=None, max_length=500)


def _public_display_url(value: str) -> str:
    return mask_metadata_text((value or "").strip())[:4096]


def fetch_single_page(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    *,
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> PageContent:
    """Fetch a public page without permitting internal-network access."""

    safe_display_url = _public_display_url(url)
    try:
        downloaded = safe_download(
            url,
            headers={
                "User-Agent": user_agent[:500],
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
            },
            timeout=DEFAULT_REQUEST_TIMEOUT,
            max_bytes=max_bytes,
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
            title=title[:500] or "Untitled",
            text=text[:100_000],
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
