from __future__ import annotations

import re
from pathlib import Path


def _frontend_source() -> str:
    path = Path(__file__).resolve().parents[2] / "frontend" / "app.js"
    return path.read_text(encoding="utf-8")


def _function_body(source: str, name: str, next_name: str) -> str:
    pattern = re.compile(
        rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{(.*?)"
        rf"(?=\nfunction\s+{re.escape(next_name)}\s*\()",
        re.DOTALL,
    )
    match = pattern.search(source)
    assert match is not None, f"{name} function was not found"
    return match.group(1)


def test_citation_cards_use_safe_dom_construction_for_graph_fields():
    source = _frontend_source()
    body = _function_body(source, "updateCitations", "setBusy")

    assert "innerHTML" not in body
    assert "insertAdjacentHTML" not in body
    assert "document.createElement" in body
    assert "textElement" in body
    assert "citation.label" in body
    assert "citation.title" in body
    assert "citation.source_type" in body
    assert "citation.page_number" in body
    assert "citation.chunk_id" in body
    assert "citation.quote || citation.snippet" in body


def test_local_graph_urls_remain_inert_and_external_links_are_allowlisted():
    source = _frontend_source()
    url_body = _function_body(source, "safeExternalUrl", "appendInlineMarkdown")
    citation_body = _function_body(source, "updateCitations", "setBusy")

    assert 'url.protocol === "http:"' in url_body
    assert 'url.protocol === "https:"' in url_body
    assert "local:" not in url_body
    assert "safeExternalUrl(citation.url)" in citation_body
    assert "citation.url || \"Local evidence\"" in citation_body
    assert 'link.rel = "noopener noreferrer"' in citation_body
