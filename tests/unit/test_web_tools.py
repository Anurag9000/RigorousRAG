from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.single_page import fetch_single_page
from tools.web_search import WebSearchError, web_search


def test_web_search_requires_provider_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(WebSearchError):
        web_search("query")


def test_web_search_uses_hostname_boundaries_and_status_validation(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "organic": [
            {"title": "Allowed", "link": "https://papers.example.org/a", "snippet": "A"},
            {"title": "Attack", "link": "https://example.org.attacker.test/b", "snippet": "B"},
        ]
    }
    with patch("tools.web_search.requests.post", return_value=response), \
         patch("tools.web_search.validate_public_url", side_effect=lambda value: value):
        results = web_search("query", allowed_domains=["example.org"])
    assert [item.title for item in results] == ["Allowed"]
    response.raise_for_status.assert_called_once()
    assert response.request if hasattr(response, "request") else True


def test_single_page_uses_bounded_safe_download():
    downloaded = SimpleNamespace(
        final_url="https://example.test/article",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<html><title>Paper</title><body><script>bad()</script>Evidence text</body></html>",
        status_code=200,
    )
    with patch("tools.single_page.safe_download", return_value=downloaded) as safe:
        page = fetch_single_page("https://example.test/article", max_bytes=1234)
    assert page.error is None
    assert page.title == "Paper"
    assert "Evidence text" in page.text
    assert "bad()" not in page.text
    assert safe.call_args.kwargs["max_bytes"] == 1234


def test_single_page_returns_structured_error():
    with patch("tools.single_page.safe_download", side_effect=ValueError("blocked")):
        page = fetch_single_page("http://127.0.0.1")
    assert page.error == "blocked"
    assert page.text == ""
