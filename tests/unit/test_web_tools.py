import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.single_page import fetch_single_page
from tools.web_search import WebSearchError, web_search


def test_web_search_requires_provider_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(WebSearchError):
        web_search("query")


def test_web_search_refuses_oversized_or_control_bearing_provider_key(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "x" * 4097)
    with pytest.raises(WebSearchError, match="key is invalid"):
        web_search("query")

    monkeypatch.setenv("SERPER_API_KEY", "key\r\nInjected: yes")
    with pytest.raises(WebSearchError, match="key is invalid"):
        web_search("query")


def test_web_search_uses_bounded_downloader_and_hostname_boundaries(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    downloaded = SimpleNamespace(
        content=json.dumps({
            "organic": [
                {
                    "title": "Allowed",
                    "link": "https://papers.example.org/a",
                    "snippet": "A",
                },
                {
                    "title": "Attack",
                    "link": "https://example.org.attacker.test/b",
                    "snippet": "B",
                },
            ]
        }).encode("utf-8"),
    )
    with patch("tools.web_search.safe_download", return_value=downloaded) as safe, \
         patch("tools.web_search.validate_public_url", side_effect=lambda value: value) as validate:
        results = web_search("query", allowed_domains=["example.org"])

    assert [item.title for item in results] == ["Allowed"]
    validate.assert_called_once_with("https://papers.example.org/a")
    kwargs = safe.call_args.kwargs
    assert kwargs["method"] == "POST"
    assert kwargs["headers"]["X-API-KEY"] == "test-key"
    assert kwargs["json_body"] == {"q": "query", "num": 5}
    assert kwargs["allowed_content_types"] == {"application/json"}
    assert kwargs["max_bytes"] > 0


def test_web_search_bounds_public_url_validation_candidates(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    downloaded = SimpleNamespace(
        content=json.dumps({
            "organic": [
                {
                    "title": f"Result {index}",
                    "link": f"https://candidate-{index}.example.test/result",
                }
                for index in range(100)
            ]
        }).encode("utf-8"),
    )
    with patch("tools.web_search.safe_download", return_value=downloaded), \
         patch("tools.web_search._MAX_RESULT_CANDIDATES", 7), \
         patch("tools.web_search.validate_public_url", side_effect=ValueError("invalid")) as validate:
        results = web_search("query", limit=10)

    assert results == []
    assert validate.call_count == 7


def test_web_search_distinguishes_invalid_provider_json_from_request_failure(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    for payload in (b"not-json", b'{"score":NaN}', b"[]", b"\xff"):
        with patch(
            "tools.web_search.safe_download",
            return_value=SimpleNamespace(content=payload),
        ):
            with pytest.raises(WebSearchError, match="invalid JSON|invalid result structure"):
                web_search("query")

    with patch(
        "tools.web_search.safe_download",
        side_effect=RuntimeError("secret provider details"),
    ):
        with pytest.raises(WebSearchError) as captured:
            web_search("query")
    assert "secret provider details" not in str(captured.value)


def test_web_search_validates_all_direct_inputs_before_network(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    network = MagicMock(side_effect=AssertionError("network should not run"))
    with patch("tools.web_search.safe_download", network):
        cases = [
            {"query": object()},
            {"query": "q" * 2001},
            {"query": "query", "allowed_domains": "example.org"},
            {"query": "query", "allowed_domains": [object()]},
            {"query": "query", "allowed_domains": ["https://example.org/path"]},
            {"query": "query", "allowed_domains": ["https://example.org?x=1"]},
            {"query": "query", "allowed_domains": ["https://example.org:443"]},
            {"query": "query", "allowed_domains": ["ftp://example.org"]},
            {"query": "query", "limit": "not-an-integer"},
            {"query": "query", "limit": True},
            {"query": "query", "limit": 1.5},
            {"query": "query", "limit": 0},
            {"query": "query", "limit": 11},
        ]
        for arguments in cases:
            with pytest.raises(ValueError):
                web_search(**arguments)

        with pytest.raises(ValueError, match="50"):
            web_search(
                "query",
                allowed_domains=[f"{index}.example" for index in range(51)],
            )
    network.assert_not_called()


def test_empty_web_query_returns_without_key_or_network(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with patch(
        "tools.web_search.safe_download",
        side_effect=AssertionError("network should not run"),
    ) as network:
        assert web_search("   ") == []
    network.assert_not_called()


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


def test_single_page_masks_sensitive_success_url_and_body_metadata():
    downloaded = SimpleNamespace(
        final_url="https://alice:password@example.test/article?token=secret-value",
        headers={"Content-Type": "text/plain; charset=utf-8"},
        content=b"Contact alice@example.com and use api_key=top-secret",
        status_code=200,
    )
    with patch("tools.single_page.safe_download", return_value=downloaded):
        page = fetch_single_page("https://example.test/article")

    assert page.error is None
    assert "alice" not in page.url
    assert "password" not in page.url
    assert "secret-value" not in page.url
    assert "[REDACTED_CREDENTIALS]" in page.url
    assert "[REDACTED_SECRET]" in page.url
    assert "alice@example.com" not in page.text
    assert "top-secret" not in page.text


def test_single_page_returns_redacted_structured_error_and_url():
    with patch(
        "tools.single_page.safe_download",
        side_effect=ValueError("secret resolver path /private/state"),
    ):
        page = fetch_single_page(
            "https://alice:password@example.test/article?api_key=top-secret"
        )
    assert page.error == "Page fetch failed (ValueError)."
    assert "secret" not in page.error
    assert "/private" not in page.error
    assert "alice" not in page.url
    assert "password" not in page.url
    assert "top-secret" not in page.url
    assert page.text == ""
