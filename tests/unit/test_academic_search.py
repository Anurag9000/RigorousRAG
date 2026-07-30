import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.academic_search import AcademicSearchError, academic_search


def provider_payload(data):
    return SimpleNamespace(content=json.dumps({"data": data}).encode("utf-8"))


def test_academic_search_rejects_invalid_year_range_before_network():
    with patch("tools.academic_search.safe_download") as network:
        with pytest.raises(ValueError, match="year_from"):
            academic_search("query", year_from=2025, year_to=2020)
    network.assert_not_called()


def test_direct_query_year_and_limit_inputs_are_strict_before_network():
    network = MagicMock(side_effect=AssertionError("network should not run"))
    with patch("tools.academic_search.safe_download", network):
        cases = [
            {"query": object()},
            {"query": "q" * 2001},
            {"query": "query", "year_from": True},
            {"query": "query", "year_from": 2020.5},
            {"query": "query", "year_from": "bad"},
            {"query": "query", "year_to": 10_000},
            {"query": "query", "limit": True},
            {"query": "query", "limit": 1.5},
            {"query": "query", "limit": "bad"},
            {"query": "query", "limit": 0},
            {"query": "query", "limit": 11},
        ]
        for arguments in cases:
            with pytest.raises(ValueError):
                academic_search(**arguments)
    network.assert_not_called()


def test_empty_query_returns_without_network():
    with patch(
        "tools.academic_search.safe_download",
        side_effect=AssertionError("network should not run"),
    ) as network:
        assert academic_search("   ") == []
    network.assert_not_called()


def test_academic_search_filters_years_and_builds_provenance():
    papers = [
        {
            "paperId": "paper-1",
            "title": "Older paper",
            "abstract": "older evidence",
            "authors": [{"name": "Author One"}],
            "year": 2019,
            "venue": "Venue",
            "url": "https://example.test/older",
            "externalIds": {"DOI": "10.1/older"},
        },
        {
            "paperId": "paper-2",
            "title": "Selected paper",
            "abstract": "selected evidence",
            "authors": [{"name": "Author Two"}],
            "year": 2022,
            "venue": "Journal",
            "url": "https://example.test/selected",
            "externalIds": {"DOI": "10.1/selected"},
        },
    ]
    with patch(
        "tools.academic_search.safe_download",
        return_value=provider_payload(papers),
    ) as safe:
        results = academic_search(
            "evidence",
            year_from=2020,
            year_to=2023,
            limit=5,
        )

    assert [item.title for item in results] == ["Selected paper"]
    assert results[0].source_id == "paper-2"
    assert results[0].metadata["year"] == 2022
    assert results[0].metadata["authors"] == "Author Two"
    assert safe.call_args.kwargs["allowed_content_types"] == {"application/json"}
    assert "year=2020-2023" in safe.call_args.args[0]


def test_missing_public_url_uses_quoted_semantic_scholar_identifier():
    paper = {
        "paperId": "paper/id with spaces",
        "title": "Paper",
        "year": 2026,
        "authors": [],
        "externalIds": {},
    }
    with patch(
        "tools.academic_search.safe_download",
        return_value=provider_payload([paper]),
    ):
        results = academic_search("query")

    assert len(results) == 1
    assert results[0].url.endswith("paper%2Fid%20with%20spaces")


def test_invalid_provider_json_and_structures_are_distinguished():
    for payload in (b"not-json", b'{"score":NaN}', b"[]", b"\xff"):
        with patch(
            "tools.academic_search.safe_download",
            return_value=SimpleNamespace(content=payload),
        ):
            with pytest.raises(
                AcademicSearchError,
                match="invalid JSON|invalid result structure",
            ):
                academic_search("query")

    with patch(
        "tools.academic_search.safe_download",
        return_value=SimpleNamespace(content=b'{"data":{}}'),
    ):
        with pytest.raises(AcademicSearchError, match="invalid result structure"):
            academic_search("query")


def test_provider_request_failure_is_generic():
    with patch(
        "tools.academic_search.safe_download",
        side_effect=RuntimeError("private provider path /secret/state"),
    ):
        with pytest.raises(AcademicSearchError) as captured:
            academic_search("query")

    assert "private provider" not in str(captured.value)
    assert "/secret" not in str(captured.value)


def test_provider_key_is_validated_before_network(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "x" * 4097)
    with patch("tools.academic_search.safe_download") as network:
        with pytest.raises(AcademicSearchError, match="key is invalid"):
            academic_search("query")
    network.assert_not_called()

    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "key\r\nInjected: yes")
    with patch("tools.academic_search.safe_download") as network:
        with pytest.raises(AcademicSearchError, match="key is invalid"):
            academic_search("query")
    network.assert_not_called()


def test_malformed_or_unsafe_provider_items_are_skipped():
    # Provider JSON cannot contain arbitrary Python objects. Use malformed values that
    # are representable on the actual wire so the adapter, rather than json.dumps,
    # receives and rejects them.
    papers = [
        None,
        7,
        "not-a-paper",
        {"title": "", "paperId": "empty"},
        {"title": "Boolean year", "paperId": "bool", "year": True},
        {"title": "Unsafe URL", "url": "file:///private/paper.pdf", "paperId": "unsafe"},
        {
            "title": "Valid",
            "paperId": "valid",
            "year": 2025,
            "url": "https://example.test/valid",
            "authors": [
                {"name": "Alice"},
                {"name": ["not", "text"]},
                9,
            ],
            "externalIds": {
                "DOI": "10.1/valid",
                "nested": {"ignored": True},
                "bool": True,
            },
        },
    ]
    with patch(
        "tools.academic_search.safe_download",
        return_value=provider_payload(papers),
    ):
        results = academic_search("query", year_from=2020)

    assert [result.title for result in results] == ["Valid"]
    assert results[0].metadata["authors"] == "Alice"
    assert "nested" not in results[0].metadata["external_ids"]
    assert results[0].metadata["external_ids"]["bool"] is True


def test_provider_candidates_are_bounded_before_citation_validation(monkeypatch):
    papers = [
        {
            "title": f"Unsafe {index}",
            "url": "file:///private/paper.pdf",
            "paperId": f"paper-{index}",
        }
        for index in range(100)
    ]
    monkeypatch.setattr(
        "tools.academic_search._MAX_PROVIDER_CANDIDATES",
        7,
    )
    with patch(
        "tools.academic_search.safe_download",
        return_value=provider_payload(papers),
    ), patch("tools.academic_search.Citation", side_effect=ValueError("unsafe")) as citation:
        assert academic_search("query", limit=10) == []

    assert citation.call_count == 7
