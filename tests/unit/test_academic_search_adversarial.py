import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.academic_search import AcademicSearchError, academic_search


def _payload(data):
    return SimpleNamespace(content=json.dumps({"data": data}).encode("utf-8"))


def test_scholarly_results_use_academic_index_provenance():
    with patch(
        "tools.academic_search.safe_download",
        return_value=_payload(
            [
                {
                    "title": "Paper",
                    "paperId": "paper-1",
                    "url": "https://example.test/paper",
                    "year": 2026,
                }
            ]
        ),
    ):
        results = academic_search("query")

    assert results[0].source_type == "academic_index"


def test_zero_year_range_is_encoded_without_truthiness_loss():
    with patch(
        "tools.academic_search.safe_download",
        return_value=_payload([]),
    ) as download:
        assert academic_search("query", year_from=0, year_to=1) == []

    assert "year=0-1" in download.call_args.args[0]


def test_provider_key_is_trimmed_before_header_use(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "  provider-key  ")
    with patch(
        "tools.academic_search.safe_download",
        return_value=_payload([]),
    ) as download:
        academic_search("query")

    assert download.call_args.kwargs["headers"]["x-api-key"] == "provider-key"


def test_hostile_paper_mapping_becomes_generic_provider_structure_failure():
    class BrokenPaper(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("private provider detail")

    with patch(
        "tools.academic_search.safe_download",
        return_value=_payload([BrokenPaper()]),
    ):
        assert academic_search("query") == []


def test_hostile_author_and_external_id_collections_do_not_leak():
    class BrokenAuthors(list):
        def __iter__(self):
            raise RuntimeError("private author detail")

    class BrokenExternalIds(dict):
        def items(self):
            raise RuntimeError("private metadata detail")

    with patch(
        "tools.academic_search.safe_download",
        return_value=_payload(
            [
                {
                    "title": "Paper",
                    "paperId": "paper-1",
                    "url": "https://example.test/paper",
                    "authors": BrokenAuthors(),
                    "externalIds": BrokenExternalIds(),
                }
            ]
        ),
    ):
        results = academic_search("query")

    assert results[0].metadata["authors"] == ""
    assert results[0].metadata["external_ids"] == {}


def test_nonbyte_provider_content_is_invalid_json():
    with patch(
        "tools.academic_search.safe_download",
        return_value=SimpleNamespace(content=object()),
    ):
        with pytest.raises(AcademicSearchError, match="invalid JSON"):
            academic_search("query")
