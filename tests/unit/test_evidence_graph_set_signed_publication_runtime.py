from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import evidence_graph_set_signed_publication_runtime as runtime
from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
)
from tools.evidence_graph_set_publish_runtime import (
    clear_evidence_graph_set_publication_journal_cache,
    get_evidence_graph_set_publication_journal,
)


def install_factory(monkeypatch):
    observed = []
    marker = object()

    def factory(path):
        observed.append(Path(path))
        return marker

    monkeypatch.setattr(runtime, "get_evidence_graph_set_publication_journal", factory)
    return observed, marker


def clear_environment(monkeypatch):
    monkeypatch.delenv("EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH", raising=False)
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH", raising=False
    )


def test_signed_runtime_uses_distinct_default_path(tmp_path, monkeypatch):
    clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    observed, marker = install_factory(monkeypatch)

    result = runtime.get_evidence_graph_set_signed_publication_journal()

    assert result is marker
    assert observed == [
        tmp_path / "data" / "evidence_graph_set_signed_publications.sqlite3"
    ]
    assert observed[0] != (
        tmp_path / "data" / "evidence_graph_set_publications.sqlite3"
    )


def test_signed_runtime_honors_explicit_override(tmp_path, monkeypatch):
    clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
        "private/signed.sqlite3",
    )
    observed, marker = install_factory(monkeypatch)

    assert runtime.get_evidence_graph_set_signed_publication_journal() is marker
    assert observed == [tmp_path / "private" / "signed.sqlite3"]


def test_signed_runtime_rejects_same_canonical_path(tmp_path, monkeypatch):
    clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "shared/publications.sqlite3",
    )
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
        str(tmp_path / "shared" / "." / "publications.sqlite3"),
    )
    observed, _marker = install_factory(monkeypatch)

    with pytest.raises(RuntimeError, match="distinct paths"):
        runtime.get_evidence_graph_set_signed_publication_journal()
    assert observed == []


def test_signed_runtime_rejects_existing_hard_link_alias(tmp_path, monkeypatch):
    if not hasattr(os, "link"):
        pytest.skip("hard links are unavailable")
    clear_environment(monkeypatch)
    common = tmp_path / "common.sqlite3"
    signed = tmp_path / "signed.sqlite3"
    common.write_bytes(b"journal")
    try:
        os.link(common, signed)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    monkeypatch.setenv("EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH", str(common))
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH", str(signed)
    )
    observed, _marker = install_factory(monkeypatch)

    with pytest.raises(RuntimeError, match="alias one file"):
        runtime.get_evidence_graph_set_signed_publication_journal()
    assert observed == []


def test_same_logical_operation_has_independent_durable_state(
    tmp_path, monkeypatch
):
    common_path = tmp_path / "authorization-only.sqlite3"
    signed_path = tmp_path / "signed.sqlite3"
    monkeypatch.setenv("EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH", str(common_path))
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH", str(signed_path)
    )
    clear_evidence_graph_set_publication_journal_cache()
    try:
        common = get_evidence_graph_set_publication_journal()
        signed = runtime.get_evidence_graph_set_signed_publication_journal()
        attempt = EvidenceGraphSetPublicationAttempt.create(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=("1" * 64,),
            expected_current_set_id=None,
            now=1.0,
        )

        common.seed(attempt)
        with pytest.raises(KeyError):
            signed.get(attempt.operation_id)

        signed.seed(attempt)
        common.cancel(
            attempt.operation_id,
            owner_id="alice",
            confirm_operation_id=attempt.operation_id,
            now=2.0,
        )

        assert common.get(attempt.operation_id).state == "cancelled"
        assert signed.get(attempt.operation_id).state == "planned"
        assert common.path != signed.path
    finally:
        clear_evidence_graph_set_publication_journal_cache()
