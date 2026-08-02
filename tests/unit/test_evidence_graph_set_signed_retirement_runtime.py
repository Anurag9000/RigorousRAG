from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import evidence_graph_set_signed_retirement_runtime as runtime


def clear(monkeypatch):
    for name in (
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime.clear_signed_publication_retirement_journal_cache()


def test_retirement_runtime_uses_third_distinct_default_path(tmp_path, monkeypatch):
    clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    journal = runtime.get_signed_publication_retirement_journal()
    assert journal.path == (
        tmp_path / "data" / "evidence_graph_set_signed_retirements.sqlite3"
    )
    assert journal.path != tmp_path / "data" / "evidence_graph_set_publications.sqlite3"
    assert journal.path != (
        tmp_path / "data" / "evidence_graph_set_signed_publications.sqlite3"
    )


def test_retirement_runtime_rejects_canonical_aliases(tmp_path, monkeypatch):
    clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH", "shared/common.sqlite3"
    )
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        str(tmp_path / "shared" / "." / "common.sqlite3"),
    )
    with pytest.raises(RuntimeError, match="distinct files"):
        runtime.get_signed_publication_retirement_journal()

    clear(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH", "shared/signed.sqlite3"
    )
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        str(tmp_path / "shared" / "signed.sqlite3"),
    )
    with pytest.raises(RuntimeError, match="distinct files"):
        runtime.get_signed_publication_retirement_journal()


def test_retirement_runtime_rejects_existing_hard_link_alias(tmp_path, monkeypatch):
    if not hasattr(os, "link"):
        pytest.skip("hard links unavailable")
    clear(monkeypatch)
    publication = tmp_path / "publication.sqlite3"
    retirement = tmp_path / "retirement.sqlite3"
    publication.write_bytes(b"journal")
    try:
        os.link(publication, retirement)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    monkeypatch.setenv("EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH", str(publication))
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH", str(retirement)
    )
    with pytest.raises(RuntimeError, match="distinct files"):
        runtime.get_signed_publication_retirement_journal()


def test_runtime_cache_is_path_scoped(tmp_path, monkeypatch):
    clear(monkeypatch)
    first = runtime.get_signed_publication_retirement_journal(tmp_path / "one.sqlite3")
    second = runtime.get_signed_publication_retirement_journal(tmp_path / "one.sqlite3")
    third = runtime.get_signed_publication_retirement_journal(tmp_path / "two.sqlite3")
    assert first is second
    assert first is not third
