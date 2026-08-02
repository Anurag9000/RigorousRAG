from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools import evidence_graph_set_signed_publication_runtime as runtime


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
