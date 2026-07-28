import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ingest_docs import _collect_files, main


def test_collect_files_filters_supported_types_and_output(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.exe").write_text("x", encoding="utf-8")
    output = tmp_path / "ingestion_manifest.json"
    output.write_text("old", encoding="utf-8")
    files = _collect_files([str(tmp_path)], recursive=False, output_path=output)
    assert [path.name for path in files] == ["a.txt", "b.pdf"]


def _indexed_document():
    document = SimpleNamespace(
        id="doc-1",
        filename="paper.txt",
        mime_type="text/plain",
        model_dump=lambda **_kwargs: {"id": "doc-1", "filename": "paper.txt"},
    )
    return SimpleNamespace(document=document, chunk_count=3)


def test_main_writes_privacy_safe_text_only_manifest(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    output = tmp_path / "manifest.json"
    store = MagicMock()
    store.register.return_value = None
    args = argparse.Namespace(
        paths=[str(path)],
        recursive=False,
        output=str(output),
        owner_id="alice",
        retain_sources=False,
        include_redacted_text=False,
        fail_fast=False,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs.get_document_store", return_value=store), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", return_value=_indexed_document()):
        code = main()
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == [{
        "id": "doc-1",
        "filename": "paper.txt",
        "chunk_count": 3,
        "source_retained": False,
    }]
    store.copy_source.assert_not_called()
    store.register.assert_called_once_with(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.txt",
        mime_type="text/plain",
        source_path=None,
    )


def test_main_retains_private_copy_without_manifest_path(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    output = tmp_path / "manifest.json"
    retained = tmp_path / "uploads" / "alice" / "random.txt"
    old = tmp_path / "uploads" / "alice" / "old.txt"
    store = MagicMock()
    store.copy_source.return_value = retained
    store.register.return_value = str(old)
    args = argparse.Namespace(
        paths=[str(path)],
        recursive=False,
        output=str(output),
        owner_id="alice",
        retain_sources=True,
        include_redacted_text=False,
        fail_fast=False,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs.get_document_store", return_value=store), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", return_value=_indexed_document()):
        assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["source_retained"] is True
    assert str(retained) not in json.dumps(payload)
    store.copy_source.assert_called_once_with(owner_id="alice", source_path=path)
    store.remove_source.assert_called_once_with(str(old))


def test_main_removes_new_copy_when_registration_fails(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    retained = tmp_path / "uploads" / "alice" / "random.txt"
    store = MagicMock()
    store.copy_source.return_value = retained
    store.register.side_effect = RuntimeError("registry down")
    args = argparse.Namespace(
        paths=[str(path)],
        recursive=False,
        output=None,
        owner_id="alice",
        retain_sources=True,
        include_redacted_text=False,
        fail_fast=True,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs.get_document_store", return_value=store), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", return_value=_indexed_document()):
        assert main() == 1
    store.remove_source.assert_called_once_with(retained)


def test_main_returns_failure_for_ingestion_errors(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    args = argparse.Namespace(
        paths=[str(path)],
        recursive=False,
        output=None,
        owner_id="alice",
        retain_sources=False,
        include_redacted_text=False,
        fail_fast=True,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs.get_document_store", return_value=MagicMock()), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", side_effect=ValueError("failed")):
        assert main() == 1
