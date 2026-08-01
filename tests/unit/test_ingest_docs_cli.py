import argparse
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ingest_docs
from ingest_docs import _atomic_manifest, _collect_files, main
from tools.document_service import IndexedDocument
from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


def _document(path: Path, *, doc_id: str = "doc-1") -> IngestedDocument:
    return IngestedDocument(
        id=doc_id,
        filename=path.name,
        file_path=str(path),
        mime_type="text/plain",
        text="evidence",
        sections=[DocumentSection(title="Full Text", content="evidence")],
    )


def _args(path: Path, output: Path | None, **overrides):
    values = {
        "paths": [str(path)],
        "recursive": False,
        "output": str(output) if output is not None else None,
        "owner_id": "alice",
        "retain_sources": False,
        "include_redacted_text": False,
        "fail_fast": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _success(path: Path, doc_id: str = "doc-1"):
    document = _document(path, doc_id=doc_id)
    return document, IngestionResult(success=True, document=document)


@contextmanager
def _patched_main(path, output, *, store, rag, result, indexed, args=None):
    snapshot = object()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "ingest_docs.parse_args",
                return_value=args or _args(path, output),
            )
        )
        stack.enter_context(patch("ingest_docs.get_rag_layer", return_value=rag))
        stack.enter_context(
            patch("ingest_docs.get_document_store", return_value=store)
        )
        stack.enter_context(patch("ingest_docs._llm_client", return_value=None))
        stack.enter_context(patch("ingest_docs.ingest_file", return_value=result))
        stack.enter_context(
            patch("ingest_docs.index_document", return_value=indexed)
        )
        capture = stack.enter_context(
            patch(
                "ingest_docs.capture_authoritative_document",
                return_value=snapshot,
            )
        )
        restore = stack.enter_context(
            patch("ingest_docs.restore_authoritative_document")
        )
        yield snapshot, capture, restore


def test_collect_files_filters_supported_types_and_output(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.exe").write_text("x", encoding="utf-8")
    output = tmp_path / "ingestion_manifest.json"
    output.write_text("old", encoding="utf-8")
    files = _collect_files([str(tmp_path)], recursive=False, output_path=output)
    assert [path.name for path in files] == ["a.txt", "b.pdf"]


def test_collect_files_refuses_symlinked_files_and_directories(tmp_path):
    real_file = tmp_path / "real.txt"
    real_file.write_text("evidence", encoding="utf-8")
    real_directory = tmp_path / "real-dir"
    real_directory.mkdir()
    (real_directory / "nested.txt").write_text("evidence", encoding="utf-8")
    file_link = tmp_path / "file-link.txt"
    directory_link = tmp_path / "directory-link"
    try:
        os.symlink(real_file, file_link)
        os.symlink(real_directory, directory_link, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are unavailable in this environment.")
    assert _collect_files(
        [str(file_link), str(directory_link)],
        recursive=True,
        output_path=None,
    ) == []


def test_collect_files_bounds_total_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_docs, "_MAX_INPUT_FILES", 2)
    for index in range(3):
        (tmp_path / f"{index}.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="At most 2"):
        _collect_files([str(tmp_path)], recursive=False, output_path=None)


def test_main_writes_privacy_safe_manifest(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    output = tmp_path / "manifest.json"
    document, result = _success(path)
    indexed = IndexedDocument(document=document, chunk_count=3)
    rag = MagicMock()
    store = MagicMock()
    store.register.return_value = None
    with _patched_main(
        path,
        output,
        store=store,
        rag=rag,
        result=result,
        indexed=indexed,
    ) as (_snapshot, capture, restore):
        assert main() == 0
        capture.assert_called_once_with(
            owner_id="alice",
            doc_id="doc-1",
            rag=rag,
        )
        restore.assert_not_called()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["id"] == "doc-1"
    assert payload[0]["chunk_count"] == 3
    assert payload[0]["source_retained"] is False
    assert "file_path" not in payload[0]
    assert "text" not in payload[0]
    assert "sections" not in payload[0]
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
    document, result = _success(path)
    indexed = IndexedDocument(document=document, chunk_count=3)
    rag = MagicMock()
    store = MagicMock()
    store.copy_source.return_value = retained
    store.register.return_value = str(old)
    store.remove_source.return_value = True
    with _patched_main(
        path,
        output,
        store=store,
        rag=rag,
        result=result,
        indexed=indexed,
        args=_args(path, output, retain_sources=True),
    ):
        assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["source_retained"] is True
    assert str(retained) not in json.dumps(payload)
    store.copy_source.assert_called_once_with(owner_id="alice", source_path=path)
    store.remove_source.assert_called_once_with(str(old))


def test_previous_source_cleanup_failure_is_nonterminal_and_visible(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    output = tmp_path / "manifest.json"
    document, result = _success(path)
    indexed = IndexedDocument(document=document, chunk_count=3)
    rag = MagicMock()
    store = MagicMock()
    store.register.return_value = "/private/old-source.txt"
    store.remove_source.side_effect = RuntimeError("cleanup failed")
    with _patched_main(
        path,
        output,
        store=store,
        rag=rag,
        result=result,
        indexed=indexed,
    ):
        assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["replacement_cleanup_pending"] is True
    assert "/private" not in json.dumps(payload)


def test_registry_failure_restores_authoritative_generation(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    retained = tmp_path / "uploads" / "alice" / "random.txt"
    document, result = _success(path)
    indexed = IndexedDocument(document=document, chunk_count=3)
    rag = MagicMock()
    store = MagicMock()
    store.copy_source.return_value = retained
    store.register.side_effect = RuntimeError("registry down")
    with _patched_main(
        path,
        None,
        store=store,
        rag=rag,
        result=result,
        indexed=indexed,
        args=_args(path, None, retain_sources=True, fail_fast=True),
    ) as (snapshot, _capture, restore):
        assert main() == 1
        restore.assert_called_once_with(snapshot, rag=rag)
    store.remove_source.assert_called_once_with(retained)
    rag.delete_document.assert_not_called()
    rag.collection.upsert.assert_not_called()


def test_index_failure_relies_on_authoritative_internal_rollback(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    _document_value, result = _success(path)
    rag = MagicMock()
    store = MagicMock()
    snapshot = object()
    with patch(
        "ingest_docs.parse_args",
        return_value=_args(path, None, fail_fast=True),
    ), patch("ingest_docs.get_rag_layer", return_value=rag), patch(
        "ingest_docs.get_document_store", return_value=store
    ), patch("ingest_docs._llm_client", return_value=None), patch(
        "ingest_docs.ingest_file", return_value=result
    ), patch(
        "ingest_docs.capture_authoritative_document",
        return_value=snapshot,
    ), patch(
        "ingest_docs.index_document",
        side_effect=RuntimeError("failed"),
    ), patch("ingest_docs.restore_authoritative_document") as restore:
        assert main() == 1
    restore.assert_not_called()
    store.register.assert_not_called()


def test_dependency_initialization_failure_is_generic(tmp_path, capsys):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")
    with patch("ingest_docs.parse_args", return_value=_args(path, None)), patch(
        "ingest_docs.get_rag_layer",
        side_effect=RuntimeError("failed at /private/vector"),
    ):
        assert main() == 1
    error = capsys.readouterr().err
    assert "dependencies are unavailable" in error
    assert "/private" not in error


def test_atomic_manifest_replaces_existing_file_and_leaves_no_temporary(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_text("old", encoding="utf-8")
    _atomic_manifest(output, [{"id": "doc-1"}])
    assert json.loads(output.read_text(encoding="utf-8")) == [{"id": "doc-1"}]
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
