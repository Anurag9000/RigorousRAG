import argparse
import json
from types import SimpleNamespace
from unittest.mock import patch

from ingest_docs import _collect_files, main


def test_collect_files_filters_supported_types_and_output(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "c.exe").write_text("x")
    output = tmp_path / "ingestion_manifest.json"
    output.write_text("old")
    files = _collect_files([str(tmp_path)], recursive=False, output_path=output)
    assert [path.name for path in files] == ["a.txt", "b.pdf"]


def test_main_writes_privacy_safe_manifest(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence")
    output = tmp_path / "manifest.json"
    document = SimpleNamespace(
        model_dump=lambda **_kwargs: {"id": "doc-1", "filename": "paper.txt"}
    )
    indexed = SimpleNamespace(document=document, chunk_count=3)
    args = argparse.Namespace(
        paths=[str(path)], recursive=False, output=str(output), owner_id="alice",
        include_redacted_text=False, fail_fast=False,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", return_value=indexed):
        code = main()
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload == [{"id": "doc-1", "filename": "paper.txt", "chunk_count": 3}]


def test_main_returns_failure_for_ingestion_errors(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence")
    args = argparse.Namespace(
        paths=[str(path)], recursive=False, output=None, owner_id="alice",
        include_redacted_text=False, fail_fast=True,
    )
    with patch("ingest_docs.parse_args", return_value=args), \
         patch("ingest_docs.get_rag_layer", return_value=object()), \
         patch("ingest_docs._llm_client", return_value=None), \
         patch("ingest_docs.ingest_and_index", side_effect=ValueError("failed")):
        assert main() == 1
