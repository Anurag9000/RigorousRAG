import os
from unittest.mock import MagicMock

import pytest

from tools.document_service import index_document
from tools.ingestion import ingest_file


def test_indexing_rejects_same_size_same_timestamp_source_replacement(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("alpha evidence", encoding="utf-8")
    result = ingest_file(str(source), owner_id="alice")
    assert result.success and result.document is not None
    original_stat = source.stat()

    source.write_text("bravo evidence", encoding="utf-8")
    assert source.stat().st_size == original_stat.st_size
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    rag = MagicMock()
    with pytest.raises(ValueError, match="changed after parsing"):
        index_document(result.document, owner_id="alice", rag=rag)

    rag.add_document.assert_not_called()


def test_indexing_accepts_unchanged_ingested_source(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("stable evidence", encoding="utf-8")
    result = ingest_file(str(source), owner_id="alice")
    assert result.success and result.document is not None
    rag = MagicMock()
    rag.add_document.return_value = 2

    indexed = index_document(result.document, owner_id="alice", rag=rag)

    assert indexed.chunk_count == 2
    rag.add_document.assert_called_once()
