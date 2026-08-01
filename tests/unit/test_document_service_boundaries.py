import hashlib
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tools.document_service import (
    _bounded_source_sha256,
    _summary_sample,
    index_document,
    summarize_document,
)
from tools.ingestion_models import DocumentSection, IngestedDocument


def _document(tmp_path, *, metadata=None, text="evidence"):
    source = tmp_path / "paper.txt"
    source.write_text(text, encoding="utf-8")
    return IngestedDocument(
        id="doc-1",
        filename="paper.txt",
        file_path=str(source),
        mime_type="text/plain",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Study",
        text=text,
        sections=[DocumentSection(title="Full Text", content=text)],
        metadata=metadata or {},
    )


def test_bounded_source_hash_matches_regular_file(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")
    assert _bounded_source_sha256(source, max_bytes=100) == hashlib.sha256(
        b"evidence"
    ).hexdigest()


def test_bounded_source_hash_rejects_oversized_replacement(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"x" * 101)
    with pytest.raises(ValueError, match="byte limit"):
        _bounded_source_sha256(source, max_bytes=100)


def test_bounded_source_hash_refuses_symlink_swap(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"private evidence")
    link = tmp_path / "paper.txt"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    with pytest.raises(ValueError, match="unavailable"):
        _bounded_source_sha256(link, max_bytes=100)


def test_bounded_source_hash_rejects_invalid_limit(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")
    for invalid in (0, True, 1.5, "bad"):
        with pytest.raises(ValueError, match="integer|between"):
            _bounded_source_sha256(source, max_bytes=invalid)


def test_summary_sample_validates_limits_and_samples_document_regions(tmp_path):
    document = _document(
        tmp_path,
        text="a" * 3000 + "middle" + "z" * 3000,
    )
    sample = _summary_sample(document, max_chars=900)
    assert "[BEGINNING]" in sample
    assert "[MIDDLE]" in sample
    assert "[END]" in sample
    assert "middle" in sample
    with pytest.raises(ValueError, match="max_chars"):
        _summary_sample(document, max_chars=0)


def test_document_metadata_cannot_override_protected_index_fields(
    tmp_path,
    monkeypatch,
):
    document = _document(
        tmp_path,
        metadata={
            "owner_id": "bob",
            "filename": "secret.pdf",
            "mime_type": "application/secret",
            "created_at": "attacker",
            "llm_summary": "attacker summary",
            "job_id": "attacker-job",
            "custom": "kept",
        },
    )
    captured = {}

    def commit(_document_value, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(vector_rows=1)

    monkeypatch.setattr(
        "tools.document_service.commit_finalized_document",
        commit,
    )
    indexed = index_document(
        document,
        owner_id="alice",
        rag=object(),
        job_id="job-1",
    )

    metadata = captured["metadata"]
    assert indexed.chunk_count == 1
    assert metadata["owner_id"] == "alice"
    assert metadata["filename"] == "paper.txt"
    assert metadata["mime_type"] == "text/plain"
    assert metadata["created_at"].startswith("2026-01-01")
    assert metadata["job_id"] == "job-1"
    assert metadata["llm_summary"] != "attacker summary"
    assert metadata["custom"] == "kept"
    assert captured["audit_metadata"] == {
        "job_id": "job-1",
        "operation": "ingestion",
    }


def test_summary_provider_output_is_bounded_masked_and_failure_safe(tmp_path):
    document = _document(tmp_path)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "https://alice:password@example.test?api_key=secret "
                        + "x" * 10_000
                    )
                )
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    summary = summarize_document(document, client=client, model="model")
    assert len(summary) <= 2000
    assert "password" not in summary
    assert "api_key=secret" not in summary

    malformed_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(choices=[])
            )
        )
    )
    assert summarize_document(document, client=malformed_client) == "evidence"
    with pytest.raises(ValueError, match="summary model"):
        summarize_document(document, client=client, model="m" * 201)


def test_invalid_job_id_or_chunk_count_is_rejected(tmp_path, monkeypatch):
    document = _document(tmp_path)
    with pytest.raises(ValueError, match="job_id"):
        index_document(
            document,
            owner_id="alice",
            rag=object(),
            job_id="j" * 201,
        )

    monkeypatch.setattr(
        "tools.document_service.commit_finalized_document",
        lambda *_args, **_kwargs: SimpleNamespace(vector_rows=-1),
    )
    with pytest.raises(ValueError, match="chunk_count"):
        index_document(document, owner_id="alice", rag=object())
