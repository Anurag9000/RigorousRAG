from pathlib import Path

from tools.ingestion import _chunk_text_semantically, ingest_file, redact_text


def test_redaction_applies_to_full_text_and_sections(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "Contact alice@example.com or +1 202-555-0114.\n\nMethods were recorded.",
        encoding="utf-8",
    )
    result = ingest_file(str(path), owner_id="alice")
    assert result.success and result.document is not None
    document = result.document
    assert "alice@example.com" not in document.text
    assert "202-555-0114" not in document.text
    assert "[REDACTED_EMAIL]" in document.text
    assert all("alice@example.com" not in section.content for section in document.sections)
    assert all("202-555-0114" not in section.content for section in document.sections)
    assert "file_path" not in document.model_dump(mode="json")


def test_document_id_is_content_and_owner_stable(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("# Stable paper\n\nA repeatable body.", encoding="utf-8")
    first = ingest_file(str(path), owner_id="alice").document
    second = ingest_file(str(path), owner_id="alice").document
    other_owner = ingest_file(str(path), owner_id="bob").document
    assert first and second and other_owner
    assert first.id == second.id
    assert first.id != other_owner.id
    assert first.metadata["content_sha256"] == second.metadata["content_sha256"]


def test_semantic_chunks_enforce_maximum_even_for_long_sentence():
    text = "word " * 600
    chunks = _chunk_text_semantically(text, max_chars=120)
    assert chunks
    assert max(map(len, chunks)) <= 120


def test_binary_content_rejected_when_named_text(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_bytes(b"hello\x00binary")
    result = ingest_file(str(path))
    assert not result.success
    assert "binary" in (result.error or "").lower()


def test_redaction_does_not_mask_arbitrary_non_luhn_number():
    text = "Experiment identifier 1234 5678 9012 3456 is not a payment card."
    assert "[REDACTED_PAYMENT_CARD]" not in redact_text(text)
