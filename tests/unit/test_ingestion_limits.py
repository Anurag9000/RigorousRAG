import os
import zipfile

import fitz
import pytest

import tools.ingestion as ingestion
from tools.ingestion import ingest_file


def test_documents_differing_only_in_redacted_pii_keep_distinct_ids(tmp_path):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("Contact alice@example.com for the study.", encoding="utf-8")
    second_path.write_text("Contact bob@example.net for the study.", encoding="utf-8")

    first = ingest_file(str(first_path), owner_id="alice")
    second = ingest_file(str(second_path), owner_id="alice")

    assert first.success and first.document is not None
    assert second.success and second.document is not None
    assert first.document.text == second.document.text
    assert first.document.metadata["content_sha256"] == second.document.metadata["content_sha256"]
    assert first.document.id != second.document.id
    assert first.document.metadata["document_identity"] == "owner_and_source_sha256"


def test_docx_uncompressed_expansion_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "_MAX_DOCX_UNCOMPRESSED_BYTES", 200)
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>" + "A" * 5000)

    result = ingest_file(str(path), owner_id="alice")

    assert not result.success
    assert "uncompressed content" in (result.error or "").lower()


def test_pdf_page_count_limit_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "_MAX_PDF_PAGES", 1)
    path = tmp_path / "many-pages.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Page one")
    pdf.new_page().insert_text((72, 72), "Page two")
    pdf.save(path)
    pdf.close()

    result = ingest_file(str(path), owner_id="alice")

    assert not result.success
    assert "1-page limit" in (result.error or "")


def test_text_character_limit_is_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "_MAX_EXTRACTED_CHARS", 20)
    path = tmp_path / "large.txt"
    path.write_text("A meaningful sentence that exceeds the limit.", encoding="utf-8")

    result = ingest_file(str(path), owner_id="alice")

    assert not result.success
    assert "character limit" in (result.error or "").lower()


def test_symlinked_input_is_rejected(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("evidence", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")

    result = ingest_file(str(link), owner_id="alice")

    assert not result.success
    assert "symbolic-link" in (result.error or "").lower()


def test_invalid_owner_is_rejected_before_identity_generation(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("evidence", encoding="utf-8")

    result = ingest_file(str(path), owner_id="../other")

    assert not result.success
    assert "owner identifiers" in (result.error or "").lower()
