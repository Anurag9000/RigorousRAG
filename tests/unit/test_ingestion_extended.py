import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from tools.ingestion import redact_text, extract_academic_metadata, _chunk_text_semantically, ingest_file

class TestIngestionExtended:
    def test_redact_text(self):
        text = "Contact me at test@example.com or call +1-234-567-8901."
        redacted = redact_text(text)
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "test@example.com" not in redacted
        assert "234-567-8901" not in redacted

    def test_extract_academic_metadata(self):
        text = "Deep Learning in RAG\nAuthors: Jane Doe\nDOI: 10.1234/nature123\nPublished in 2023."
        meta = extract_academic_metadata(text)
        assert meta["doi"] == "10.1234/nature123"
        assert meta["year"] == "2023"
        assert meta["extracted_title"] == "Deep Learning in RAG"

    def test_chunk_text_semantically(self):
        text = "Para 1\n\nPara 2 is long and should be kept together if possible but we will test the limit."
        chunks = _chunk_text_semantically(text, max_chars=20)
        # "Para 1" is 6 chars. "Para 2..." is 80+ chars.
        # Should split "Para 2..." into smaller pieces.
        assert len(chunks) > 1
        assert chunks[0] == "Para 1"

    @patch('tools.ingestion.fitz.open')
    def test_ingest_pdf(self, mock_fitz_open, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("fake")
        
        mock_doc = mock_fitz_open.return_value
        mock_doc.metadata = {"title": "PDF Title"}
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page Content"
        mock_doc.__iter__.return_value = [mock_page]
        
        result = ingest_file(str(f))
        assert result.success
        assert result.document.title == "PDF Title"
        assert "Page Content" in result.document.text

    @patch('tools.ingestion.docx.Document')
    def test_ingest_docx(self, mock_docx_class, tmp_path):
        f = tmp_path / "test.docx"
        f.write_text("fake")
        
        mock_doc = mock_docx_class.return_value
        mock_doc.core_properties.title = "Docx Title"
        
        # Mocking paragraphs
        p1 = MagicMock()
        p1.text = "Heading 1"
        p1.style.name = "Heading 1"
        p2 = MagicMock()
        p2.text = "Para Content"
        p2.style.name = "Normal"
        mock_doc.paragraphs = [p1, p2]
        
        result = ingest_file(str(f))
        assert result.success
        assert result.document.title == "Docx Title"
        assert "Heading 1" in result.document.text

    def test_ingest_text_unicode_fallback(self, tmp_path):
        f = tmp_path / "test.txt"
        # Write some non-utf8 content (latin-1)
        content = "Héllo".encode('latin-1')
        f.write_bytes(content)
        
        result = ingest_file(str(f))
        assert result.success
        assert "Héllo" in result.document.text

    def test_ingest_unsupported(self, tmp_path):
        f = tmp_path / "test.exe"
        f.write_text("fake")
        result = ingest_file(str(f))
        assert not result.success
        assert "Unsupported file type" in result.error
