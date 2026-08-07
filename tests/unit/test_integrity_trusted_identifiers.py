from __future__ import annotations

import json
from unittest.mock import patch

import fitz

from tools.document_store import get_document_store
from tools.integrity import check_visual_entailment


def _make_figure_pdf(path):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 600), "Figure 1. Accuracy")
    page.draw_rect(fitz.Rect(100, 250, 400, 520), color=(0, 0, 0))
    document.save(path)
    document.close()


def test_arbitrary_registry_identifier_does_not_bypass_privacy_mask(monkeypatch, tmp_path):
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "figure.pdf"
    source.parent.mkdir(parents=True)
    _make_figure_pdf(source)
    arbitrary_identifier = "555-123-4567"

    monkeypatch.setenv("UPLOAD_DIR", str(upload_root))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    get_document_store().register(
        owner_id="alice",
        doc_id=arbitrary_identifier,
        filename="figure.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    with patch(
        "tools.integrity._document_metadata",
        return_value={"filename": "figure.pdf"},
    ):
        result = json.loads(
            check_visual_entailment(
                "Accuracy increased.",
                "Figure 1",
                arbitrary_identifier,
                owner_id="alice",
                client=None,
            )
        )

    citation = result["citations"][0]
    assert citation["doc_id"] == "[REDACTED_PHONE]"
    assert arbitrary_identifier not in citation["url"]
