import itertools
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

import pytest
from pydantic import ValidationError

import tools.ingestion_models as ingestion_models
from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


def _document(**overrides):
    values = {
        "id": "doc-1",
        "filename": "paper.txt",
        "file_path": "/private/uploads/paper.txt",
        "mime_type": "text/plain",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "title": "Paper",
        "text": "evidence",
        "sections": [DocumentSection(title="Full Text", content="evidence")],
        "metadata": {},
    }
    values.update(overrides)
    return IngestedDocument(**values)


def test_success_and_failure_states_are_consistent_and_boolean_strict():
    document = _document()

    assert IngestionResult(success=True, document=document).error is None
    failed = IngestionResult(success=False)
    assert failed.document is None
    assert failed.error == "Document ingestion failed."

    with pytest.raises(ValidationError, match="requires a document"):
        IngestionResult(success=True)
    with pytest.raises(ValidationError, match="may not include a document"):
        IngestionResult(success=False, document=document, error="failed")
    with pytest.raises(ValidationError, match="may not include an error"):
        IngestionResult(success=True, document=document, error="unexpected")
    for value in (1, 0, "true", object()):
        with pytest.raises(ValidationError, match="success"):
            IngestionResult(success=value)


def test_internal_path_is_bounded_excluded_and_assignment_validated():
    document = _document()
    document.file_path = "/private/uploads/updated.txt"

    payload = document.model_dump(mode="json")

    assert "file_path" not in payload
    assert document.file_path.endswith("updated.txt")
    with pytest.raises(ValidationError):
        _document(file_path="x" * 4097)
    for value in ("bad\x00path", "bad\npath", "bad\rpath", "bad\x7fpath"):
        with pytest.raises(ValidationError):
            document.file_path = value


def test_required_identifiers_mime_and_text_are_strict():
    for field, value in (
        ("id", object()),
        ("id", ""),
        ("id", "bad\x00id"),
        ("id", "bad\nid"),
        ("id", "bad\rid"),
        ("id", "bad\x7fid"),
        ("file_path", object()),
        ("file_path", "bad\x00path"),
        ("file_path", "bad\npath"),
        ("file_path", "bad\rpath"),
        ("file_path", "bad\x7fpath"),
        ("mime_type", object()),
        ("mime_type", "text/plain; charset=utf-8"),
        ("mime_type", "not-a-mime"),
        ("text", object()),
        ("text", "bad\x00text"),
        ("text", "bad\x01text"),
        ("text", "bad\x7ftext"),
    ):
        with pytest.raises(ValidationError):
            _document(**{field: value})
    with pytest.raises(ValidationError):
        _document(id="d" * 201)
    with pytest.raises(ValidationError):
        _document(mime_type="m" * 201)


def test_document_text_preserves_normal_layout_whitespace():
    value = "line one\nline two\tcell\r\nline three"

    document = _document(
        text=value,
        sections=[DocumentSection(title="Layout", content=value)],
    )

    assert document.text == value
    assert document.sections[0].content == value


def test_created_at_requires_timezone_and_is_normalized_to_utc():
    with pytest.raises(ValidationError, match="timezone"):
        _document(created_at=datetime(2026, 1, 1))

    offset = timezone(timedelta(hours=5, minutes=30))
    document = _document(
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=offset)
    )

    assert document.created_at.tzinfo == timezone.utc
    assert document.created_at.hour == 6
    assert document.created_at.minute == 30


def test_assignment_validation_masks_display_metadata():
    document = _document()

    document.filename = "analyst@example.com-paper.txt"
    document.title = "Study at /private/report.txt"
    document.sections[0].title = "Call +1 202-555-0114"

    payload = document.model_dump(mode="json")
    serialized = json.dumps(payload)
    assert "analyst@example.com" not in serialized
    assert "/private" not in serialized
    assert "202-555-0114" not in serialized


def test_metadata_cycles_nonfinite_values_and_hostile_objects_are_safe():
    cyclic = {}
    cyclic["self"] = cyclic

    class Hostile:
        def __str__(self):
            raise RuntimeError("private /secret/path")

    document = _document(
        metadata={
            "cycle": cyclic,
            "nan": float("nan"),
            "private": "file:///var/lib/secret.txt",
            "hostile": Hostile(),
        }
    )

    payload = document.model_dump(mode="json")
    assert payload["metadata"]["cycle"]["self"] == "[CIRCULAR_REFERENCE]"
    assert payload["metadata"]["nan"] is None
    assert payload["metadata"]["private"] == "[REDACTED_PATH]"
    assert payload["metadata"]["hostile"] == "[UNPRINTABLE_Hostile]"
    json.dumps(payload, allow_nan=False)


def test_metadata_item_count_is_bounded_with_marker_on_assignment():
    document = _document()
    document.metadata = {
        "email": "alice@example.com",
        **{f"key-{index}": index for index in range(1100)},
    }

    assert document.metadata["email"] == "[REDACTED_EMAIL]"
    assert len(document.metadata) == 1001
    assert document.metadata["__truncated_items__"] is True


def test_section_fields_and_extra_inputs_are_strict():
    for page_number in (True, 2.0, Decimal("2"), Fraction(4, 2), 0, 1_000_001):
        with pytest.raises(ValidationError, match="page_number"):
            DocumentSection(
                title="Section",
                content="evidence",
                page_number=page_number,
            )
    with pytest.raises(ValidationError, match="content"):
        DocumentSection(title="Section", content=object())
    for content in ("bad\x00content", "bad\x01content", "bad\x7fcontent"):
        with pytest.raises(ValidationError, match="content"):
            DocumentSection(title="Section", content=content)
    with pytest.raises(ValidationError):
        DocumentSection(title="Empty", content="")
    with pytest.raises(ValidationError, match="Extra inputs"):
        DocumentSection(
            title="Section",
            content="evidence",
            unexpected=True,
        )


def test_section_page_number_accepts_exact_index_protocol():
    class ExactInteger:
        def __index__(self):
            return 3

    section = DocumentSection(
        title="Section",
        content="evidence",
        page_number=ExactInteger(),
    )
    section.page_number = ExactInteger()

    assert section.page_number == 3


def test_section_count_and_infinite_iterables_are_bounded_before_materialization():
    section = DocumentSection(title="Section", content="evidence")
    with pytest.raises(ValidationError, match="at most 10000"):
        _document(sections=itertools.repeat(section))
    with pytest.raises(ValidationError, match="list of document sections"):
        _document(sections="not-a-list")
    with pytest.raises(ValidationError):
        _document(
            sections=[
                DocumentSection(title=f"Section {index}", content="evidence")
                for index in range(10_001)
            ]
        )


def test_aggregate_section_text_limit_applies_on_creation_and_assignment(monkeypatch):
    monkeypatch.setattr(ingestion_models, "_MAX_DOCUMENT_TEXT_CHARS", 10)
    sections = [
        DocumentSection(title="One", content="123456"),
        DocumentSection(title="Two", content="abcdef"),
    ]

    with pytest.raises(ValidationError, match="Aggregate semantic section text"):
        _document(sections=sections, text="short")

    document = _document(
        sections=[DocumentSection(title="One", content="123")],
        text="short",
    )
    with pytest.raises(ValidationError, match="Aggregate semantic section text"):
        document.sections = sections


def test_extra_document_and_result_fields_are_forbidden():
    with pytest.raises(ValidationError, match="Extra inputs"):
        _document(unexpected=True)
    with pytest.raises(ValidationError, match="Extra inputs"):
        IngestionResult(success=False, unexpected=True)
