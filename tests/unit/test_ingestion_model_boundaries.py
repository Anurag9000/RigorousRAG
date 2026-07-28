import json

import pytest
from pydantic import ValidationError

from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


def _document(**overrides):
    values = {
        "id": "doc-1",
        "filename": "paper.txt",
        "file_path": "/private/uploads/paper.txt",
        "mime_type": "text/plain",
        "text": "evidence",
        "sections": [DocumentSection(title="Full Text", content="evidence")],
    }
    values.update(overrides)
    return IngestedDocument(**values)


def test_success_and_failure_states_are_consistent():
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


def test_internal_path_is_bounded_and_excluded_from_serialization():
    document = _document()

    payload = document.model_dump(mode="json")

    assert "file_path" not in payload
    with pytest.raises(ValidationError):
        _document(file_path="x" * 4097)


def test_required_identifiers_reject_whitespace_and_oversized_values():
    with pytest.raises(ValidationError, match="may not be empty"):
        _document(id="   ")
    with pytest.raises(ValidationError, match="may not be empty"):
        _document(mime_type="   ")
    with pytest.raises(ValidationError):
        _document(id="d" * 201)
    with pytest.raises(ValidationError):
        _document(mime_type="m" * 201)


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


def test_metadata_cycles_and_nonfinite_values_are_json_safe():
    cyclic = {}
    cyclic["self"] = cyclic
    document = _document(
        metadata={
            "cycle": cyclic,
            "nan": float("nan"),
            "private": "file:///var/lib/secret.txt",
        }
    )

    payload = document.model_dump(mode="json")
    assert payload["metadata"]["cycle"]["self"] == "[CIRCULAR_REFERENCE]"
    assert payload["metadata"]["nan"] is None
    assert payload["metadata"]["private"] == "[REDACTED_PATH]"
    json.dumps(payload, allow_nan=False)


def test_metadata_item_count_is_bounded_with_marker():
    document = _document(
        metadata={f"key-{index}": index for index in range(1500)}
    )

    assert len(document.metadata) == 1001
    assert document.metadata["__truncated_items__"] is True


def test_section_count_and_content_are_bounded():
    with pytest.raises(ValidationError):
        _document(
            sections=[
                DocumentSection(title=f"Section {index}", content="evidence")
                for index in range(10_001)
            ]
        )
    with pytest.raises(ValidationError):
        DocumentSection(title="Empty", content="")
