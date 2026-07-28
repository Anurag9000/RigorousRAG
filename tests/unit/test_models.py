import json
import math

import pytest
from pydantic import ValidationError

from tools.models import AgentAnswer, Citation


def test_non_finite_answer_metadata_is_normalized_to_null():
    answer = AgentAnswer(
        answer="grounded response",
        metadata={
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "finite": 1.25,
            "nested": {"score": float("nan")},
            "values": [1, float("inf"), True, "evidence"],
        },
    )

    payload = answer.model_dump()
    assert payload["metadata"]["nan"] is None
    assert payload["metadata"]["positive_infinity"] is None
    assert payload["metadata"]["negative_infinity"] is None
    assert payload["metadata"]["finite"] == 1.25
    assert payload["metadata"]["nested"]["score"] is None
    assert payload["metadata"]["values"] == [1, None, True, "evidence"]
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_non_finite_citation_metadata_is_json_safe():
    citation = Citation(
        label="[1]",
        title="Evidence",
        url="https://example.test/evidence",
        metadata={"distance": math.inf, "rank": 1},
    )

    payload = citation.model_dump()
    assert payload["metadata"] == {"distance": None, "rank": 1}
    json.dumps(payload, allow_nan=False)


def test_whitespace_only_public_fields_are_rejected():
    with pytest.raises(ValidationError, match="answers may not be empty"):
        AgentAnswer(answer="   ")
    with pytest.raises(ValidationError, match="titles may not be empty"):
        Citation(label="[1]", title="   ", url="https://example.test")
    with pytest.raises(ValidationError, match="URLs may not be empty"):
        Citation(label="[1]", title="Evidence", url="   ")
    with pytest.raises(ValidationError, match="non-empty bracket notation"):
        Citation(label="[ ]", title="Evidence", url="https://example.test")


def test_public_citation_fields_redact_credentials_paths_and_pii():
    citation = Citation(
        label="[1]",
        title="Contact analyst@example.com at /private/report.txt",
        url="https://alice:password@example.test/paper?api_key=secret",
        source_id="file:///var/lib/rigorousrag/source.pdf",
        snippet="Evidence from 10 Main Street and 192.168.1.20",
    )

    payload = citation.model_dump()
    serialized = json.dumps(payload)
    assert "analyst@example.com" not in serialized
    assert "/private" not in serialized
    assert "alice" not in payload["url"]
    assert "password" not in payload["url"]
    assert "api_key=secret" not in payload["url"]
    assert "/var/lib" not in serialized
    assert "10 Main Street" not in serialized
    assert "192.168.1.20" not in serialized


def test_metadata_and_warning_truncation_is_explicit():
    answer = AgentAnswer(
        answer="grounded",
        warnings=["warning"] * 100,
        metadata={f"key-{index}": index for index in range(150)},
    )

    payload = answer.model_dump()
    assert len(payload["warnings"]) == 100
    assert len(payload["metadata"]) == 101
    assert payload["metadata"]["__truncated_items__"] is True


def test_hostile_metadata_object_is_serialized_safely():
    class BrokenString:
        def __str__(self):
            raise RuntimeError("private /secret/path")

    answer = AgentAnswer(
        answer="grounded",
        metadata={"object": BrokenString()},
    )

    assert answer.metadata["object"] == "[UNPRINTABLE_BrokenString]"
