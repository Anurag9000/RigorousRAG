import itertools
import json
import math
from decimal import Decimal
from fractions import Fraction

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


def test_nested_metadata_lists_preserve_bounded_structures():
    answer = AgentAnswer(
        answer="grounded",
        metadata={
            "values": [
                {"score": 1.0, "path": "/private/state.json"},
                ["analyst@example.com", {"finite": 2}],
            ]
        },
    )

    values = answer.metadata["values"]
    assert values[0]["score"] == 1.0
    assert values[0]["path"] == "[REDACTED_PATH]"
    assert values[1][0] == "[REDACTED_EMAIL]"
    assert values[1][1] == {"finite": 2}


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


def test_citation_url_schemes_hostnames_and_page_numbers_are_strict():
    for url in (
        "file:///private/evidence.pdf",
        "javascript:alert(1)",
        "doi:10.1000/test",
        "https:///missing-host",
        "https://not a hostname.test/evidence",
        "https://example..test/evidence",
        "https://-bad.example.test/evidence",
        "https://singlelabel/evidence",
        "http://127.0.0.1/evidence",
        "http://[::1]/evidence",
        "https://alice:password@example.test/evidence",
        "https://example.test\\@evil.test/evidence",
        "local://",
    ):
        with pytest.raises(ValidationError):
            Citation(label="[1]", title="Evidence", url=url)

    for page_number in (True, 2.0, Decimal("2"), Fraction(4, 2), 0, 1_000_001):
        with pytest.raises(ValidationError, match="page_number"):
            Citation(
                label="[1]",
                title="Evidence",
                url="local://doc-1",
                page_number=page_number,
            )

    citation = Citation(
        label="[1]",
        title="Evidence",
        url="local://doc-1",
        page_number=2,
    )
    assert citation.url == "local://doc-1"
    assert citation.page_number == 2


def test_page_number_accepts_exact_index_protocol_value():
    class ExactInteger:
        def __index__(self):
            return 3

    citation = Citation(
        label="[1]",
        title="Evidence",
        url="local://doc-1",
        page_number=ExactInteger(),
    )

    assert citation.page_number == 3


def test_public_citation_fields_redact_paths_pii_and_query_secrets():
    citation = Citation(
        label="[1]",
        title="Contact analyst@example.com at /private/report.txt",
        url="https://example.test/paper?api_key=secret",
        source_id="file:///var/lib/rigorousrag/source.pdf",
        snippet="Evidence from 10 Main Street and 192.168.1.20",
    )

    payload = citation.model_dump()
    serialized = json.dumps(payload)
    assert "analyst@example.com" not in serialized
    assert "/private" not in serialized
    assert "api_key=secret" not in payload["url"]
    assert "/var/lib" not in serialized
    assert "10 Main Street" not in serialized
    assert "192.168.1.20" not in serialized


def test_public_ipv6_and_default_url_components_are_normalized():
    citation = Citation(
        label="[1]",
        title="Evidence",
        url="https://[2606:4700:4700::1111]:443/paper",
    )

    assert citation.url == "https://[2606:4700:4700::1111]:443/paper"


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


def test_citation_and_warning_iterables_are_bounded_before_materialization():
    citation = Citation(
        label="[1]",
        title="Evidence",
        url="local://doc-1",
    )

    with pytest.raises(ValidationError, match="at most 100"):
        AgentAnswer(
            answer="grounded",
            citations=itertools.repeat(citation),
        )
    with pytest.raises(ValidationError, match="at most 100"):
        AgentAnswer(
            answer="grounded",
            warnings=itertools.repeat("warning"),
        )
    with pytest.raises(ValidationError, match="must be a list"):
        AgentAnswer(answer="grounded", citations="not-a-list")
    with pytest.raises(ValidationError, match="must be a list"):
        AgentAnswer(answer="grounded", warnings="not-a-list")


def test_duplicate_labels_and_duplicate_evidence_are_removed():
    first = Citation(
        label="[1]",
        title="First",
        url="local://doc-1",
        source_id="chunk-1",
        quote="evidence",
    )
    duplicate_label = Citation(
        label="[1]",
        title="Second",
        url="local://doc-2",
        source_id="chunk-2",
        quote="other",
    )
    duplicate_evidence = Citation(
        label="[2]",
        title="First copy",
        url="local://doc-1",
        source_id="chunk-1",
        quote="evidence",
    )

    answer = AgentAnswer(
        answer="grounded",
        citations=[first, duplicate_label, duplicate_evidence],
    )

    assert answer.citations == [first]


def test_assignment_validation_prevents_post_construction_bypass():
    citation = Citation(
        label="[1]",
        title="Evidence",
        url="local://doc-1",
    )
    answer = AgentAnswer(answer="grounded")

    with pytest.raises(ValidationError):
        citation.url = "file:///private/evidence.pdf"
    with pytest.raises(ValidationError):
        citation.url = "https://alice:password@example.test/evidence"
    with pytest.raises(ValidationError):
        citation.page_number = True
    with pytest.raises(ValidationError):
        citation.page_number = 2.0
    with pytest.raises(ValidationError):
        answer.answer = "   "
    with pytest.raises(ValidationError):
        answer.warnings = ["warning"] * 101


def test_hostile_metadata_object_is_serialized_safely():
    class BrokenString:
        def __str__(self):
            raise RuntimeError("private /secret/path")

    answer = AgentAnswer(
        answer="grounded",
        metadata={"object": BrokenString()},
    )

    assert answer.metadata["object"] == "[UNPRINTABLE_BrokenString]"
