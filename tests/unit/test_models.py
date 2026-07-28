import json
import math

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
