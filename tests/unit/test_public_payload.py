import math

import pytest

from tools.public_payload import public_model_payload, sanitize_public_payload


def test_nested_private_fields_are_removed_without_removing_public_siblings():
    value = {
        "doc_id": "d1",
        "metadata": {
            "source_path": "/private/file",
            "authorization": "Bearer secret",
            "public": "retained",
        },
    }
    assert sanitize_public_payload(value) == {
        "doc_id": "d1",
        "metadata": {"public": "retained"},
    }


def test_nonfinite_controls_depth_and_unsupported_values_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        sanitize_public_payload({"value": math.nan})
    with pytest.raises(ValueError, match="invalid"):
        sanitize_public_payload({"value": "bad\x7f"})
    value = {}
    cursor = value
    for _ in range(7):
        cursor["nested"] = {}
        cursor = cursor["nested"]
    with pytest.raises(ValueError, match="nesting"):
        sanitize_public_payload(value)
    with pytest.raises(ValueError, match="unsupported"):
        sanitize_public_payload({"value": object()})


def test_model_payload_does_not_stringify_or_iterate_arbitrary_objects():
    class Hostile:
        called = False
        def __str__(self):
            self.called = True
            raise AssertionError("must not stringify")
    hostile = Hostile()
    assert public_model_payload({"value": hostile}) is None
    assert hostile.called is False


def test_model_dump_is_contained_and_sanitized():
    class Model:
        def model_dump(self, **kwargs):
            return {
                "doc_id": "d1",
                "metadata": {"file_path": "/private", "public": 1},
            }
    assert public_model_payload(Model()) == {
        "doc_id": "d1",
        "metadata": {"public": 1},
    }
