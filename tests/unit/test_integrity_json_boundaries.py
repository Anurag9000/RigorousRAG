import pytest

from tools.integrity import _parse_json_object


def test_scientific_json_rejects_nonstandard_numeric_constants():
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="Non-standard JSON constant"):
            _parse_json_object(f'{{"confidence": {constant}}}')


def test_scientific_json_requires_an_object():
    with pytest.raises(ValueError, match="JSON object"):
        _parse_json_object('["not", "an", "object"]')


def test_scientific_json_accepts_bounded_fenced_object():
    parsed = _parse_json_object('```json\n{"verdict": "uncertain"}\n```')

    assert parsed == {"verdict": "uncertain"}


def test_scientific_json_rejects_oversized_output_before_parsing():
    with pytest.raises(ValueError, match="size limit"):
        _parse_json_object('{"value":"' + ('x' * 100_001) + '"}')
