import math
from decimal import Decimal
from fractions import Fraction

import pytest

from tools.config import (
    bounded_float_env,
    bounded_int_env,
    bounded_optional_int_env,
)


def test_environment_names_are_validated_before_access():
    for name in (
        None,
        object(),
        "",
        "x" * 201,
        "BAD=NAME",
        "BAD\x00NAME",
        "BAD\nNAME",
        "BAD NAME",
        "9STARTS_WITH_DIGIT",
        "NON-PORTABLE",
    ):
        with pytest.raises(ValueError, match="Environment variable names"):
            bounded_int_env(name, 1, minimum=0, maximum=2)
        with pytest.raises(ValueError, match="Environment variable names"):
            bounded_float_env(name, 1.0, minimum=0.0, maximum=2.0)
        with pytest.raises(ValueError, match="Environment variable names"):
            bounded_optional_int_env(name, minimum=0, maximum=2)


def test_integer_helper_validates_parameters_and_preserves_clamping(monkeypatch):
    for arguments in (
        {"default": True, "minimum": 0, "maximum": 2},
        {"default": 1.5, "minimum": 0, "maximum": 2},
        {"default": Decimal("1.5"), "minimum": 0, "maximum": 2},
        {"default": Fraction(3, 2), "minimum": 0, "maximum": 2},
        {"default": 1, "minimum": True, "maximum": 2},
        {"default": 1, "minimum": Decimal("0.5"), "maximum": 2},
        {"default": 1, "minimum": 0, "maximum": 1.5},
    ):
        with pytest.raises(ValueError):
            bounded_int_env("VALUE", **arguments)
    with pytest.raises(ValueError, match="minimum may not exceed"):
        bounded_int_env("VALUE", 1, minimum=2, maximum=1)
    with pytest.raises(ValueError, match="write_back"):
        bounded_int_env("VALUE", 1, minimum=0, maximum=2, write_back=1)

    monkeypatch.setenv("VALUE", "999")
    assert bounded_int_env("VALUE", 1, minimum=0, maximum=10) == 10
    monkeypatch.setenv("VALUE", "bad")
    assert bounded_int_env("VALUE", 7, minimum=0, maximum=10) == 7


def test_integer_helper_accepts_exact_index_protocol_values(monkeypatch):
    class ExactInteger:
        def __index__(self):
            return 4

    monkeypatch.delenv("VALUE", raising=False)

    assert bounded_int_env(
        "VALUE",
        ExactInteger(),
        minimum=ExactInteger(),
        maximum=10,
    ) == 4


def test_float_helper_validates_parameters_and_normalizes_environment(monkeypatch):
    for arguments in (
        {"default": True, "minimum": 0.0, "maximum": 2.0},
        {"default": float("nan"), "minimum": 0.0, "maximum": 2.0},
        {"default": 1.0, "minimum": float("-inf"), "maximum": 2.0},
        {"default": 1.0, "minimum": 0.0, "maximum": float("inf")},
    ):
        with pytest.raises(ValueError):
            bounded_float_env("VALUE_FLOAT", **arguments)
    with pytest.raises(ValueError, match="minimum may not exceed"):
        bounded_float_env("VALUE_FLOAT", 1.0, minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError, match="write_back"):
        bounded_float_env(
            "VALUE_FLOAT",
            1.0,
            minimum=0.0,
            maximum=2.0,
            write_back="yes",
        )

    monkeypatch.setenv("VALUE_FLOAT", "nan")
    value = bounded_float_env("VALUE_FLOAT", 1.5, minimum=0.0, maximum=2.0)
    assert value == 1.5
    assert math.isfinite(value)
    monkeypatch.setenv("VALUE_FLOAT", "999")
    assert bounded_float_env("VALUE_FLOAT", 1.5, minimum=0.0, maximum=2.0) == 2.0


def test_optional_integer_helper_distinguishes_missing_malformed_and_bounded(monkeypatch):
    monkeypatch.delenv("OPTIONAL", raising=False)
    assert bounded_optional_int_env("OPTIONAL", minimum=1, maximum=10) is None
    monkeypatch.setenv("OPTIONAL", "")
    assert bounded_optional_int_env("OPTIONAL", minimum=1, maximum=10) is None
    monkeypatch.setenv("OPTIONAL", "bad")
    assert bounded_optional_int_env("OPTIONAL", minimum=1, maximum=10) is None
    monkeypatch.setenv("OPTIONAL", "999")
    assert bounded_optional_int_env("OPTIONAL", minimum=1, maximum=10) == 10

    with pytest.raises(ValueError, match="minimum may not exceed"):
        bounded_optional_int_env("OPTIONAL", minimum=10, maximum=1)
    with pytest.raises(ValueError):
        bounded_optional_int_env("OPTIONAL", minimum=True, maximum=10)
    with pytest.raises(ValueError):
        bounded_optional_int_env(
            "OPTIONAL",
            minimum=Fraction(1, 2),
            maximum=10,
        )
