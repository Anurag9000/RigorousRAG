import math

import pytest

from tools.rate_limit import SlidingWindowRateLimiter


def test_rate_limiter_rejects_malformed_or_out_of_range_configuration():
    for arguments in (
        {"requests_per_minute": "bad"},
        {"requests_per_minute": True},
        {"requests_per_minute": 1.5},
        {"requests_per_minute": 0},
        {"requests_per_minute": 1_000_001},
        {"requests_per_minute": 10, "max_keys": "bad"},
        {"requests_per_minute": 10, "max_keys": False},
        {"requests_per_minute": 10, "max_keys": 1.5},
        {"requests_per_minute": 10, "max_keys": 0},
    ):
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(**arguments)


def test_rate_limiter_rejects_invalid_keys_and_clocks():
    limiter = SlidingWindowRateLimiter()

    for key in (None, object(), "", "k" * 201, "bad\nkey", "bad\x00key"):
        with pytest.raises(ValueError, match="keys"):
            limiter.retry_after(key)
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="finite and non-negative"):
            limiter.retry_after("alice", now=value)


def test_rate_limiter_rejects_backwards_time_without_corrupting_queue():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)
    assert limiter.retry_after("alice", now=10) == 0.0

    with pytest.raises(ValueError, match="must not move backwards"):
        limiter.retry_after("alice", now=9)

    assert list(limiter._events["alice"]) == [10.0]
    assert limiter.retry_after("alice", now=11) == 0.0


def test_rate_limiter_returns_finite_retry_delay():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)

    assert limiter.retry_after("alice", now=0) == 0.0
    assert limiter.retry_after("alice", now=1) == 0.0
    retry = limiter.retry_after("alice", now=2)

    assert 0 < retry <= 60
    assert math.isfinite(retry)


def test_event_at_exact_window_boundary_is_expired():
    limiter = SlidingWindowRateLimiter(requests_per_minute=1)
    assert limiter.retry_after("alice", now=0) == 0.0
    assert limiter.retry_after("alice", now=60) == 0.0
    assert list(limiter._events["alice"]) == [60.0]


def test_stale_key_is_reclaimed_when_capacity_is_needed():
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, max_keys=1)
    assert limiter.retry_after("alice", now=0) == 0.0

    assert limiter.retry_after("bob", now=61) == 0.0

    assert "alice" not in limiter._events
    assert "bob" in limiter._events


def test_live_key_capacity_exhaustion_fails_closed_without_creating_key():
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, max_keys=1)
    assert limiter.retry_after("alice", now=0) == 0.0

    with pytest.raises(RuntimeError, match="capacity"):
        limiter.retry_after("bob", now=1)

    assert set(limiter._events) == {"alice"}
