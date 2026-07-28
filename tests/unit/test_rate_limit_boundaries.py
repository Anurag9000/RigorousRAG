import math

import pytest

from tools.rate_limit import SlidingWindowRateLimiter


def test_rate_limiter_rejects_malformed_configuration():
    with pytest.raises(ValueError, match="must be integers"):
        SlidingWindowRateLimiter("bad")
    with pytest.raises(ValueError, match="must be integers"):
        SlidingWindowRateLimiter(10, max_keys="bad")


def test_rate_limiter_rejects_invalid_keys_and_clocks():
    limiter = SlidingWindowRateLimiter()

    with pytest.raises(ValueError, match="keys"):
        limiter.retry_after("")
    with pytest.raises(ValueError, match="keys"):
        limiter.retry_after("k" * 201)
    for value in (float("nan"), float("inf"), -1):
        with pytest.raises(ValueError, match="finite and non-negative"):
            limiter.retry_after("alice", now=value)


def test_rate_limiter_returns_finite_retry_delay():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)

    assert limiter.retry_after("alice", now=0) == 0.0
    assert limiter.retry_after("alice", now=1) == 0.0
    retry = limiter.retry_after("alice", now=2)

    assert 0 < retry <= 60
    assert math.isfinite(retry)


def test_stale_key_is_reclaimed_when_capacity_is_needed():
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, max_keys=1)
    assert limiter.retry_after("alice", now=0) == 0.0

    assert limiter.retry_after("bob", now=61) == 0.0

    assert "alice" not in limiter._events
    assert "bob" in limiter._events


def test_live_key_capacity_exhaustion_fails_closed():
    limiter = SlidingWindowRateLimiter(requests_per_minute=1, max_keys=1)
    assert limiter.retry_after("alice", now=0) == 0.0

    with pytest.raises(RuntimeError, match="capacity"):
        limiter.retry_after("bob", now=1)
