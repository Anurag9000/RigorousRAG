"""Fail-closed exact-expiry boundary for signed review actor assertions."""

from __future__ import annotations

import time
from typing import Any

from tools import evidence_graph_relation_actor_assertion as _assertions

_MARKER = "_exact_review_actor_assertion_expiry_installed"


def install_exact_actor_assertion_expiry() -> None:
    if getattr(_assertions, _MARKER, False):
        return
    original = _assertions.verify_review_actor_assertion

    def verify_with_exact_expiry(*args: Any, **kwargs: Any):
        supplied_now = kwargs.get("now")
        current = time.time() if supplied_now is None else supplied_now
        kwargs["now"] = current
        result = original(*args, **kwargs)
        if result.expires_at < float(current):
            raise PermissionError("actor assertion has expired.")
        return result

    _assertions._verify_review_actor_assertion_with_skew = original
    _assertions.verify_review_actor_assertion = verify_with_exact_expiry
    setattr(_assertions, _MARKER, True)


install_exact_actor_assertion_expiry()

__all__ = ["install_exact_actor_assertion_expiry"]
