from decimal import Decimal
from fractions import Fraction

import pytest

from tools.due_scheduler import DueScheduler, _MAX_CLOCK_RECHECK_SECONDS


def test_scheduler_rejects_fractional_capacity_objects():
    for value in (Decimal("1.5"), Fraction(3, 2)):
        with pytest.raises(ValueError, match="integer"):
            DueScheduler(max_pending_keys=value)


def test_scheduler_accepts_exact_index_protocol_capacity():
    class ExactInteger:
        def __index__(self):
            return 2

    scheduler = DueScheduler(max_pending_keys=ExactInteger())
    try:
        assert scheduler.max_pending_keys == 2
    finally:
        scheduler.shutdown()


def test_scheduler_rejects_boolean_deadlines_without_starting_thread():
    scheduler = DueScheduler(name="boolean-deadline")
    try:
        for deadline in (True, False):
            with pytest.raises(ValueError, match="not boolean"):
                scheduler.schedule("job", deadline, lambda: None)
        assert scheduler.thread_started() is False
        assert scheduler.pending_count() == 0
    finally:
        scheduler.shutdown()


def test_scheduler_rechecks_long_wall_clock_waits_periodically():
    assert 0 < _MAX_CLOCK_RECHECK_SECONDS <= 60
