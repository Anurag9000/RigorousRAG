"""Verify the active v3 loader without downloading datasets or starting training."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

from tools import dataset_cohort_runtime_entry as canonical
from tools import dataset_cohort_runtime_entry_v3 as v3


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\\0".encode("ascii") + payload).hexdigest()


def test_v3_pins_match_corrected_sources():
    assert _sha(ROOT / v3.BASE_PATH) == v3.BASE_BLOB
    assert _sha(ROOT / v3.RUNTIME_PATH) == v3.RUNTIME_BLOB
    assert v3.BASE_COMMIT == "955a092e4a3e2cc04adfd8007206acd6d1341dce"
    assert v3.RUNTIME_COMMIT == "0160deb303f6a4d2a48b8453244dc01ceaae1295"


def test_active_canonical_loader_delegates_to_v3():
    sentinel = object()
    with mock.patch.object(v3, "load_runtime", return_value=sentinel) as load:
        assert canonical.load_runtime(ROOT) is sentinel
    load.assert_called_once_with(ROOT)


def test_historical_v1_loader_pins_remain_unchanged():
    assert canonical.RUNTIME_COMMIT == "0c50adb23e5ba58b7c49b18401950f0bf7e5b736"
    assert canonical.RUNTIME_BLOB == "1ffb1af0f70812d3418a6f0959ae88c7a145939e"
