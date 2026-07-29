import json
from contextlib import contextmanager

import pytest

import tools.logger as logger


def test_log_activity_writes_bounded_jsonl_to_regular_private_file(monkeypatch, tmp_path):
    destination = tmp_path / "state" / "usage.jsonl"
    monkeypatch.setattr(logger, "LOG_FILE", str(destination))
    monkeypatch.setattr(logger, "LOG_MAX_BYTES", 1024 * 1024)
    monkeypatch.setattr(logger, "LOG_BACKUPS", 2)

    logger.log_activity(
        "agent@example.com",
        {
            "path": "/private/state.sqlite3",
            "score": float("nan"),
        },
    )

    assert destination.is_file()
    line = destination.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    serialized = json.dumps(payload, allow_nan=False)
    assert "agent@example.com" not in serialized
    assert "/private" not in serialized
    assert payload["details"]["score"] is None
    if hasattr(destination.stat(), "st_mode"):
        assert destination.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("control", ["\t", "\n", "\r", "\x7f"])
def test_control_bearing_telemetry_path_creates_nothing(monkeypatch, tmp_path, control):
    destination = tmp_path / f"unsafe{control}state" / "usage.jsonl"
    monkeypatch.setattr(logger, "LOG_FILE", str(destination))

    logger.log_activity("event", {"value": 1})

    assert not destination.exists()
    assert not destination.parent.exists()


def test_initial_symlinked_parent_is_refused(monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "state"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")
    monkeypatch.setattr(logger, "LOG_FILE", str(linked_parent / "usage.jsonl"))

    logger.log_activity("event", {"value": 1})

    assert list(outside.iterdir()) == []


def test_parent_swap_does_not_redirect_lock_or_append(monkeypatch, tmp_path):
    parent = tmp_path / "state"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "usage.jsonl"
    moved = tmp_path / "state-original"
    monkeypatch.setattr(logger, "LOG_FILE", str(destination))
    original_lock = logger._process_log_lock
    swapped = False

    @contextmanager
    def swap_after_lock(path, parent_fd=None):
        nonlocal swapped
        with original_lock(path, parent_fd):
            if not swapped:
                parent.rename(moved)
                try:
                    parent.symlink_to(outside, target_is_directory=True)
                except (OSError, NotImplementedError):
                    moved.rename(parent)
                    pytest.skip("Directory symlinks are unavailable in this environment.")
                swapped = True
            yield

    monkeypatch.setattr(logger, "_process_log_lock", swap_after_lock)

    logger.log_activity("event", {"value": 1})

    assert swapped is True
    assert list(outside.iterdir()) == []
    assert (moved / "usage.jsonl").is_file()
    assert json.loads((moved / "usage.jsonl").read_text(encoding="utf-8"))["type"] == "event"
