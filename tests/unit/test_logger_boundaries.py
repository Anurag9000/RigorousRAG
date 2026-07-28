import json
import os

import pytest

import tools.logger as logger
from tools.logger import log_activity, log_agent_run, log_tool_call


def test_telemetry_masks_diagnostics_and_normalizes_non_finite_numbers(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr("tools.logger.LOG_FILE", str(path))

    log_activity(
        "failure at /private/service.log",
        {
            "error": "file:///var/lib/rigorousrag/jobs.sqlite3",
            "provider": "https://alice:password@example.test?api_key=secret",
            "score": float("nan"),
            "values": [float("inf"), float("-inf"), 1.5],
        },
    )
    log_tool_call("tool", float("nan"), False, tokens=float("inf"))
    log_agent_run("private query", float("inf"), float("nan"), owner_id="alice")

    raw = path.read_text(encoding="utf-8")
    entries = [json.loads(line) for line in raw.splitlines()]
    assert len(entries) == 3
    assert "/private" not in raw
    assert "/var/lib" not in raw
    assert "password" not in raw
    assert "api_key=secret" not in raw
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert entries[0]["details"]["score"] is None
    assert entries[0]["details"]["values"] == [None, None, 1.5]
    assert entries[1]["details"]["duration_sec"] == 0.0
    assert entries[1]["details"]["estimated_tokens"] == 0
    assert entries[2]["details"]["duration_sec"] == 0.0
    assert entries[2]["details"]["citations"] == 0


def test_symlinked_telemetry_path_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "target.txt"
    target.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "metrics.jsonl"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    monkeypatch.setattr("tools.logger.LOG_FILE", str(link))

    log_activity("event", {"value": "should not be written"})

    assert target.read_text(encoding="utf-8") == "unchanged"
    assert link.is_symlink()


def test_symlinked_telemetry_parent_is_refused(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "logs"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable on this platform.")
    monkeypatch.setattr(logger, "LOG_FILE", str(linked_parent / "metrics.jsonl"))

    log_activity("event", {"value": "not written outside"})

    assert list(outside.iterdir()) == []


def test_nonregular_telemetry_destination_is_refused_without_blocking(
    tmp_path,
    monkeypatch,
):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform.")
    fifo = tmp_path / "metrics.jsonl"
    os.mkfifo(fifo)
    monkeypatch.setattr(logger, "LOG_FILE", str(fifo))

    log_activity("event", {"value": "must not block"})

    assert fifo.exists()
    assert not fifo.is_file()


def test_malformed_telemetry_integer_environment_uses_default(monkeypatch):
    monkeypatch.setenv("BAD_TELEMETRY_INTEGER", "not-an-integer")

    value = logger._bounded_int_env(
        "BAD_TELEMETRY_INTEGER",
        17,
        minimum=1,
        maximum=100,
    )

    assert value == 17
