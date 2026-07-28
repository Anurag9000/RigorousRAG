import json
import os

import pytest

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
