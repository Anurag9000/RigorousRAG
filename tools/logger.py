"""Failure-isolated, privacy-conscious JSONL telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOG_FILE = os.getenv("USAGE_LOG_FILE", "usage_metrics.jsonl")
_LOG_LOCK = threading.Lock()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def log_activity(activity_type: str, details: Dict[str, Any]) -> None:
    """Append one event. Telemetry failure never fails the user request."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": str(activity_type),
        "details": _json_safe(details),
    }
    try:
        path = Path(LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception:
        return


def log_tool_call(
    tool_name: str,
    duration: float,
    success: bool,
    tokens: int = 0,
    error_type: str | None = None,
) -> None:
    log_activity(
        "tool_call",
        {
            "tool": tool_name,
            "duration_sec": round(max(duration, 0.0), 3),
            "success": bool(success),
            "estimated_tokens": max(int(tokens), 0),
            "error_type": error_type,
        },
    )


def log_agent_run(
    query: str,
    total_time: float,
    citation_count: int,
    *,
    success: bool = True,
    owner_id: str | None = None,
) -> None:
    query_bytes = (query or "").encode("utf-8")
    log_activity(
        "agent_run",
        {
            "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "query_length": len(query or ""),
            "duration_sec": round(max(total_time, 0.0), 3),
            "citations": max(int(citation_count), 0),
            "success": bool(success),
            "owner_id": owner_id,
        },
    )
