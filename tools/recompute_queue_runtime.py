"""Environment-governed durable transport construction for research recomputation.

Distributed recomputation is opt-in. Local recomputation remains available without a
transport, while publish/worker modes fail closed unless a durable backend is explicitly
configured. SQLite is a same-host cross-process provider, not a multi-host queue.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tools.durable_queue import DurableQueue, SQLiteDurableQueue


@dataclass(frozen=True)
class RecomputeTransportConfig:
    backend: str
    sqlite_path: Path | None
    namespace: str
    queue_max_attempts: int
    queue_max_payload_bytes: int
    ledger_max_attempts: int
    claim_timeout_seconds: float
    visibility_timeout_seconds: float
    busy_retry_delay_seconds: float


def _text(source: Mapping[str, str], key: str, default: str, *, maximum: int = 256) -> str:
    raw = source.get(key, default)
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a string")
    value = raw.strip()
    if (
        not value
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{key} is invalid")
    return value


def _integer(
    source: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(key, "")
    if raw is None or not str(raw).strip():
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _seconds(
    source: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = source.get(key, "")
    if raw is None or not str(raw).strip():
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def load_recompute_transport_config(
    environ: Mapping[str, str] | None = None,
) -> RecomputeTransportConfig:
    source = os.environ if environ is None else environ
    backend = str(source.get("RECOMPUTE_QUEUE_BACKEND", "disabled")).strip().lower()
    if backend not in {"disabled", "sqlite"}:
        raise ValueError("RECOMPUTE_QUEUE_BACKEND must be disabled or sqlite")

    namespace = _text(
        source,
        "RECOMPUTE_QUEUE_NAMESPACE",
        "research-recompute-v1",
        maximum=128,
    )
    queue_max_attempts = _integer(
        source,
        "RECOMPUTE_QUEUE_MAX_ATTEMPTS",
        5,
        minimum=1,
        maximum=100,
    )
    queue_max_payload_bytes = _integer(
        source,
        "RECOMPUTE_QUEUE_MAX_PAYLOAD_BYTES",
        4096,
        minimum=256,
        maximum=1_048_576,
    )
    ledger_max_attempts = _integer(
        source,
        "RECOMPUTE_LEDGER_MAX_ATTEMPTS",
        5,
        minimum=1,
        maximum=100,
    )
    claim_timeout_seconds = _seconds(
        source,
        "RECOMPUTE_CLAIM_TIMEOUT_SECONDS",
        900.0,
        minimum=1.0,
        maximum=86_400.0,
    )
    visibility_timeout_seconds = _seconds(
        source,
        "RECOMPUTE_VISIBILITY_TIMEOUT_SECONDS",
        1800.0,
        minimum=1.0,
        maximum=86_400.0,
    )
    busy_retry_delay_seconds = _seconds(
        source,
        "RECOMPUTE_BUSY_RETRY_SECONDS",
        30.0,
        minimum=0.0,
        maximum=86_400.0,
    )
    if visibility_timeout_seconds < claim_timeout_seconds:
        raise ValueError(
            "RECOMPUTE_VISIBILITY_TIMEOUT_SECONDS must be at least "
            "RECOMPUTE_CLAIM_TIMEOUT_SECONDS"
        )

    sqlite_path: Path | None = None
    if backend == "sqlite":
        configured = str(source.get("RECOMPUTE_QUEUE_DB_PATH", "")).strip()
        if configured:
            sqlite_path = Path(configured)
        else:
            storage = Path(str(source.get("CLASSIC_STORAGE_DIR", "data")).strip() or "data")
            sqlite_path = storage / "governance" / "research_recompute_queue.sqlite3"

    return RecomputeTransportConfig(
        backend=backend,
        sqlite_path=sqlite_path,
        namespace=namespace,
        queue_max_attempts=queue_max_attempts,
        queue_max_payload_bytes=queue_max_payload_bytes,
        ledger_max_attempts=ledger_max_attempts,
        claim_timeout_seconds=claim_timeout_seconds,
        visibility_timeout_seconds=visibility_timeout_seconds,
        busy_retry_delay_seconds=busy_retry_delay_seconds,
    )


def build_recompute_queue(config: RecomputeTransportConfig) -> DurableQueue:
    if not isinstance(config, RecomputeTransportConfig):
        raise TypeError("config must be RecomputeTransportConfig")
    if config.backend == "disabled":
        raise RuntimeError(
            "distributed recompute transport is disabled; set "
            "RECOMPUTE_QUEUE_BACKEND=sqlite for same-host durable workers"
        )
    if config.backend == "sqlite":
        if config.sqlite_path is None:
            raise RuntimeError("sqlite recompute queue path is not configured")
        return SQLiteDurableQueue(
            config.sqlite_path,
            namespace=config.namespace,
            max_attempts=config.queue_max_attempts,
            max_payload_bytes=config.queue_max_payload_bytes,
        )
    raise RuntimeError(f"unsupported recompute queue backend: {config.backend}")


__all__ = [
    "RecomputeTransportConfig",
    "build_recompute_queue",
    "load_recompute_transport_config",
]
