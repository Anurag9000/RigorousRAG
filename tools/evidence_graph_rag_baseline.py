"""Append-only governed historical baselines for evidence-graph benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.evidence_graph_rag_benchmark import GraphRAGBenchmarkReport
from tools.evidence_graph_rag_regression import (
    GraphRAGRegressionPolicy,
    GraphRAGRegressionReport,
    PairedMetricInterval,
    report_from_mapping,
)

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
_SCHEMA_VERSION = 1


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(
        character not in "0123456789abcdef" for character in cleaned
    ):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("baseline database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("baseline database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("baseline database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("baseline database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class GraphRAGBaselineRecord:
    benchmark_fingerprint: str
    benchmark_id: str
    policy_id: str
    policy_digest: str
    benchmark_report: GraphRAGBenchmarkReport
    previous_baseline_digest: str | None
    activation_regression_digest: str | None
    activated_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "benchmark_fingerprint",
            _digest(self.benchmark_fingerprint, "benchmark_fingerprint"),
        )
        object.__setattr__(
            self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id", 500)
        )
        object.__setattr__(
            self, "policy_id", _identifier(self.policy_id, "policy_id", 200)
        )
        object.__setattr__(
            self, "policy_digest", _digest(self.policy_digest, "policy_digest")
        )
        if not isinstance(self.benchmark_report, GraphRAGBenchmarkReport):
            raise ValueError("benchmark_report must be GraphRAGBenchmarkReport.")
        if (
            self.benchmark_report.benchmark_fingerprint
            != self.benchmark_fingerprint
            or self.benchmark_report.benchmark_id != self.benchmark_id
        ):
            raise ValueError("baseline report identity differs from baseline scope.")
        if self.previous_baseline_digest is not None:
            object.__setattr__(
                self,
                "previous_baseline_digest",
                _digest(
                    self.previous_baseline_digest,
                    "previous_baseline_digest",
                ),
            )
        if self.activation_regression_digest is not None:
            object.__setattr__(
                self,
                "activation_regression_digest",
                _digest(
                    self.activation_regression_digest,
                    "activation_regression_digest",
                ),
            )
        if (self.previous_baseline_digest is None) != (
            self.activation_regression_digest is None
        ):
            raise ValueError(
                "initial baselines require neither previous nor regression digest; "
                "replacement baselines require both."
            )
        object.__setattr__(
            self, "activated_at", _timestamp(self.activated_at, "activated_at")
        )
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("baseline schema is unsupported.")

    @property
    def baseline_digest(self) -> str:
        return _sha256(
            {
                "benchmark_fingerprint": self.benchmark_fingerprint,
                "benchmark_id": self.benchmark_id,
                "policy_id": self.policy_id,
                "policy_digest": self.policy_digest,
                "benchmark_report_digest": self.benchmark_report.report_digest,
                "previous_baseline_digest": self.previous_baseline_digest,
                "activation_regression_digest": self.activation_regression_digest,
                "schema_version": self.schema_version,
            }
        )


class GraphRAGBaselineStore:
    """Append-only baselines with one current pointer per contract and policy."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("baseline database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("baseline database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("baseline database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("baseline database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None
        ) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_rag_baselines (
                    benchmark_fingerprint TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    baseline_digest TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    benchmark_report_digest TEXT NOT NULL,
                    previous_baseline_digest TEXT,
                    activation_regression_digest TEXT,
                    payload_json TEXT NOT NULL,
                    activated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(benchmark_fingerprint, policy_id, baseline_digest)
                );
                CREATE TABLE IF NOT EXISTS graph_rag_baseline_current (
                    benchmark_fingerprint TEXT NOT NULL,
                    policy_id TEXT NOT NULL,
                    baseline_digest TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    benchmark_report_digest TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(benchmark_fingerprint, policy_id),
                    FOREIGN KEY(
                        benchmark_fingerprint, policy_id, baseline_digest
                    ) REFERENCES graph_rag_baselines(
                        benchmark_fingerprint, policy_id, baseline_digest
                    ) ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_rag_baseline_history
                    ON graph_rag_baselines(
                        benchmark_fingerprint, policy_id, activated_at DESC,
                        baseline_digest
                    );
                """
            )

    @staticmethod
    def _encode(value: GraphRAGBaselineRecord) -> str:
        report = asdict(value.benchmark_report)
        report["report_digest"] = value.benchmark_report.report_digest
        payload = {
            "benchmark_fingerprint": value.benchmark_fingerprint,
            "benchmark_id": value.benchmark_id,
            "policy_id": value.policy_id,
            "policy_digest": value.policy_digest,
            "benchmark_report": report,
            "previous_baseline_digest": value.previous_baseline_digest,
            "activation_regression_digest": value.activation_regression_digest,
            "activated_at": value.activated_at,
            "schema_version": value.schema_version,
        }
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if not rendered or len(rendered.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("baseline payload exceeds the byte limit.")
        return rendered

    @staticmethod
    def _decode(row: sqlite3.Row) -> GraphRAGBaselineRecord:
        if int(row["schema_version"]) != _SCHEMA_VERSION:
            raise RuntimeError("stored baseline schema is unsupported.")
        payload = row["payload_json"]
        if (
            not isinstance(payload, str)
            or not payload
            or len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES
        ):
            raise RuntimeError("stored baseline payload is corrupt.")
        try:
            raw = json.loads(payload)
            report = report_from_mapping(raw.pop("benchmark_report"))
            value = GraphRAGBaselineRecord(
                benchmark_report=report,
                **raw,
            )
        except Exception as exc:
            raise RuntimeError("stored baseline payload is corrupt.") from exc
        if (
            value.baseline_digest != row["baseline_digest"]
            or value.benchmark_fingerprint != row["benchmark_fingerprint"]
            or value.policy_id != row["policy_id"]
            or value.policy_digest != row["policy_digest"]
            or value.benchmark_report.report_digest
            != row["benchmark_report_digest"]
            or value.previous_baseline_digest
            != row["previous_baseline_digest"]
            or value.activation_regression_digest
            != row["activation_regression_digest"]
        ):
            raise RuntimeError("stored baseline row identity is corrupt.")
        return value

    def current(
        self,
        *,
        benchmark_fingerprint: str,
        policy_id: str,
    ) -> GraphRAGBaselineRecord | None:
        fingerprint = _digest(
            benchmark_fingerprint, "benchmark_fingerprint"
        )
        selected_policy = _identifier(policy_id, "policy_id", 200)
        with self._lock, self._connect() as connection:
            pointer = connection.execute(
                """
                SELECT * FROM graph_rag_baseline_current
                WHERE benchmark_fingerprint=? AND policy_id=?
                """,
                (fingerprint, selected_policy),
            ).fetchone()
            if pointer is None:
                return None
            row = connection.execute(
                """
                SELECT * FROM graph_rag_baselines
                WHERE benchmark_fingerprint=? AND policy_id=?
                  AND baseline_digest=?
                """,
                (
                    fingerprint,
                    selected_policy,
                    pointer["baseline_digest"],
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("baseline current pointer target is missing.")
        value = self._decode(row)
        if (
            value.policy_digest != pointer["policy_digest"]
            or value.benchmark_report.report_digest
            != pointer["benchmark_report_digest"]
        ):
            raise RuntimeError("baseline current pointer identity is corrupt.")
        return value

    def history(
        self,
        *,
        benchmark_fingerprint: str,
        policy_id: str,
        limit: int = 100,
    ) -> tuple[GraphRAGBaselineRecord, ...]:
        fingerprint = _digest(
            benchmark_fingerprint, "benchmark_fingerprint"
        )
        selected_policy = _identifier(policy_id, "policy_id", 200)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000.")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_rag_baselines
                WHERE benchmark_fingerprint=? AND policy_id=?
                ORDER BY activated_at DESC, baseline_digest DESC LIMIT ?
                """,
                (fingerprint, selected_policy, limit),
            ).fetchall()
        return tuple(self._decode(row) for row in rows)

    def activate(
        self,
        candidate: GraphRAGBenchmarkReport,
        policy: GraphRAGRegressionPolicy,
        *,
        expected_current_baseline_digest: str | None,
        regression: GraphRAGRegressionReport | None = None,
        now: float | None = None,
    ) -> GraphRAGBaselineRecord:
        if not isinstance(candidate, GraphRAGBenchmarkReport):
            raise ValueError("candidate must be GraphRAGBenchmarkReport.")
        if not isinstance(policy, GraphRAGRegressionPolicy):
            raise ValueError("policy must be GraphRAGRegressionPolicy.")
        expected = (
            None
            if expected_current_baseline_digest is None
            else _digest(
                expected_current_baseline_digest,
                "expected_current_baseline_digest",
            )
        )
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pointer = connection.execute(
                    """
                    SELECT * FROM graph_rag_baseline_current
                    WHERE benchmark_fingerprint=? AND policy_id=?
                    """,
                    (candidate.benchmark_fingerprint, policy.policy_id),
                ).fetchone()
                actual = None if pointer is None else pointer["baseline_digest"]
                if actual != expected:
                    raise RuntimeError(
                        "baseline current pointer differs from the explicit expectation."
                    )
                previous: GraphRAGBaselineRecord | None = None
                if pointer is not None:
                    row = connection.execute(
                        """
                        SELECT * FROM graph_rag_baselines
                        WHERE benchmark_fingerprint=? AND policy_id=?
                          AND baseline_digest=?
                        """,
                        (
                            candidate.benchmark_fingerprint,
                            policy.policy_id,
                            actual,
                        ),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError(
                            "baseline current pointer target is missing."
                        )
                    previous = self._decode(row)
                if previous is None:
                    if regression is not None:
                        raise ValueError(
                            "initial baseline activation may not use a regression report."
                        )
                    previous_digest = None
                    regression_digest = None
                else:
                    if not isinstance(regression, GraphRAGRegressionReport):
                        raise ValueError(
                            "replacement baseline activation requires a regression report."
                        )
                    if regression.decision != "eligible" or regression.reason_codes:
                        raise RuntimeError(
                            "replacement baseline regression report is not eligible."
                        )
                    if (
                        regression.benchmark_fingerprint
                        != candidate.benchmark_fingerprint
                        or regression.baseline_report_digest
                        != previous.benchmark_report.report_digest
                        or regression.candidate_report_digest
                        != candidate.report_digest
                        or regression.policy_id != policy.policy_id
                        or regression.policy_digest != policy.policy_digest
                    ):
                        raise RuntimeError(
                            "replacement regression identities differ from the baseline transition."
                        )
                    previous_digest = previous.baseline_digest
                    regression_digest = regression.report_digest
                value = GraphRAGBaselineRecord(
                    benchmark_fingerprint=candidate.benchmark_fingerprint,
                    benchmark_id=candidate.benchmark_id,
                    policy_id=policy.policy_id,
                    policy_digest=policy.policy_digest,
                    benchmark_report=candidate,
                    previous_baseline_digest=previous_digest,
                    activation_regression_digest=regression_digest,
                    activated_at=timestamp,
                )
                payload = self._encode(value)
                existing = connection.execute(
                    """
                    SELECT * FROM graph_rag_baselines
                    WHERE benchmark_fingerprint=? AND policy_id=?
                      AND baseline_digest=?
                    """,
                    (
                        value.benchmark_fingerprint,
                        value.policy_id,
                        value.baseline_digest,
                    ),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO graph_rag_baselines VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            value.benchmark_fingerprint,
                            value.policy_id,
                            value.baseline_digest,
                            value.policy_digest,
                            value.benchmark_report.report_digest,
                            value.previous_baseline_digest,
                            value.activation_regression_digest,
                            payload,
                            value.activated_at,
                        ),
                    )
                elif self._decode(existing).baseline_digest != value.baseline_digest:
                    raise RuntimeError("baseline identity collision detected.")
                connection.execute(
                    """
                    INSERT INTO graph_rag_baseline_current VALUES
                    (?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(benchmark_fingerprint, policy_id) DO UPDATE SET
                        baseline_digest=excluded.baseline_digest,
                        policy_digest=excluded.policy_digest,
                        benchmark_report_digest=excluded.benchmark_report_digest,
                        updated_at=excluded.updated_at,
                        schema_version=excluded.schema_version
                    """,
                    (
                        value.benchmark_fingerprint,
                        value.policy_id,
                        value.baseline_digest,
                        value.policy_digest,
                        value.benchmark_report.report_digest,
                        timestamp,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.current(
            benchmark_fingerprint=candidate.benchmark_fingerprint,
            policy_id=policy.policy_id,
        )
        if result is None:
            raise RuntimeError("activated baseline disappeared.")
        return result


def regression_report_from_mapping(
    value: Mapping[str, Any],
) -> GraphRAGRegressionReport:
    allowed = {
        "benchmark_fingerprint",
        "baseline_report_digest",
        "candidate_report_digest",
        "policy_id",
        "policy_digest",
        "decision",
        "reason_codes",
        "aggregate_deltas",
        "paired_intervals",
        "work_ratio",
        "report_digest",
        "paired_interval_method",
        "contains_raw_query",
        "contains_evidence_text",
        "runtime_policy_changed",
    }
    required = {
        "benchmark_fingerprint",
        "baseline_report_digest",
        "candidate_report_digest",
        "policy_id",
        "policy_digest",
        "decision",
        "reason_codes",
        "aggregate_deltas",
        "paired_intervals",
        "work_ratio",
    }
    if (
        not isinstance(value, Mapping)
        or not required <= set(value)
        or not set(value) <= allowed
    ):
        raise ValueError("graph RAG regression report schema is invalid.")
    if value.get("contains_raw_query") not in (None, False) or value.get(
        "contains_evidence_text"
    ) not in (None, False) or value.get("runtime_policy_changed") not in (
        None,
        False,
    ):
        raise ValueError("regression report contains unsupported privacy/runtime claims.")
    intervals_raw = value["paired_intervals"]
    if not isinstance(intervals_raw, Mapping):
        raise ValueError("paired_intervals must be an object.")
    report = GraphRAGRegressionReport(
        benchmark_fingerprint=value["benchmark_fingerprint"],
        baseline_report_digest=value["baseline_report_digest"],
        candidate_report_digest=value["candidate_report_digest"],
        policy_id=value["policy_id"],
        policy_digest=value["policy_digest"],
        decision=value["decision"],
        reason_codes=tuple(value["reason_codes"]),
        aggregate_deltas=value["aggregate_deltas"],
        paired_intervals={
            name: PairedMetricInterval(**interval)
            for name, interval in intervals_raw.items()
        },
        work_ratio=value["work_ratio"],
    )
    if value.get("report_digest") is not None and value[
        "report_digest"
    ] != report.report_digest:
        raise ValueError("regression report digest is invalid.")
    return report


__all__ = [
    "GraphRAGBaselineRecord",
    "GraphRAGBaselineStore",
    "regression_report_from_mapping",
]
