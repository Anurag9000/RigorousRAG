"""Durable resumable storage for text-free evidence-graph benchmark runs."""

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
from typing import Any, Callable

from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkCase,
    GraphRAGBenchmarkReport,
    GraphRAGBenchmarkRun,
    GraphRAGBenchmarkRunReport,
    evaluate_graph_observation,
)
from tools.evidence_graph_rag_evaluation import (
    GraphRAGAggregate,
    aggregate_graph_evaluations,
)
from tools.evidence_graph_rag_live_benchmark import (
    GraphRAGLiveBenchmarkPlan,
    GraphRAGLiveBenchmarkResult,
    _query,
    observation_from_selection,
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
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


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


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("run-store path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("run-store path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or bool(
            int(getattr(info, "st_file_attributes", 0)) & _REPARSE
        ):
            raise ValueError("run-store path may not contain redirects.")
    return absolute


def _benchmark_fingerprint(plan: GraphRAGLiveBenchmarkPlan) -> str:
    return _sha256(
        {
            "scope": "rigorousrag-evidence-graph-rag-benchmark-v1",
            "benchmark_id": plan.benchmark_id,
            "ordered_gold_case_digests": [value.case_digest for value in plan.gold_cases],
            "run_seeds": list(plan.run_seeds),
            "schema_version": 1,
        }
    )


def _run_id(index: int, seed: int) -> str:
    return f"run-{index:04d}-seed-{seed}"


def _run_contract_digest(run_id: str, seed: int, plan: GraphRAGLiveBenchmarkPlan) -> str:
    return _sha256(
        {
            "run_id": run_id,
            "seed": seed,
            "gold_case_digests": [value.case_digest for value in plan.gold_cases],
        }
    )


@dataclass(frozen=True)
class GraphRAGStoredRun:
    plan_fingerprint: str
    benchmark_fingerprint: str
    benchmark_id: str
    run_id: str
    seed: int
    case_count: int
    run_contract_digest: str
    run_report: GraphRAGBenchmarkRunReport
    completed_at: float

    def __post_init__(self) -> None:
        for name in (
            "plan_fingerprint",
            "benchmark_fingerprint",
            "run_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id", 500))
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id", 500))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0, 2**63 - 1))
        object.__setattr__(self, "case_count", _integer(self.case_count, "case_count", 1, 1_000_000))
        if not isinstance(self.run_report, GraphRAGBenchmarkRunReport):
            raise ValueError("run_report must be GraphRAGBenchmarkRunReport.")
        if (
            self.run_report.run_id != self.run_id
            or self.run_report.seed != self.seed
            or self.run_report.run_contract_digest != self.run_contract_digest
            or self.run_report.aggregate.case_count != self.case_count
        ):
            raise ValueError("stored run identities differ from its report.")
        object.__setattr__(self, "completed_at", _timestamp(self.completed_at, "completed_at"))

    @property
    def stored_run_digest(self) -> str:
        return _sha256(
            {
                "plan_fingerprint": self.plan_fingerprint,
                "benchmark_fingerprint": self.benchmark_fingerprint,
                "benchmark_id": self.benchmark_id,
                "run_id": self.run_id,
                "seed": self.seed,
                "case_count": self.case_count,
                "run_contract_digest": self.run_contract_digest,
                "run_report_digest": self.run_report.report_digest,
            }
        )


class GraphRAGBenchmarkRunStore:
    """Append-only completed runs keyed by exact plan and run contract."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode):
            raise ValueError("run-store parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("run-store database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity:
            raise RuntimeError("run-store parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("run-store database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_rag_runs (
                    plan_fingerprint TEXT NOT NULL,
                    benchmark_fingerprint TEXT NOT NULL,
                    benchmark_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    case_count INTEGER NOT NULL,
                    run_contract_digest TEXT NOT NULL,
                    run_report_digest TEXT NOT NULL,
                    stored_run_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    completed_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(plan_fingerprint, run_id)
                );
                CREATE INDEX IF NOT EXISTS graph_rag_run_history
                    ON graph_rag_runs(benchmark_fingerprint, completed_at, run_id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> GraphRAGStoredRun:
        if int(row["schema_version"]) != _SCHEMA_VERSION:
            raise RuntimeError("stored run schema is unsupported.")
        payload = row["payload_json"]
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise RuntimeError("stored run payload is corrupt.")
        try:
            raw = json.loads(payload)
            report_raw = raw.pop("run_report")
            aggregate = GraphRAGAggregate(**report_raw.pop("aggregate"))
            report = GraphRAGBenchmarkRunReport(aggregate=aggregate, **report_raw)
            value = GraphRAGStoredRun(run_report=report, **raw)
        except Exception as exc:
            raise RuntimeError("stored run payload is corrupt.") from exc
        if (
            value.stored_run_digest != row["stored_run_digest"]
            or value.run_report.report_digest != row["run_report_digest"]
            or value.plan_fingerprint != row["plan_fingerprint"]
            or value.run_id != row["run_id"]
        ):
            raise RuntimeError("stored run row identity is corrupt.")
        return value

    def write(self, value: GraphRAGStoredRun) -> GraphRAGStoredRun:
        if not isinstance(value, GraphRAGStoredRun):
            raise ValueError("value must be GraphRAGStoredRun.")
        payload = json.dumps(
            asdict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            raise ValueError("run payload exceeds the byte limit.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM graph_rag_runs WHERE plan_fingerprint=? AND run_id=?",
                    (value.plan_fingerprint, value.run_id),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO graph_rag_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            value.plan_fingerprint,
                            value.benchmark_fingerprint,
                            value.benchmark_id,
                            value.run_id,
                            value.seed,
                            value.case_count,
                            value.run_contract_digest,
                            value.run_report.report_digest,
                            value.stored_run_digest,
                            payload,
                            value.completed_at,
                        ),
                    )
                elif self._decode(row).stored_run_digest != value.stored_run_digest:
                    raise RuntimeError("completed run already exists with different results.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(value.plan_fingerprint, value.run_id)
        if result is None:
            raise RuntimeError("stored run disappeared after write.")
        return result

    def get(self, plan_fingerprint: str, run_id: str) -> GraphRAGStoredRun | None:
        plan = _digest(plan_fingerprint, "plan_fingerprint")
        selected = _identifier(run_id, "run_id", 500)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_rag_runs WHERE plan_fingerprint=? AND run_id=?",
                (plan, selected),
            ).fetchone()
        return None if row is None else self._decode(row)

    def list_plan(self, plan_fingerprint: str) -> tuple[GraphRAGStoredRun, ...]:
        plan = _digest(plan_fingerprint, "plan_fingerprint")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM graph_rag_runs WHERE plan_fingerprint=? ORDER BY run_id",
                (plan,),
            ).fetchall()
        return tuple(self._decode(row) for row in rows)

    def remove_plan(self, plan_fingerprint: str, *, confirm_plan_fingerprint: str) -> bool:
        plan = _digest(plan_fingerprint, "plan_fingerprint")
        confirmation = _digest(confirm_plan_fingerprint, "confirm_plan_fingerprint")
        if plan != confirmation:
            raise ValueError("confirmation must exactly match plan_fingerprint.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "DELETE FROM graph_rag_runs WHERE plan_fingerprint=?", (plan,)
                )
                connection.execute("COMMIT")
                return cursor.rowcount > 0
            except Exception:
                connection.execute("ROLLBACK")
                raise


def _run_report(run: GraphRAGBenchmarkRun) -> GraphRAGBenchmarkRunReport:
    evaluations = tuple(
        evaluate_graph_observation(value.observation, value.gold) for value in run.cases
    )
    return GraphRAGBenchmarkRunReport(
        run_id=run.run_id,
        seed=run.seed,
        aggregate=aggregate_graph_evaluations(evaluations),
        run_contract_digest=run.run_contract_digest,
        run_result_digest=run.run_result_digest,
    )


def _assemble_report(
    plan: GraphRAGLiveBenchmarkPlan,
    values: tuple[GraphRAGStoredRun, ...],
) -> GraphRAGBenchmarkReport:
    expected = [
        (_run_id(index, seed), seed, _run_contract_digest(_run_id(index, seed), seed, plan))
        for index, seed in enumerate(plan.run_seeds)
    ]
    lookup = {(value.run_id, value.seed): value for value in values}
    if set(lookup) != {(run_id, seed) for run_id, seed, _digest_value in expected}:
        raise RuntimeError("completed run set differs from the benchmark plan.")
    reports = []
    evaluations = []
    for run_id, seed, contract_digest in expected:
        stored = lookup[(run_id, seed)]
        if (
            stored.plan_fingerprint != plan.plan_fingerprint
            or stored.benchmark_fingerprint != _benchmark_fingerprint(plan)
            or stored.run_contract_digest != contract_digest
            or stored.case_count != len(plan.gold_cases)
        ):
            raise RuntimeError("completed run identity differs from the benchmark plan.")
        reports.append(stored.run_report)
        evaluations.extend(stored.run_report.aggregate.evaluation_digests)
    count = len(reports)

    def mean(name: str) -> float:
        return sum(float(getattr(value.aggregate, name)) for value in reports) / count

    aggregate = GraphRAGAggregate(
        case_count=count * len(plan.gold_cases),
        macro_node_precision=mean("macro_node_precision"),
        macro_node_recall=mean("macro_node_recall"),
        macro_node_f1=mean("macro_node_f1"),
        macro_document_precision=mean("macro_document_precision"),
        macro_document_recall=mean("macro_document_recall"),
        macro_document_f1=mean("macro_document_f1"),
        macro_edge_precision=mean("macro_edge_precision"),
        macro_edge_recall=mean("macro_edge_recall"),
        macro_edge_f1=mean("macro_edge_f1"),
        complete_required_path_rate=mean("complete_required_path_rate"),
        mean_lineage_completeness=mean("mean_lineage_completeness"),
        abstention_accuracy=mean("abstention_accuracy"),
        mean_evidence_count=mean("mean_evidence_count"),
        mean_traversal_count=mean("mean_traversal_count"),
        mean_estimated_work_units=mean("mean_estimated_work_units"),
        evaluation_digests=tuple(evaluations),
    )
    return GraphRAGBenchmarkReport(
        benchmark_id=plan.benchmark_id,
        benchmark_fingerprint=_benchmark_fingerprint(plan),
        run_count=len(reports),
        seed_count=len(set(plan.run_seeds)),
        case_count_per_run=len(plan.gold_cases),
        run_reports=tuple(reports),
        aggregate=aggregate,
    )


@dataclass(frozen=True)
class GraphRAGResumableBenchmarkResult:
    result: GraphRAGLiveBenchmarkResult
    executed_run_count: int
    reused_run_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.result, GraphRAGLiveBenchmarkResult):
            raise ValueError("result must be GraphRAGLiveBenchmarkResult.")
        object.__setattr__(
            self,
            "executed_run_count",
            _integer(self.executed_run_count, "executed_run_count", 0, 10_000),
        )
        object.__setattr__(
            self,
            "reused_run_count",
            _integer(self.reused_run_count, "reused_run_count", 0, 10_000),
        )
        if self.executed_run_count + self.reused_run_count != self.result.report.run_count:
            raise ValueError("executed/reused counts must match report run_count.")


def execute_resumable_live_graph_rag_benchmark(
    plan: GraphRAGLiveBenchmarkPlan,
    *,
    query_resolver: Callable[[str], str],
    selection_runner: Callable[..., Any],
    store: GraphRAGBenchmarkRunStore,
    now: Callable[[], float] = time.time,
) -> GraphRAGResumableBenchmarkResult:
    if not isinstance(plan, GraphRAGLiveBenchmarkPlan):
        raise ValueError("plan must be GraphRAGLiveBenchmarkPlan.")
    if not isinstance(store, GraphRAGBenchmarkRunStore):
        raise ValueError("store must be GraphRAGBenchmarkRunStore.")
    if not callable(query_resolver) or not callable(selection_runner) or not callable(now):
        raise ValueError("query_resolver, selection_runner and now must be callable.")
    benchmark_fingerprint = _benchmark_fingerprint(plan)
    executed = 0
    reused = 0
    for index, seed in enumerate(plan.run_seeds):
        run_id = _run_id(index, seed)
        contract_digest = _run_contract_digest(run_id, seed, plan)
        existing = store.get(plan.plan_fingerprint, run_id)
        if existing is not None:
            if (
                existing.benchmark_fingerprint != benchmark_fingerprint
                or existing.run_contract_digest != contract_digest
                or existing.case_count != len(plan.gold_cases)
            ):
                raise RuntimeError("stored completed run differs from the plan contract.")
            reused += 1
            continue
        cases = []
        for gold in plan.gold_cases:
            query = _query(query_resolver(gold.query_id), gold.query_digest)
            selection = selection_runner(
                query=query,
                query_id=gold.query_id,
                seed=seed,
                selector_config=dict(plan.selector_config),
            )
            cases.append(
                GraphRAGBenchmarkCase(
                    gold=gold,
                    observation=observation_from_selection(selection),
                )
            )
            del query
            del selection
        run = GraphRAGBenchmarkRun(run_id=run_id, seed=seed, cases=tuple(cases))
        if run.run_contract_digest != contract_digest:
            raise RuntimeError("executed run contract differs from the plan contract.")
        stored = GraphRAGStoredRun(
            plan_fingerprint=plan.plan_fingerprint,
            benchmark_fingerprint=benchmark_fingerprint,
            benchmark_id=plan.benchmark_id,
            run_id=run_id,
            seed=seed,
            case_count=len(plan.gold_cases),
            run_contract_digest=contract_digest,
            run_report=_run_report(run),
            completed_at=_timestamp(now(), "completed_at"),
        )
        store.write(stored)
        executed += 1
    report = _assemble_report(plan, store.list_plan(plan.plan_fingerprint))
    return GraphRAGResumableBenchmarkResult(
        result=GraphRAGLiveBenchmarkResult(
            plan_fingerprint=plan.plan_fingerprint,
            report=report,
        ),
        executed_run_count=executed,
        reused_run_count=reused,
    )


__all__ = [
    "GraphRAGBenchmarkRunStore",
    "GraphRAGResumableBenchmarkResult",
    "GraphRAGStoredRun",
    "execute_resumable_live_graph_rag_benchmark",
]
