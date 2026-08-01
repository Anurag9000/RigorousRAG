"""Privacy-safe owner-scoped multi-hop diagnostic operations."""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

from tools.multihop_budget import MultiHopBudget
from tools.multihop_retrieval import MultiHopResult
from tools.multihop_trace_backend import MultiHopTraceBackend
from tools.multihop_trace_types import (
    MAX_HOPS,
    MultiHopTraceAggregate,
    MultiHopTraceHop,
    MultiHopTraceRecord,
    MultiHopTraceSummary,
    identifier,
    integer,
)
from tools.query_decomposition import DecompositionPlan
from tools.security import normalize_owner_id


class MultiHopTraceStore:
    """Persist aggregate diagnostics without raw questions or evidence."""

    def __init__(self, path: str | Any) -> None:
        self._backend = MultiHopTraceBackend(path)
        self.path = self._backend.path

    @staticmethod
    def _summary(row: sqlite3.Row) -> MultiHopTraceSummary:
        if any(row[name] not in (0, 1) for name in ("abstain", "exhausted", "used_model")):
            raise RuntimeError("Multi-hop trace boolean state is corrupt.")
        try:
            return MultiHopTraceSummary(
                run_id=row["run_id"],
                owner_id=row["owner_id"],
                plan_fingerprint=row["plan_fingerprint"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                subquestion_count=row["subquestion_count"],
                batch_count=row["batch_count"],
                terminal_count=row["terminal_count"],
                evidence_count=row["evidence_count"],
                join_count=row["join_count"],
                terminal_evidence_count=row["terminal_evidence_count"],
                abstain=bool(row["abstain"]),
                exhausted=bool(row["exhausted"]),
                used_model=bool(row["used_model"]),
                planner_quality=row["planner_quality"],
                budget_limit=row["budget_limit"],
                allocated_budget=row["allocated_budget"],
                error_hops=row["error_hops"],
                timeout_hops=row["timeout_hops"],
                skipped_hops=row["skipped_hops"],
            )
        except ValueError as exc:
            raise RuntimeError("Multi-hop trace summary is corrupt.") from exc

    @staticmethod
    def _hops(
        connection: sqlite3.Connection, run_id: str
    ) -> tuple[MultiHopTraceHop, ...]:
        rows = connection.execute(
            "SELECT * FROM multihop_hops WHERE run_id=? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        result: list[MultiHopTraceHop] = []
        try:
            for row in rows:
                result.append(
                    MultiHopTraceHop(
                        sequence=row["sequence"],
                        hop_id=row["hop_id"],
                        dependency_count=row["dependency_count"],
                        status=row["status"],
                        returned_evidence=row["returned_evidence"],
                        accepted_evidence=row["accepted_evidence"],
                        error_type=row["error_type"],
                    )
                )
        except ValueError as exc:
            raise RuntimeError("Multi-hop trace hop state is corrupt.") from exc
        if tuple(item.sequence for item in result) != tuple(range(len(result))):
            raise RuntimeError("Multi-hop trace sequence is corrupt.")
        return tuple(result)

    def record_result(
        self,
        *,
        owner_id: str,
        plan: DecompositionPlan,
        retrieval: MultiHopResult,
        budget: MultiHopBudget,
        used_model: bool,
        planner_quality: float,
        run_id: str | None = None,
        started_at: float | None = None,
        completed_at: float | None = None,
    ) -> str:
        if not isinstance(plan, DecompositionPlan):
            raise ValueError("plan must be a DecompositionPlan.")
        if not isinstance(retrieval, MultiHopResult):
            raise ValueError("retrieval must be a MultiHopResult.")
        if not isinstance(budget, MultiHopBudget):
            raise ValueError("budget must be a MultiHopBudget.")
        if not isinstance(used_model, bool):
            raise ValueError("used_model must be a boolean.")
        if retrieval.plan_fingerprint != plan.fingerprint:
            raise ValueError("retrieval and decomposition plan fingerprints differ.")
        traces = tuple(retrieval.traces)
        if len(traces) != len(plan.subquestions):
            raise ValueError("retrieval trace count does not match the decomposition plan.")
        summary = MultiHopTraceSummary(
            run_id=run_id or uuid.uuid4().hex,
            owner_id=normalize_owner_id(owner_id),
            plan_fingerprint=plan.fingerprint,
            started_at=time.time() if started_at is None else started_at,
            completed_at=time.time() if completed_at is None else completed_at,
            subquestion_count=len(plan.subquestions),
            batch_count=len(plan.batches),
            terminal_count=len(plan.terminal_questions),
            evidence_count=len(retrieval.evidence),
            join_count=len(retrieval.joins),
            terminal_evidence_count=retrieval.terminal_evidence_count,
            abstain=retrieval.abstain,
            exhausted=retrieval.exhausted,
            used_model=used_model,
            planner_quality=planner_quality,
            budget_limit=budget.total_limit,
            allocated_budget=budget.allocated_cost,
            error_hops=sum(trace.status == "error" for trace in traces),
            timeout_hops=sum("timeout" in trace.status for trace in traces),
            skipped_hops=sum(trace.status.startswith("skipped_") for trace in traces),
        )
        with self._backend.lock, self._backend.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM multihop_runs WHERE run_id=?", (summary.run_id,)
            ).fetchone()
            if existing is not None:
                if self._summary(existing) == summary:
                    return summary.run_id
                raise ValueError("run_id already identifies a different multi-hop trace.")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO multihop_runs VALUES("
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    summary.run_id,
                    summary.owner_id,
                    summary.plan_fingerprint,
                    summary.started_at,
                    summary.completed_at,
                    summary.subquestion_count,
                    summary.batch_count,
                    summary.terminal_count,
                    summary.evidence_count,
                    summary.join_count,
                    summary.terminal_evidence_count,
                    int(summary.abstain),
                    int(summary.exhausted),
                    int(summary.used_model),
                    summary.planner_quality,
                    summary.budget_limit,
                    summary.allocated_budget,
                    summary.error_hops,
                    summary.timeout_hops,
                    summary.skipped_hops,
                ),
            )
            connection.executemany(
                "INSERT INTO multihop_hops VALUES(?,?,?,?,?,?,?,?)",
                [
                    (
                        summary.run_id,
                        sequence,
                        trace.hop_id,
                        len(trace.dependencies),
                        trace.status,
                        trace.returned_evidence,
                        trace.accepted_evidence,
                        trace.error_type,
                    )
                    for sequence, trace in enumerate(traces)
                ],
            )
            connection.commit()
        self._backend.verify_identity()
        return summary.run_id

    def get_run(
        self, *, owner_id: str, run_id: str
    ) -> MultiHopTraceRecord | None:
        owner = normalize_owner_id(owner_id)
        selected = identifier(run_id, "run_id")
        with self._backend.lock, self._backend.connect() as connection:
            row = connection.execute(
                "SELECT * FROM multihop_runs WHERE owner_id=? AND run_id=?",
                (owner, selected),
            ).fetchone()
            if row is None:
                return None
            summary = self._summary(row)
            hops = self._hops(connection, selected)
            if len(hops) != summary.subquestion_count:
                raise RuntimeError("Multi-hop trace hop count is corrupt.")
            result = MultiHopTraceRecord(summary, hops)
        self._backend.verify_identity()
        return result

    def list_runs(
        self, *, owner_id: str, limit: int = 100
    ) -> tuple[MultiHopTraceSummary, ...]:
        owner = normalize_owner_id(owner_id)
        bounded = integer(limit, "limit", 1, 1_000)
        with self._backend.lock, self._backend.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM multihop_runs WHERE owner_id=? "
                "ORDER BY completed_at DESC, run_id DESC LIMIT ?",
                (owner, bounded),
            ).fetchall()
            result = tuple(self._summary(row) for row in rows)
        self._backend.verify_identity()
        return result

    def aggregate(
        self, *, owner_id: str, limit: int = 1_000
    ) -> MultiHopTraceAggregate:
        summaries = self.list_runs(owner_id=owner_id, limit=limit)
        if not summaries:
            return MultiHopTraceAggregate(0, 0, 0, 0, 0, 0, 0.0, 0.0, ())
        run_ids = [summary.run_id for summary in summaries]
        placeholders = ",".join("?" for _ in run_ids)
        with self._backend.lock, self._backend.connect() as connection:
            rows = connection.execute(
                f"SELECT status, COUNT(*) count FROM multihop_hops "
                f"WHERE run_id IN ({placeholders}) "
                "GROUP BY status ORDER BY status",
                run_ids,
            ).fetchall()
        return MultiHopTraceAggregate(
            run_count=len(summaries),
            abstention_count=sum(summary.abstain for summary in summaries),
            exhausted_count=sum(summary.exhausted for summary in summaries),
            model_plan_count=sum(summary.used_model for summary in summaries),
            error_run_count=sum(summary.error_hops > 0 for summary in summaries),
            timeout_run_count=sum(summary.timeout_hops > 0 for summary in summaries),
            mean_planner_quality=round(
                sum(summary.planner_quality for summary in summaries)
                / len(summaries),
                9,
            ),
            mean_allocated_budget=round(
                sum(summary.allocated_budget for summary in summaries)
                / len(summaries),
                9,
            ),
            hop_statuses=tuple(
                (row["status"], int(row["count"])) for row in rows
            ),
        )

    def prune_owner(
        self, *, owner_id: str, retain_latest: int = 10_000
    ) -> int:
        owner = normalize_owner_id(owner_id)
        retain = integer(retain_latest, "retain_latest", 0, 1_000_000)
        with self._backend.lock, self._backend.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM multihop_runs WHERE owner_id=? AND run_id IN ("
                "SELECT run_id FROM multihop_runs WHERE owner_id=? "
                "ORDER BY completed_at DESC, run_id DESC LIMIT -1 OFFSET ?)",
                (owner, owner, retain),
            )
            connection.commit()
            deleted = max(int(cursor.rowcount), 0)
        self._backend.verify_identity()
        return deleted

    def ping(self) -> bool:
        return self._backend.ping()


__all__ = ["MultiHopTraceStore"]
