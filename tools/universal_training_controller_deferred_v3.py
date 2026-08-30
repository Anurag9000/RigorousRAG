#!/usr/bin/env python3
"""Strict deferred-producer validation for universal controller v18.

The original deferred fan-out layer correctly restricted pre-audit execution to
non-training producer jobs, but its exact-checkpoint branch contained a tautology
that could accept an ``exact_checkpoint`` producer without an explicit
``checkpoint_contract.exact_resume=true`` declaration.  v3 closes that hole
without changing any resource scheduling semantics.
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

import universal_training_controller_deferred as core
import universal_training_controller_deferred_v2 as hardened
import universal_training_controller_training_contracts as training_contracts


def _strict_validate_producers(
    records: Sequence[Mapping[str, object]],
    producer_ids: Sequence[str],
) -> None:
    index = core._by_id(records)
    producers = [index[job_id] for job_id in producer_ids]
    core.dag._dependency_waves([dict(record) for record in producers])
    failures: List[str] = []
    for job in producers:
        job_id = str(job.get("id") or "")
        if training_contracts._is_training_job(job):
            failures.append(
                f"{job_id}: training job cannot be a deferred-expansion prerequisite"
            )
            continue
        strategy = str(job.get("resume_strategy") or "").strip().lower()
        contract = job.get("checkpoint_contract")
        declared_exact = (
            isinstance(contract, Mapping)
            and contract.get("exact_resume") is True
        )
        if strategy == "restart_exact":
            if not all(
                job.get(key) is True
                for key in ("deterministic", "idempotent", "atomic_outputs")
            ):
                failures.append(
                    f"{job_id}: restart_exact producer lacks "
                    "deterministic/idempotent/atomic_outputs"
                )
        elif strategy in {"exact_checkpoint", "framework_exact_checkpoint"}:
            if not declared_exact:
                failures.append(
                    f"{job_id}: {strategy} producer lacks explicit "
                    "checkpoint_contract.exact_resume=true"
                )
        else:
            failures.append(
                f"{job_id}: producer durability is not exact/restart_exact "
                f"({strategy or 'missing'})"
            )
    if failures:
        raise SystemExit(
            "Deferred producer preflight failed:\n  - " + "\n  - ".join(failures)
        )


def main() -> int:
    original = core._validate_producers
    core._validate_producers = _strict_validate_producers
    try:
        return hardened.main()
    finally:
        core._validate_producers = original


if __name__ == "__main__":
    raise SystemExit(main())
