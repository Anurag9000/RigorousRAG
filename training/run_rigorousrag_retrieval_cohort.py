#!/usr/bin/env python3
"""Compile, audit and execute the RigorousRAG learned-retrieval dataset cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import authoritative_training_suite_catalog as catalog
from training.dataset_cohort_contract import model_spec_from_job, write_certificate
from training.rigorousrag_retrieval_cohort import build_registry, cohort_job_metadata
from tools.dataset_cohort_runtime_entry import load_runtime

runtime = load_runtime()
SCHEMA = "rigorousrag-retrieval-cohort-worker/v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_ROOT = ROOT / "artifacts" / "training_suite" / "dataset_cohorts" / "retrieval"
DEFAULT_CERTIFICATE = ROOT / "artifacts" / "training_suite" / "dataset_cohorts" / "retrieval_plan.json"


def _jobs() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for raw in catalog.iter_jobs("exhaustive"):
        if str(raw.get("phase")) != "training" or str(raw.get("dataset")) != "retrieval":
            continue
        row = dict(raw)
        config = row.get("recipe_config")
        if not isinstance(config, str) or not config:
            raise ValueError(f"{row.get('id')}: retrieval job has no recipe_config")
        config_path = (ROOT / config).resolve(strict=True)
        row.update(cohort_job_metadata(config_path))
        rows.append(row)
    expected = {
        "train:retrieval:dense-base", "train:retrieval:dense-distilled",
        "train:retrieval:splade-base", "train:retrieval:splade-distilled",
        "train:retrieval:unicoil-base", "train:retrieval:unicoil-distilled",
        "train:retrieval:colbert-base", "train:retrieval:colbert-distilled",
        "train:retrieval:cross-encoder-listwise",
    }
    actual = {str(row["id"]) for row in rows}
    if actual != expected:
        raise ValueError(f"retrieval cohort member drift: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    return tuple(rows)


def compile_plan() -> tuple[Any, Any, tuple[dict[str, Any], ...]]:
    jobs = _jobs()
    specs = tuple(model_spec_from_job(job) for job in jobs)
    plan = runtime.build_cohort_plan(specs)
    if len(plan.cohorts) != 1:
        raise ValueError(f"RigorousRAG retrieval must compile to exactly one dataset cohort, got {len(plan.cohorts)}")
    cohort = plan.cohorts[0]
    if cohort.dataset_key != "retrieval" or cohort.overlap or len(cohort.models) != 9:
        raise ValueError("retrieval cohort topology is not the authoritative 9-model single-dataset group")
    registry = build_registry(cohort)
    audit = registry.audit(specs, plan)
    if audit.get("pass") is not True:
        raise RuntimeError(f"retrieval cohort adapter audit failed: {audit.get('errors')}")
    return plan, registry, jobs


def report(plan: Any, registry: Any, jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cohort = plan.cohorts[0]
    return {
        "schema": SCHEMA,
        "pass": True,
        "dataset": cohort.dataset_key,
        "group_index": cohort.index,
        "overlap": cohort.overlap,
        "uniform_physical_batch_size": cohort.uniform_batch_size,
        "distinct_model_families": cohort.distinct_model_families,
        "member_count": len(cohort.models),
        "members": [model.job_id for model in cohort.models],
        "view_keys": sorted({model.view_key for model in cohort.models}),
        "cpu_variant": True,
        "gpu_first_variant": True,
        "plan": plan.to_dict(),
        "source_jobs": [str(job["id"]) for job in jobs],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan, registry, jobs = compile_plan()
    payload = report(plan, registry, jobs)
    write_certificate(args.certificate, payload)
    if args.audit_only:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    executor = runtime.CohortExecutor(registry, state_root=args.state_root)
    results = executor.run_plan(plan, backend=args.backend, seed=args.seed)
    output = {**payload, "execution": results}
    write_certificate(args.certificate, output)
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
