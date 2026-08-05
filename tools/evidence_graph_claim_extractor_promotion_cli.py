"""Privacy-conscious CLI for claim extractor assessment, promotion, and rollback."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

from tools.evidence_graph_claim_extractor_benchmark import (
    ScientificClaimExtractorBenchmarkCase,
    ScientificClaimExtractorBenchmarkSuite,
)
from tools.evidence_graph_claim_extractor_promotion import (
    GovernedScientificClaimExtractorPromotionService,
    assess_scientific_claim_extractor_promotion,
)
from tools.evidence_graph_claim_extractor_promotion_runtime import (
    get_scientific_claim_extractor_promotion_policy,
    get_scientific_claim_extractor_promotion_store,
)
from tools.evidence_graph_claim_extractor_runtime import (
    get_scientific_claim_extractor_registry,
)
from tools.evidence_graph_relation_actor import load_relation_review_actor

_MAX_SUITE_BYTES = 20_000_000
_MAX_PATH = 4096
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("benchmark suite contains a duplicate JSON key.")
        result[key] = value
    return result


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("benchmark suite path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("benchmark suite path is invalid.")
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
            raise ValueError("benchmark suite path may not contain redirects.")
    return absolute


def _read_suite(path: str | os.PathLike[str]) -> ScientificClaimExtractorBenchmarkSuite:
    selected = _path(path)
    descriptor = os.open(
        selected,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_SUITE_BYTES:
            raise ValueError("benchmark suite file is invalid or too large.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("benchmark suite changed while reading.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("benchmark suite grew while reading.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_size) != int(before.st_size)
        ):
            raise RuntimeError("benchmark suite identity changed while reading.")
    finally:
        os.close(descriptor)
    try:
        raw = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("benchmark suite JSON is invalid.") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "benchmark_id",
        "owner_id",
        "extractor_name",
        "extractor_version",
        "extractor_record_digest",
        "case_count",
        "gold_count",
        "proposal_count",
        "matched_count",
        "precision",
        "recall",
        "f1",
        "exact_evidence_accuracy",
        "exact_locator_accuracy",
        "mean_span_iou",
        "mean_claim_token_f1",
        "claim_type_accuracy",
        "modality_accuracy",
        "confidence_brier_score",
        "cases",
        "suite_digest",
        "contains_claim_text",
        "contains_evidence_text",
        "schema_version",
    }:
        raise ValueError("benchmark suite schema is invalid.")
    if not isinstance(raw["cases"], list):
        raise ValueError("benchmark suite cases must be an array.")
    cases = tuple(
        ScientificClaimExtractorBenchmarkCase(**value)
        for value in raw["cases"]
    )
    value = dict(raw)
    value["cases"] = cases
    return ScientificClaimExtractorBenchmarkSuite(**value)


def _activation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "activation_id": value.activation_id,
        "owner_id": value.owner_id,
        "extractor_name": value.extractor_name,
        "extractor_version": value.extractor_version,
        "extractor_record_digest": value.extractor_record_digest,
        "promotion_report_digest": value.promotion_report_digest,
        "action": value.action,
        "previous_activation_id": value.previous_activation_id,
        "actor_id": value.actor_id,
        "actor_binding_method": value.actor_binding_method,
        "actor_binding_digest": value.actor_binding_digest,
        "activated_at": value.activated_at,
        "activation_digest": value.activation_digest,
    }


def _report(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "extractor_name": value.extractor_name,
        "extractor_version": value.extractor_version,
        "extractor_record_digest": value.extractor_record_digest,
        "benchmark_id": value.benchmark_id,
        "benchmark_suite_digest": value.benchmark_suite_digest,
        "policy_digest": value.policy_digest,
        "thresholds_digest": value.thresholds_digest,
        "eligible": value.eligible,
        "reasons": list(value.reasons),
        "assessed_at": value.assessed_at,
        "report_digest": value.report_digest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_claim_extractor_promotion_cli",
        description=(
            "Assess, promote, inspect, or roll back exact claim extractor versions. "
            "Benchmark suite files and outputs contain no claim or evidence text."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    assess = commands.add_parser("assess")
    assess.add_argument("suite_path")

    promote = commands.add_parser("promote")
    promote.add_argument("suite_path")
    promote.add_argument("--expected-current-activation-id")

    current = commands.add_parser("current")
    current.add_argument("--owner-id", required=True)
    current.add_argument("--extractor-name", required=True)

    history = commands.add_parser("history")
    history.add_argument("--owner-id", required=True)
    history.add_argument("--extractor-name", required=True)
    history.add_argument("--limit", type=int, default=100)

    resolve = commands.add_parser("resolve")
    resolve.add_argument("--owner-id", required=True)
    resolve.add_argument("--extractor-name", required=True)

    rollback = commands.add_parser("rollback")
    rollback.add_argument("--target-promotion-report-digest", required=True)
    rollback.add_argument("--expected-current-activation-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry = get_scientific_claim_extractor_registry()
        store = get_scientific_claim_extractor_promotion_store()
        if args.command == "current":
            value = store.current(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
            )
            _print(
                {
                    "current": _activation(value),
                    "mutation_performed": False,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        if args.command == "history":
            values = store.history(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                limit=args.limit,
            )
            _print(
                {
                    "item_count": len(values),
                    "items": [_activation(value) for value in values],
                    "mutation_performed": False,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        policy = get_scientific_claim_extractor_promotion_policy()
        service = GovernedScientificClaimExtractorPromotionService(
            extractor_registry=registry,
            promotion_store=store,
            policy=policy,
        )
        if args.command == "assess":
            suite = _read_suite(args.suite_path)
            record = registry.require_active(
                owner_id=suite.owner_id,
                extractor_name=suite.extractor_name,
                extractor_version=suite.extractor_version,
            )
            report = assess_scientific_claim_extractor_promotion(
                extractor_record=record,
                benchmark_suite=suite,
                policy=policy,
            )
            _print(
                {
                    **_report(report),
                    "mutation_performed": False,
                    "activation_performed": False,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        if args.command == "promote":
            suite = _read_suite(args.suite_path)
            report, activation = service.promote(
                benchmark_suite=suite,
                expected_current_activation_id=(
                    args.expected_current_activation_id
                ),
                actor=load_relation_review_actor(),
            )
            _print(
                {
                    "report": _report(report),
                    "activation": _activation(activation),
                    "mutation_performed": True,
                    "activation_performed": activation is not None,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        if args.command == "resolve":
            record = service.resolve_current(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
            )
            current = store.current(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
            )
            _print(
                {
                    "current": _activation(current),
                    "extractor_version": record.extractor_version,
                    "extractor_record_digest": record.record_digest,
                    "state": record.state,
                    "mutation_performed": False,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        if args.command == "rollback":
            activation = service.rollback(
                target_promotion_report_digest=(
                    args.target_promotion_report_digest
                ),
                expected_current_activation_id=(
                    args.expected_current_activation_id
                ),
                actor=load_relation_review_actor(),
            )
            _print(
                {
                    "activation": _activation(activation),
                    "mutation_performed": True,
                    "rollback_performed": True,
                    "contains_claim_text": False,
                    "contains_evidence_text": False,
                }
            )
            return 0
        raise ValueError("unsupported claim extractor promotion command.")
    except (KeyError, OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
