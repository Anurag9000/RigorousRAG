#!/usr/bin/env python3
"""Authoritative deterministic calibration/materialization jobs for RigorousRAG.

These jobs fit thresholds/calibrators from already-produced held-out scores. They are
scheduler-visible calibration jobs, not iterative neural/classical training jobs: each
execution is deterministic from immutable input bytes + config + source revision and
publishes one atomic content-addressed result manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.calibration import CalibrationExample as EvaluationExample
from evaluation.calibration import HistogramCalibrator
from evaluation.conformal import fit_nonconformity_threshold, fit_retrieval_calibration
from evaluation.conformal_retrieval import fit_split_conformal_threshold
from tools.calibration import optimize_threshold
from tools.confidence_calibration import CalibrationExample as ConfidenceExample
from tools.confidence_calibration import fit_isotonic_calibrator as fit_confidence_isotonic
from tools.cross_profile_fusion import (
    CalibrationContract,
    RetrieverScoreProfile,
    ScoreCalibrationExample,
    ScoreDirection,
    fit_isotonic_calibrator as fit_cross_profile_isotonic,
)

SCHEMA = "rigorousrag-authoritative-calibration-job/v1"
RESULT_SCHEMA = "rigorousrag-authoritative-calibration-result/v1"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HEX = frozenset("0123456789abcdef")
_KINDS = frozenset(
    {
        "evaluation_histogram",
        "evaluation_conformal_nonconformity",
        "evaluation_conformal_retrieval",
        "retrieval_split_conformal",
        "threshold_decision",
        "confidence_isotonic",
        "cross_profile_isotonic",
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(value: Any) -> str:
    requested = str(value if value is not None else "auto").strip().lower()
    if requested == "auto":
        try:
            requested = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.STDOUT, text=True
            ).strip().lower()
        except Exception as exc:
            raise RuntimeError("source_revision=auto requires an exact Git checkout") from exc
    if requested in {"0" * 40, "0" * 64}:
        raise ValueError("source_revision must not be an all-zero placeholder")
    if len(requested) not in {40, 64} or any(ch not in _HEX for ch in requested):
        raise ValueError("source_revision must be auto or a full 40/64-character hexadecimal Git object id")
    return requested


def _resolve_path(config_path: Path, value: Any, label: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(value).expanduser()
    selected = raw if raw.is_absolute() else config_path.parent / raw
    # Fail if any existing lexical component is a symlink before resolving it away.
    probe = selected.absolute()
    current = Path(probe.anchor) if probe.is_absolute() else Path()
    for part in probe.parts[1:] if probe.is_absolute() else probe.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
    selected = selected.resolve(strict=must_exist)
    if must_exist and not selected.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return selected


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON input {path}") from exc


def _expected_digest(config: Mapping[str, Any], actual: str) -> None:
    expected = config.get("expected_input_sha256")
    if expected is None:
        return
    if not isinstance(expected, str):
        raise ValueError("expected_input_sha256 must be a SHA-256 string")
    expected = expected.strip().lower()
    if expected == "0" * 64:
        raise ValueError("expected_input_sha256 must not be an all-zero placeholder")
    if len(expected) != 64 or any(ch not in _HEX for ch in expected):
        raise ValueError("expected_input_sha256 must be a SHA-256 string")
    if expected != actual:
        raise ValueError(f"input digest mismatch: expected {expected}, got {actual}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except Exception:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _evaluation_examples(payload: Any) -> tuple[EvaluationExample, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("evaluation histogram input must be a non-empty JSON list")
    return tuple(
        EvaluationExample(
            confidence=row["confidence"],
            correct=row["correct"],
            weight=row.get("weight", 1.0),
        )
        for row in payload
        if isinstance(row, Mapping)
    )


def _run(kind: str, payload: Any, config: Mapping[str, Any]) -> Any:
    options = config.get("options") if isinstance(config.get("options"), Mapping) else {}
    if kind == "evaluation_histogram":
        rows = _evaluation_examples(payload)
        calibrator = HistogramCalibrator(
            bin_count=int(options.get("bin_count", 10)), smoothing=float(options.get("smoothing", 1.0))
        ).fit(rows)
        probabilities = getattr(calibrator, "_probabilities", None)
        if probabilities is None:
            raise RuntimeError("histogram calibrator did not fit probabilities")
        return {"bin_count": calibrator.bin_count, "smoothing": calibrator.smoothing, "probabilities": list(probabilities)}
    if kind == "evaluation_conformal_nonconformity":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("scores"), list):
            raise ValueError("conformal input must contain scores[]")
        return asdict(fit_nonconformity_threshold(payload["scores"], alpha=float(options.get("alpha", 0.1))))
    if kind == "evaluation_conformal_retrieval":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("relevant_scores"), list):
            raise ValueError("retrieval conformal input must contain relevant_scores[]")
        return asdict(fit_retrieval_calibration(payload["relevant_scores"], alpha=float(options.get("alpha", 0.1))))
    if kind == "retrieval_split_conformal":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("nonconformity_scores"), list):
            raise ValueError("split conformal input must contain nonconformity_scores[]")
        required = (
            "calibration_id", "dataset_manifest_digest", "split_digest", "retrieval_stack_digest",
            "scoring_contract_digest", "domain_id",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"split conformal input missing fields: {missing}")
        fitted = fit_split_conformal_threshold(
            payload["nonconformity_scores"],
            calibration_id=payload["calibration_id"],
            dataset_manifest_digest=payload["dataset_manifest_digest"],
            split_digest=payload["split_digest"],
            retrieval_stack_digest=payload["retrieval_stack_digest"],
            scoring_contract_digest=payload["scoring_contract_digest"],
            domain_id=payload["domain_id"],
            alpha=float(options.get("alpha", 0.1)),
        )
        return {"manifest": asdict(fitted.manifest), "nonconformity_threshold": fitted.nonconformity_threshold, "finite_sample_rank": fitted.finite_sample_rank}
    if kind == "threshold_decision":
        if not isinstance(payload, Mapping) or not isinstance(payload.get("confidences"), list) or not isinstance(payload.get("labels"), list):
            raise ValueError("threshold input must contain confidences[] and labels[]")
        return asdict(
            optimize_threshold(
                payload["confidences"], payload["labels"],
                false_positive_cost=float(options.get("false_positive_cost", 1.0)),
                false_negative_cost=float(options.get("false_negative_cost", 1.0)),
                abstain_cost=float(options.get("abstain_cost", 0.0)),
            )
        )
    if kind == "confidence_isotonic":
        if not isinstance(payload, list) or not payload:
            raise ValueError("confidence isotonic input must be a non-empty JSON list")
        rows = tuple(ConfidenceExample(confidence=row["confidence"], correct=row["correct"]) for row in payload if isinstance(row, Mapping))
        return asdict(fit_confidence_isotonic(rows))
    if kind == "cross_profile_isotonic":
        if not isinstance(payload, Mapping):
            raise ValueError("cross-profile isotonic input must be a JSON object")
        profile_raw = payload.get("profile")
        contract_raw = payload.get("contract")
        examples_raw = payload.get("examples")
        if not isinstance(profile_raw, Mapping) or not isinstance(contract_raw, Mapping) or not isinstance(examples_raw, list):
            raise ValueError("cross-profile input requires profile, contract, and examples")
        profile = RetrieverScoreProfile(
            profile_id=profile_raw["profile_id"],
            family=profile_raw["family"],
            scoring_contract_sha256=profile_raw["scoring_contract_sha256"],
            model_profile_sha256=profile_raw["model_profile_sha256"],
            score_direction=ScoreDirection(profile_raw.get("score_direction", "higher_is_better")),
        )
        contract = CalibrationContract(**dict(contract_raw))
        examples = tuple(
            ScoreCalibrationExample(raw_score=row["raw_score"], relevant=row["relevant"], weight=row.get("weight", 1.0))
            for row in examples_raw if isinstance(row, Mapping)
        )
        artifact = fit_cross_profile_isotonic(profile=profile, contract=contract, examples=examples)
        return asdict(artifact)
    raise ValueError(f"unsupported calibration kind {kind!r}")


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    if selected.is_symlink():
        raise ValueError("config path must not be a symlink")
    config = _load_json(selected)
    if not isinstance(config, Mapping) or config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    kind = str(config.get("kind") or "").strip()
    if kind not in _KINDS:
        raise ValueError(f"unsupported calibration kind {kind!r}")
    source_revision = _source_revision(config.get("source_revision", "auto"))
    input_path = _resolve_path(selected, config.get("input"), "input", must_exist=True)
    output_path = _resolve_path(selected, config.get("output"), "output", must_exist=False)
    input_sha = _sha_file(input_path)
    _expected_digest(config, input_sha)
    payload = _load_json(input_path)
    result = _run(kind, payload, config)
    manifest: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "kind": kind,
        "source_revision": source_revision,
        "config_sha256": _sha_file(selected),
        "input_path": input_path.as_posix(),
        "input_sha256": input_sha,
        "result": result,
    }
    manifest["result_sha256"] = _sha_bytes(_canonical(manifest))
    _atomic_json(output_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="materialize one deterministic calibration artifact")
    fit.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fit":
        print(json.dumps(run_config(args.config), sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError(f"unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_SCHEMA", "SCHEMA", "main", "run_config"]
