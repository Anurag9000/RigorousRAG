"""Authoritative exact-resumable training CLI for RigorousRAG pure-Python learners.

This entrypoint closes the execution gap for repository-owned learners that do not
use a framework optimizer and therefore are easy for framework-oriented source
scanners to miss:

* calibrated cross-profile fusion weights;
* query-grouped ListNet cross-profile fusion;
* learned query-domain classification; and
* learned query-plan pairwise ranking.

No training runs on import.  Every run is bound to immutable input-file digests and
configuration. Fusion/ListNet state is advanced at most one mini-batch between
content-addressed state commits. Domain/plan fitting reuses ``ResumeStateStore``
from :mod:`training.query_plan_resume`, which persists exact epoch/batch cursor,
permutation, RNG state, current/best parameters, validation state and early-stop
counters. Final artifacts are written atomically together with a run manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.learned_query_planning import FeatureVector
from training.cross_profile_fusion_fitting import (
    FusionWeightExample,
    FusionWeightTrainingConfig,
    FusionWeightTrainingSpec,
    FusionWeightTrainingState,
    LearnedFusionWeightArtifact,
    advance_training,
    initialize_training_state,
)
from training.cross_profile_listwise_fusion import (
    FusionRankingCandidate,
    FusionRankingQuery,
    LearnedListwiseFusionArtifact,
    ListwiseFusionTrainingConfig,
    ListwiseFusionTrainingSpec,
    ListwiseFusionTrainingState,
    advance_listwise_training,
    initialize_listwise_training,
)
from training.query_plan_fitting import (
    DomainFitExample,
    FittingConfig,
    PlanPreferenceExample,
)
from training.query_plan_resume import (
    ResumeStateStore,
    fit_domain_classifier_resumable,
    fit_plan_ranker_resumable,
)

SCHEMA = "rigorousrag-authoritative-classical-training/v1"
_ALLOWED_KINDS = {
    "fusion_weight",
    "listwise_fusion",
    "domain_classifier",
    "plan_ranker",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _path(root: Path, value: Any, label: str, *, must_exist: bool) -> Path:
    selected = Path(_identifier(value, label, 4000)).expanduser()
    path = selected if selected.is_absolute() else root / selected
    resolved = path.resolve()
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist as a file: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no training rows")
    return tuple(rows)


class ContentAddressedStateStore:
    """Atomic content-addressed exact-state store with a digest-verified latest pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("state root may not be a symlink")

    def _pointer(self, name: str) -> Path:
        return self.root / f"{_identifier(name, 'state name', 200)}-latest.json"

    def exists(self, name: str) -> bool:
        return self._pointer(name).is_file()

    def save(self, name: str, payload: Mapping[str, Any]) -> str:
        selected = _identifier(name, "state name", 200)
        encoded = _canonical(payload) + b"\n"
        digest = hashlib.sha256(encoded).hexdigest()
        destination = self.root / f"{selected}-{digest}.json"
        if not destination.exists():
            descriptor, temporary = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.root)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        _atomic_json(self._pointer(selected), {"digest": digest, "filename": destination.name})
        return digest

    def load_latest(self, name: str) -> Mapping[str, Any]:
        pointer = _read_json(self._pointer(name))
        digest = _identifier(pointer.get("digest"), "state digest", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("latest state pointer has an invalid digest")
        filename = _identifier(pointer.get("filename"), "state filename", 500)
        if "/" in filename or "\\" in filename:
            raise ValueError("state filename must be a basename")
        source = (self.root / filename).resolve(strict=True)
        if source.parent != self.root or source.is_symlink():
            raise ValueError("state pointer escaped configured state root")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError("state payload digest mismatch")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("state payload must be a JSON object")
        return value


def _common(config_path: Path) -> tuple[Path, Mapping[str, Any], str, Path, Path, Path]:
    config = _read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA!r}")
    kind = _identifier(config.get("kind"), "kind", 100)
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"unsupported classical training kind {kind!r}")
    root = config_path.parent.resolve()
    train_path = _path(root, config.get("train_data"), "train_data", must_exist=True)
    validation_path = _path(root, config.get("validation_data"), "validation_data", must_exist=True)
    output_dir = _path(root, config.get("output_dir"), "output_dir", must_exist=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    return root, config, kind, train_path, validation_path, output_dir


def _fusion_examples(path: Path) -> tuple[FusionWeightExample, ...]:
    result = []
    for row in _read_jsonl(path):
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError("fusion row probabilities must be an object")
        relevant = row.get("relevant")
        if not isinstance(relevant, bool):
            raise ValueError("fusion row relevant must be boolean")
        result.append(FusionWeightExample(probabilities, relevant, float(row.get("weight", 1.0))))
    return tuple(result)


def _fusion_state(payload: Mapping[str, Any]) -> FusionWeightTrainingState:
    return FusionWeightTrainingState(
        spec_sha256=payload["spec_sha256"],
        train_examples_sha256=payload["train_examples_sha256"],
        validation_examples_sha256=payload["validation_examples_sha256"],
        epoch=int(payload["epoch"]),
        batch_index=int(payload["batch_index"]),
        theta=tuple(float(value) for value in payload["theta"]),
        best_theta=tuple(float(value) for value in payload["best_theta"]),
        best_validation_loss=None if payload.get("best_validation_loss") is None else float(payload["best_validation_loss"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        stale_epochs=int(payload["stale_epochs"]),
        completed=bool(payload["completed"]),
    )


def _run_fusion(config: Mapping[str, Any], train_path: Path, validation_path: Path, output_dir: Path) -> Mapping[str, Any]:
    train = _fusion_examples(train_path)
    validation = _fusion_examples(validation_path)
    profiles = tuple(str(value) for value in config.get("profile_ids", ()))
    calibration = config.get("calibration_artifact_sha256s")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration_artifact_sha256s must be a profile->SHA256 object")
    training_config = FusionWeightTrainingConfig(**dict(config.get("training", {})))
    spec = FusionWeightTrainingSpec(
        profile_ids=profiles,
        calibration_contract_sha256=config["calibration_contract_sha256"],
        calibration_artifact_sha256s=tuple((str(key), str(value)) for key, value in calibration.items()),
        train_split_sha256=_sha_file(train_path),
        validation_split_sha256=_sha_file(validation_path),
        source_revision=config["source_revision"],
        config=training_config,
    )
    store = ContentAddressedStateStore(output_dir / "state")
    name = "fusion-weight"
    if store.exists(name):
        state = _fusion_state(store.load_latest(name))
        if state.spec_sha256 != spec.spec_sha256:
            raise ValueError("existing fusion state belongs to a different spec/config/data identity")
    else:
        state = initialize_training_state(spec, train, validation)
        store.save(name, asdict(state))
    while not state.completed:
        state = advance_training(spec, state, train, validation, max_batches=1)
        store.save(name, asdict(state))
    artifact = LearnedFusionWeightArtifact.build(spec=spec, state=state)
    return {
        "kind": "fusion_weight",
        "spec_sha256": spec.spec_sha256,
        "state_sha256": state.state_sha256,
        "artifact": asdict(artifact),
    }


def _ranking_queries(path: Path) -> tuple[FusionRankingQuery, ...]:
    queries = []
    for row in _read_jsonl(path):
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
            raise ValueError("listwise row candidates must be an array")
        candidates = []
        for value in raw_candidates:
            if not isinstance(value, Mapping) or not isinstance(value.get("probabilities"), Mapping):
                raise ValueError("listwise candidate must contain a probability object")
            candidates.append(
                FusionRankingCandidate(
                    candidate_id=value["candidate_id"],
                    probabilities=value["probabilities"],
                    relevance_grade=float(value["relevance_grade"]),
                )
            )
        queries.append(
            FusionRankingQuery(
                query_sha256=row["query_sha256"],
                candidates=tuple(candidates),
                weight=float(row.get("weight", 1.0)),
            )
        )
    return tuple(queries)


def _listwise_state(payload: Mapping[str, Any]) -> ListwiseFusionTrainingState:
    return ListwiseFusionTrainingState(
        spec_sha256=str(payload["spec_sha256"]),
        train_queries_sha256=str(payload["train_queries_sha256"]),
        validation_queries_sha256=str(payload["validation_queries_sha256"]),
        epoch=int(payload["epoch"]),
        batch_index=int(payload["batch_index"]),
        theta=tuple(float(value) for value in payload["theta"]),
        best_theta=tuple(float(value) for value in payload["best_theta"]),
        best_validation_loss=None if payload.get("best_validation_loss") is None else float(payload["best_validation_loss"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        stale_epochs=int(payload["stale_epochs"]),
        completed=bool(payload["completed"]),
    )


def _run_listwise(config: Mapping[str, Any], train_path: Path, validation_path: Path, output_dir: Path) -> Mapping[str, Any]:
    train = _ranking_queries(train_path)
    validation = _ranking_queries(validation_path)
    profiles = tuple(str(value) for value in config.get("profile_ids", ()))
    calibration = config.get("calibration_artifact_sha256s")
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration_artifact_sha256s must be a profile->SHA256 object")
    training_config = ListwiseFusionTrainingConfig(**dict(config.get("training", {})))
    spec = ListwiseFusionTrainingSpec(
        profile_ids=profiles,
        calibration_contract_sha256=config["calibration_contract_sha256"],
        calibration_artifact_sha256s=tuple((str(key), str(value)) for key, value in calibration.items()),
        train_split_sha256=_sha_file(train_path),
        validation_split_sha256=_sha_file(validation_path),
        source_revision=config["source_revision"],
        config=training_config,
    )
    store = ContentAddressedStateStore(output_dir / "state")
    name = "listwise-fusion"
    if store.exists(name):
        state = _listwise_state(store.load_latest(name))
        if state.spec_sha256 != spec.spec_sha256:
            raise ValueError("existing listwise state belongs to a different spec/config/data identity")
    else:
        state = initialize_listwise_training(spec, train, validation)
        store.save(name, asdict(state))
    while not state.completed:
        state = advance_listwise_training(spec, state, train, validation, max_batches=1)
        store.save(name, asdict(state))
    artifact = LearnedListwiseFusionArtifact.build(spec, state)
    return {
        "kind": "listwise_fusion",
        "spec_sha256": spec.spec_sha256,
        "state_sha256": state.state_sha256,
        "artifact": asdict(artifact),
    }


def _feature_vector(value: Any, label: str) -> FeatureVector:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    schema = value.get("schema")
    values = value.get("values")
    if not isinstance(schema, Sequence) or isinstance(schema, (str, bytes, bytearray)):
        raise ValueError(f"{label}.schema must be an array")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label}.values must be an array")
    return FeatureVector(tuple(str(item) for item in schema), tuple(float(item) for item in values))


def _domain_examples(path: Path) -> tuple[DomainFitExample, ...]:
    return tuple(
        DomainFitExample(_feature_vector(row.get("features"), "features"), row["label"])
        for row in _read_jsonl(path)
    )


def _plan_examples(path: Path) -> tuple[PlanPreferenceExample, ...]:
    return tuple(
        PlanPreferenceExample(
            _feature_vector(row.get("preferred_features"), "preferred_features"),
            _feature_vector(row.get("rejected_features"), "rejected_features"),
            float(row.get("weight", 1.0)),
        )
        for row in _read_jsonl(path)
    )


def _run_domain(config: Mapping[str, Any], train_path: Path, validation_path: Path, output_dir: Path) -> Mapping[str, Any]:
    training = _domain_examples(train_path)
    validation = _domain_examples(validation_path)
    manifest = _digest({"train_sha256": _sha_file(train_path), "validation_sha256": _sha_file(validation_path)})
    fitting = FittingConfig(**dict(config.get("training", {})))
    store = ResumeStateStore(output_dir / "state")
    pointer = store.root / "domain-classifier-latest.json"
    artifact, result = fit_domain_classifier_resumable(
        training,
        labels=tuple(str(value) for value in config.get("labels", ())),
        fallback_label=config["fallback_label"],
        training_manifest_digest=manifest,
        validation=validation,
        minimum_confidence=float(config.get("minimum_confidence", 0.55)),
        config=fitting,
        state_store=store,
        resume=pointer.is_file(),
        checkpoint_every_batches=1,
    )
    return {"kind": "domain_classifier", "training_manifest_digest": manifest, "artifact": asdict(artifact), "fit_result": asdict(result)}


def _run_plan(config: Mapping[str, Any], train_path: Path, validation_path: Path, output_dir: Path) -> Mapping[str, Any]:
    training = _plan_examples(train_path)
    validation = _plan_examples(validation_path)
    manifest = _digest({"train_sha256": _sha_file(train_path), "validation_sha256": _sha_file(validation_path)})
    fitting = FittingConfig(**dict(config.get("training", {})))
    store = ResumeStateStore(output_dir / "state")
    pointer = store.root / "plan-ranker-latest.json"
    artifact, result = fit_plan_ranker_resumable(
        training,
        training_manifest_digest=manifest,
        validation=validation,
        config=fitting,
        latency_penalty=float(config.get("latency_penalty", 0.0)),
        cost_penalty=float(config.get("cost_penalty", 0.0)),
        risk_penalty=float(config.get("risk_penalty", 0.0)),
        state_store=store,
        resume=pointer.is_file(),
        checkpoint_every_batches=1,
    )
    return {"kind": "plan_ranker", "training_manifest_digest": manifest, "artifact": asdict(artifact), "fit_result": asdict(result)}


_RUNNERS: Mapping[str, Callable[[Mapping[str, Any], Path, Path, Path], Mapping[str, Any]]] = {
    "fusion_weight": _run_fusion,
    "listwise_fusion": _run_listwise,
    "domain_classifier": _run_domain,
    "plan_ranker": _run_plan,
}


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    _, config, kind, train_path, validation_path, output_dir = _common(selected)
    result = _RUNNERS[kind](config, train_path, validation_path, output_dir)
    manifest = {
        "schema": "rigorousrag-authoritative-classical-training-result/v1",
        "config_sha256": _sha_file(selected),
        "train_data_sha256": _sha_file(train_path),
        "validation_data_sha256": _sha_file(validation_path),
        "result": result,
    }
    manifest["result_sha256"] = _digest(manifest)
    _atomic_json(output_dir / "training_result.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run or exactly resume one classical training configuration")
    train.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = run_config(args.config)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError(f"unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ContentAddressedStateStore", "SCHEMA", "main", "run_config"]
