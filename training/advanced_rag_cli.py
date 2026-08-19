"""Local-only command line entry point for complete advanced RAG workflows.

Source-level lifecycle::

    validate -> train/resume -> verify-checkpoint -> evaluation-receipt -> export
             -> verify-artifact -> qualify

No command has network fallback. Model execution occurs only for explicit ``train`` or
``load-artifact`` commands; evaluation runs themselves remain external/explicit and are
materialized here as governed receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from evaluation.advanced_rag_receipts import (
    AdvancedEvaluationReceipt,
    AdvancedEvaluationRun,
    build_advanced_evaluation_receipt,
    read_advanced_evaluation_receipt,
    write_advanced_evaluation_receipt,
)
from training.advanced_rag_artifacts import MetricQualificationPolicy, export_dynamic_policy_artifact, export_grounded_generator_artifact
from training.advanced_rag_authoritative_runner import AuthoritativeDynamicRagPolicyTrainingRunner, AuthoritativeGroundedGeneratorTrainingRunner
from training.advanced_rag_config import DynamicConfiguredRun, GroundedConfiguredRun, load_advanced_run_config
from training.advanced_rag_data import ManifestBoundAdvancedJsonlDataset
from training.advanced_rag_promotion_evidence import build_advanced_promotion_evidence, write_advanced_promotion_evidence
from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding, verify_checkpoint_against_run_config
from training.advanced_rag_runtime_loading import load_dynamic_policy_artifact, load_grounded_artifact, read_advanced_artifact_manifest
from training.checkpointing import CheckpointManager
from training.grounded_supervision_pipeline import CachedDocumentUtilityRetrieverBatchBuilder, PairwiseCandidateRetriever
from training.local_artifact_loading import (
    assert_dynamic_generator_binding,
    assert_grounded_artifact_bindings,
    load_local_language_model,
    load_local_sequence_classifier,
    load_local_tokenizer,
    local_tree_sha256,
)

_MAX_JSON_BYTES = 64 * 1024 * 1024


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str))


def _read_json(path: str | Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0 or source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser()
    if destination.exists() and destination.is_symlink():
        raise ValueError("output path may not be a symlink")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_hidden_size(model: Any) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("loaded model does not expose a configuration")
    candidates = []
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            candidates.append(value)
    decoder = getattr(config, "decoder", None)
    if decoder is not None:
        for name in ("hidden_size", "d_model", "n_embd"):
            value = getattr(decoder, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                candidates.append(value)
    if not candidates:
        raise ValueError("could not determine model hidden width from local configuration")
    if len(set(candidates)) != 1:
        raise ValueError(f"model configuration exposes inconsistent hidden widths: {sorted(set(candidates))}")
    return candidates[0]


def _verify_cache(spec: Any | None, *, dataset_manifest_sha256: str, tokenizer_sha256: str, source_commit: str, producer_sha256: str | None = None, label: str) -> Mapping[str, Any] | None:
    if spec is None:
        return None
    identity = spec.identity
    if identity.dataset_manifest_sha256 != dataset_manifest_sha256:
        raise ValueError(f"{label} dataset-manifest identity differs from the training plan")
    if identity.tokenizer_sha256 != tokenizer_sha256:
        raise ValueError(f"{label} tokenizer identity differs from the configured tokenizer")
    if identity.source_commit != source_commit:
        raise ValueError(f"{label} source commit differs from the training plan")
    if producer_sha256 is not None and identity.producer_sha256 != producer_sha256:
        raise ValueError(f"{label} producer identity differs from the training plan")
    return {"root": spec.root, "identity_sha256": identity.digest, "cache_kind": identity.cache_kind}


def _validate_grounded(config: GroundedConfiguredRun, *, load_models: bool) -> Mapping[str, Any]:
    assert_grounded_artifact_bindings(base_model=config.base_model, tokenizer=config.tokenizer, base_model_sha256=config.plan.base_model_sha256, tokenizer_sha256=config.plan.tokenizer_sha256)
    train = ManifestBoundAdvancedJsonlDataset(config.train_split.path, expected_sha256=config.train_split.content_sha256, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, split_name=config.train_split.split_name, record_kind="grounded_generation", expected_record_count=config.train_split.expected_record_count)
    validation = ManifestBoundAdvancedJsonlDataset(config.validation_split.path, expected_sha256=config.validation_split.content_sha256, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, split_name=config.validation_split.split_name, record_kind="grounded_generation", expected_record_count=config.validation_split.expected_record_count)
    cache_summary = {
        "teacher": _verify_cache(config.teacher_cache, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, tokenizer_sha256=config.plan.tokenizer_sha256, source_commit=config.plan.source_commit, producer_sha256=config.plan.teacher_model_sha256, label="teacher cache") if config.teacher_cache is not None else None,
        "reference": _verify_cache(config.reference_cache, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, tokenizer_sha256=config.plan.tokenizer_sha256, source_commit=config.plan.source_commit, label="reference cache") if config.reference_cache is not None else None,
        "retriever_utility": _verify_cache(config.retriever_utility_cache, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, tokenizer_sha256=config.plan.tokenizer_sha256, source_commit=config.plan.source_commit, label="retriever utility cache") if config.retriever_utility_cache is not None else None,
    }
    if config.retriever_model is not None:
        config.retriever_model.verify()
        if config.plan.retriever_stack_sha256 != config.retriever_model.expected_sha256:
            raise ValueError("retriever local artifact differs from the immutable grounded plan")
    if (config.retriever_model is None) != (config.retriever_utility_cache is None):
        raise ValueError("retriever model and utility cache must be configured together")
    model_width = None
    if load_models:
        base_model = load_local_language_model(config.base_model)
        model_width = _model_hidden_size(base_model)
        if model_width != config.plan.architecture.hidden_size:
            raise ValueError("configured grounded hidden_size differs from local model hidden width")
    return {
        "kind": "grounded_generation", "plan_sha256": config.plan.plan_sha256,
        "generator_family": config.base_model.artifact_kind, "base_model_sha256": config.base_model.expected_sha256,
        "tokenizer_sha256": config.tokenizer.expected_sha256, "training_data_sha256": train.binding.content_sha256,
        "validation_data_sha256": validation.binding.content_sha256, "training_records": len(train), "validation_records": len(validation),
        "architecture_hidden_size": config.plan.architecture.hidden_size, "loaded_model_hidden_size": model_width,
        "caches": cache_summary, "resume_checkpoint_digest": config.resume_checkpoint_digest,
    }


def _validate_dynamic(config: DynamicConfiguredRun, *, load_models: bool) -> Mapping[str, Any]:
    assert_dynamic_generator_binding(generator=config.generator, base_generator_sha256=config.plan.base_generator_sha256)
    config.tokenizer.verify()
    train = ManifestBoundAdvancedJsonlDataset(config.train_split.path, expected_sha256=config.train_split.content_sha256, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, split_name=config.train_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=config.train_split.expected_record_count)
    validation = ManifestBoundAdvancedJsonlDataset(config.validation_split.path, expected_sha256=config.validation_split.content_sha256, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, split_name=config.validation_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=config.validation_split.expected_record_count)
    cache = _verify_cache(config.hidden_state_cache, dataset_manifest_sha256=config.plan.dataset_manifest_sha256, tokenizer_sha256=config.tokenizer.expected_sha256, source_commit=config.plan.source_commit, producer_sha256=config.plan.base_generator_sha256, label="hidden-state cache") if config.hidden_state_cache is not None else None
    model_width = None
    if load_models:
        generator = load_local_language_model(config.generator)
        model_width = _model_hidden_size(generator)
        if model_width != config.plan.architecture.context_hidden_size:
            raise ValueError("dynamic context_hidden_size differs from local generator hidden width")
    return {
        "kind": "dynamic_rag_policy", "plan_sha256": config.plan.plan_sha256,
        "generator_sha256": config.generator.expected_sha256, "tokenizer_sha256": config.tokenizer.expected_sha256,
        "training_data_sha256": train.binding.content_sha256, "validation_data_sha256": validation.binding.content_sha256,
        "training_records": len(train), "validation_records": len(validation), "context_hidden_size": config.plan.architecture.context_hidden_size,
        "loaded_generator_hidden_size": model_width, "hidden_state_cache": cache, "resume_checkpoint_digest": config.resume_checkpoint_digest,
    }


def validate_config(path: str | Path, *, load_models: bool = False) -> Mapping[str, Any]:
    configured = load_advanced_run_config(path)
    if isinstance(configured, GroundedConfiguredRun):
        return _validate_grounded(configured, load_models=load_models)
    if isinstance(configured, DynamicConfiguredRun):
        return _validate_dynamic(configured, load_models=load_models)
    raise TypeError("unsupported configured run type")


def _train_grounded(config: GroundedConfiguredRun) -> Mapping[str, Any]:
    _validate_grounded(config, load_models=False)
    tokenizer = load_local_tokenizer(config.tokenizer)
    base_model = load_local_language_model(config.base_model)
    if _model_hidden_size(base_model) != config.plan.architecture.hidden_size:
        raise ValueError("local grounded model hidden width differs from plan")
    teacher_cache = config.teacher_cache.build() if config.teacher_cache is not None else None
    reference_cache = config.reference_cache.build() if config.reference_cache is not None else None
    retriever_model = retriever_builder = None
    if config.retriever_model is not None:
        pair_model = load_local_sequence_classifier(config.retriever_model)
        retriever_model = PairwiseCandidateRetriever(pair_model, positive_label_index=config.retriever_coupling.positive_label_index)
        if config.retriever_utility_cache is None:
            raise ValueError("retriever utility cache is required when retriever model is configured")
        retriever_builder = CachedDocumentUtilityRetrieverBatchBuilder(tokenizer, config.retriever_utility_cache.build(), tokenizer_sha256=config.plan.tokenizer_sha256, config=config.retriever_coupling)
    runner = AuthoritativeGroundedGeneratorTrainingRunner(
        plan=config.plan, base_model=base_model, tokenizer=tokenizer, generator_family=config.base_model.artifact_kind,
        train_split=config.train_split, validation_split=config.validation_split, checkpoint_root=config.checkpoint_root,
        execution=config.execution, collator_config=config.collator, retriever_model=retriever_model,
        teacher_cache=teacher_cache, reference_cache=reference_cache, retriever_batch_builder=retriever_builder,
        trainability=config.trainability,
    )
    result = runner.run(resume_checkpoint_digest=config.resume_checkpoint_digest)
    return {
        "kind": "grounded_generation", "plan_sha256": result.plan_sha256, "training_input_sha256": result.training_input_sha256,
        "training_data_sha256": result.training_data_sha256, "validation_data_sha256": result.validation_data_sha256,
        "checkpoint_root": result.checkpoint_root, "summary": asdict(result.summary),
    }


def _train_dynamic(config: DynamicConfiguredRun) -> Mapping[str, Any]:
    _validate_dynamic(config, load_models=False)
    tokenizer = load_local_tokenizer(config.tokenizer)
    hidden_cache = config.hidden_state_cache.build() if config.hidden_state_cache is not None else None
    runner = AuthoritativeDynamicRagPolicyTrainingRunner(
        plan=config.plan, tokenizer=tokenizer, tokenizer_sha256=config.tokenizer.expected_sha256,
        train_split=config.train_split, validation_split=config.validation_split, checkpoint_root=config.checkpoint_root,
        execution=config.execution, collator_config=config.collator, hidden_state_cache=hidden_cache, trainability=config.trainability,
    )
    result = runner.run(resume_checkpoint_digest=config.resume_checkpoint_digest)
    return {
        "kind": "dynamic_rag_policy", "plan_sha256": result.plan_sha256, "training_input_sha256": result.training_input_sha256,
        "training_data_sha256": result.training_data_sha256, "validation_data_sha256": result.validation_data_sha256,
        "checkpoint_root": result.checkpoint_root, "summary": asdict(result.summary),
    }


def train_from_config(path: str | Path) -> Mapping[str, Any]:
    configured = load_advanced_run_config(path)
    if isinstance(configured, GroundedConfiguredRun):
        return _train_grounded(configured)
    if isinstance(configured, DynamicConfiguredRun):
        return _train_dynamic(configured)
    raise TypeError("unsupported configured run type")


def verify_checkpoint_from_config(path: str | Path, *, checkpoint_digest: str) -> VerifiedAdvancedCheckpointBinding:
    configured = load_advanced_run_config(path)
    if isinstance(configured, GroundedConfiguredRun):
        _validate_grounded(configured, load_models=False)
    else:
        _validate_dynamic(configured, load_models=False)
    return verify_checkpoint_against_run_config(CheckpointManager(configured.checkpoint_root), checkpoint_digest, configured)


def _assert_evaluation_binding(receipt: AdvancedEvaluationReceipt, binding: VerifiedAdvancedCheckpointBinding) -> None:
    comparisons = {
        "kind": receipt.kind == binding.kind,
        "checkpoint_digest": receipt.checkpoint_digest == binding.checkpoint_digest,
        "plan_sha256": receipt.plan_sha256 == binding.plan_sha256,
        "training_input_sha256": receipt.training_input_sha256 == binding.training_input_sha256,
        "training_config_sha256": receipt.training_config_sha256 == binding.training_config_sha256,
        "source_commit": receipt.source_commit == binding.source_commit,
    }
    failures = [key for key, ok in comparisons.items() if not ok]
    if failures:
        raise ValueError(f"evaluation receipt differs from verified checkpoint binding: {','.join(failures)}")


def build_evaluation_receipt_from_config(
    config_path: str | Path, *, checkpoint_digest: str, runs_path: str | Path,
    output_path: str | Path, aggregation: str = "mean",
) -> AdvancedEvaluationReceipt:
    binding = verify_checkpoint_from_config(config_path, checkpoint_digest=checkpoint_digest)
    raw = _read_json(runs_path, label="evaluation runs file")
    if set(raw) != {"schema", "runs"} or raw.get("schema") != "rigorousrag-advanced-evaluation-runs/v1" or not isinstance(raw.get("runs"), list):
        raise ValueError("evaluation runs file must be rigorousrag-advanced-evaluation-runs/v1")
    allowed = {field.name for field in fields(AdvancedEvaluationRun)}
    runs = []
    for index, item in enumerate(raw["runs"]):
        if not isinstance(item, Mapping) or set(item) != allowed:
            raise ValueError(f"evaluation run {index} fields differ from AdvancedEvaluationRun schema")
        runs.append(AdvancedEvaluationRun(**dict(item)))
    receipt = build_advanced_evaluation_receipt(binding, runs, aggregation=aggregation)
    write_advanced_evaluation_receipt(output_path, receipt)
    return receipt


def export_from_config(
    path: str | Path, *, checkpoint_digest: str, destination: str | Path,
    evaluation_receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    configured = load_advanced_run_config(path)
    binding = verify_checkpoint_from_config(path, checkpoint_digest=checkpoint_digest)
    evaluation_sha = None
    if evaluation_receipt_path is not None:
        evaluation = read_advanced_evaluation_receipt(evaluation_receipt_path)
        _assert_evaluation_binding(evaluation, binding)
        evaluation_sha = evaluation.receipt_sha256
    manager = CheckpointManager(configured.checkpoint_root)
    if isinstance(configured, GroundedConfiguredRun):
        manifest = export_grounded_generator_artifact(
            checkpoint_manager=manager, checkpoint_digest=checkpoint_digest, plan=configured.plan,
            verified_binding=binding, destination_root=destination, evaluation_receipt_sha256=evaluation_sha,
            include_retriever=configured.retriever_model is not None,
        )
    elif isinstance(configured, DynamicConfiguredRun):
        manifest = export_dynamic_policy_artifact(
            checkpoint_manager=manager, checkpoint_digest=checkpoint_digest, plan=configured.plan,
            verified_binding=binding, destination_root=destination, evaluation_receipt_sha256=evaluation_sha,
        )
    else:
        raise TypeError("unsupported configured run type")
    return asdict(manifest)


def verify_artifact(directory: str | Path) -> Mapping[str, Any]:
    return asdict(read_advanced_artifact_manifest(directory))


def load_artifact_from_config(config_path: str | Path, artifact_directory: str | Path) -> Mapping[str, Any]:
    configured = load_advanced_run_config(config_path)
    manifest = read_advanced_artifact_manifest(artifact_directory)
    if isinstance(configured, GroundedConfiguredRun):
        loaded = load_grounded_artifact(
            artifact_directory, base_model=configured.base_model, tokenizer=configured.tokenizer,
            retriever_model=configured.retriever_model,
        )
        return {"kind": loaded.manifest.kind, "artifact_sha256": loaded.manifest.artifact_sha256, "generator_family": loaded.manifest.generator_family}
    if manifest.kind != "dynamic_rag_policy":
        raise ValueError("dynamic config cannot load a grounded artifact")
    loaded = load_dynamic_policy_artifact(artifact_directory)
    return {"kind": loaded.manifest.kind, "artifact_sha256": loaded.manifest.artifact_sha256, "budget_sha256": loaded.manifest.budget_sha256}


def _promotion_policy(path: str | Path) -> MetricQualificationPolicy:
    raw = _read_json(path, label="promotion policy")
    if set(raw) != {"schema", "minimum", "maximum"} or raw.get("schema") != "rigorousrag-advanced-promotion-policy-config/v1":
        raise ValueError("promotion policy must be rigorousrag-advanced-promotion-policy-config/v1")
    if not isinstance(raw["minimum"], Mapping) or not isinstance(raw["maximum"], Mapping):
        raise ValueError("promotion policy minimum/maximum must be objects")
    return MetricQualificationPolicy(minimum=raw["minimum"], maximum=raw["maximum"])


def qualify_artifact(
    artifact_directory: str | Path, *, evaluation_receipt_path: str | Path,
    policy_path: str | Path, output_path: str | Path | None = None,
) -> Mapping[str, Any]:
    manifest = read_advanced_artifact_manifest(artifact_directory)
    evaluation = read_advanced_evaluation_receipt(evaluation_receipt_path)
    policy = _promotion_policy(policy_path)
    evidence = build_advanced_promotion_evidence(manifest, evaluation, policy)
    if output_path is not None:
        write_advanced_promotion_evidence(output_path, evidence)
    return asdict(evidence)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rigorousrag-advanced-training", description="Local-only advanced RAG training and artifact orchestration")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate exact local artifacts, data and identities without training")
    validate.add_argument("--config", required=True); validate.add_argument("--load-model-config", action="store_true")
    train = sub.add_parser("train", help="launch or resume the configured training run"); train.add_argument("--config", required=True)
    verify_checkpoint = sub.add_parser("verify-checkpoint", help="prove a checkpoint matches the exact configured training identity")
    verify_checkpoint.add_argument("--config", required=True); verify_checkpoint.add_argument("--checkpoint-digest", required=True)
    receipt = sub.add_parser("evaluation-receipt", help="materialize governed repeated evaluation results into a content-addressed receipt")
    receipt.add_argument("--config", required=True); receipt.add_argument("--checkpoint-digest", required=True); receipt.add_argument("--runs", required=True); receipt.add_argument("--output", required=True); receipt.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    export = sub.add_parser("export", help="export a verified inference artifact")
    export.add_argument("--config", required=True); export.add_argument("--checkpoint-digest", required=True); export.add_argument("--destination", required=True); export.add_argument("--evaluation-receipt")
    verify_art = sub.add_parser("verify-artifact", help="verify artifact directory, manifest and safetensor bytes"); verify_art.add_argument("--artifact", required=True)
    load_art = sub.add_parser("load-artifact", help="strictly reconstruct an artifact using local bindings from config")
    load_art.add_argument("--config", required=True); load_art.add_argument("--artifact", required=True)
    qualify = sub.add_parser("qualify", help="qualify an artifact from a bound evaluation receipt and metric policy")
    qualify.add_argument("--artifact", required=True); qualify.add_argument("--evaluation-receipt", required=True); qualify.add_argument("--policy", required=True); qualify.add_argument("--output")
    tree = sub.add_parser("hash-tree", help="compute the RigorousRAG local artifact tree SHA-256"); tree.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        _print_json(validate_config(args.config, load_models=bool(args.load_model_config)))
    elif args.command == "train":
        _print_json(train_from_config(args.config))
    elif args.command == "verify-checkpoint":
        _print_json(asdict(verify_checkpoint_from_config(args.config, checkpoint_digest=args.checkpoint_digest)))
    elif args.command == "evaluation-receipt":
        _print_json(asdict(build_evaluation_receipt_from_config(args.config, checkpoint_digest=args.checkpoint_digest, runs_path=args.runs, output_path=args.output, aggregation=args.aggregation)))
    elif args.command == "export":
        _print_json(export_from_config(args.config, checkpoint_digest=args.checkpoint_digest, destination=args.destination, evaluation_receipt_path=args.evaluation_receipt))
    elif args.command == "verify-artifact":
        _print_json(verify_artifact(args.artifact))
    elif args.command == "load-artifact":
        _print_json(load_artifact_from_config(args.config, args.artifact))
    elif args.command == "qualify":
        _print_json(qualify_artifact(args.artifact, evaluation_receipt_path=args.evaluation_receipt, policy_path=args.policy, output_path=args.output))
    elif args.command == "hash-tree":
        _print_json({"path": str(Path(args.path).expanduser().resolve()), "tree_sha256": local_tree_sha256(args.path)})
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_evaluation_receipt_from_config", "export_from_config", "load_artifact_from_config", "main",
    "qualify_artifact", "train_from_config", "validate_config", "verify_artifact", "verify_checkpoint_from_config",
]
