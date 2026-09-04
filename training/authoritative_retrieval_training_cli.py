"""Authoritative local-only learned-retrieval training CLI.

This module turns the repository's existing learned retrieval architectures, governed
JSONL data pipeline, distilled losses and :class:`TorchTrainingEngine` into one explicit
execution authority.  It does **not** implement a second optimizer/checkpoint loop.
Instead it constructs the requested architecture/collator/step and delegates execution
to the existing engine, whose checkpoint captures model, optimizer, scheduler, scaler,
trainer cursor, Python/NumPy/Torch/CUDA RNG, resumable sampler and collator state.

Supported trainable families are dense bi-encoder, SPLADE, uniCOIL, ColBERT and
listwise cross-encoder reranking. Dense/SPLADE/uniCOIL/ColBERT use the repository's
masked teacher-distillation steps; setting ``distillation_weight`` to zero recovers the
ordinary contrastive objective without changing the execution path.

All pretrained artifacts are local-only and remote code is disabled. The complete local
model/tokenizer trees are SHA-256 bound into the checkpoint architecture identity so an
exact resume fails if the admitted base artifacts change between invocations. Validation
uses a fresh deterministic collator derived from ``optimizer_step``; evaluation state is
therefore independent of interruption history while training sampler/collator state remains
checkpointed exactly.
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

from training.checkpointing import CheckpointManager
from training.data_pipeline import (
    BiEncoderCollator,
    CollatorConfig,
    CrossEncoderCollator,
    ManifestBoundJsonlDataset,
    ResumableDeterministicSampler,
    sha256_file,
)
from training.distilled_steps import (
    DistillationConfig,
    DistilledColBERTContrastiveStep,
    DistilledDenseContrastiveStep,
    DistilledSparseContrastiveStep,
)
from training.model_architectures import (
    ColBERTConfig,
    ColBERTEncoder,
    CrossEncoderReranker,
    DenseBiEncoder,
    EncoderConfig,
    ListwiseReranker,
    PoolingStrategy,
    SpladeConfig,
    SpladeEncoder,
    UniCOILConfig,
    UniCOILEncoder,
)
from training.torch_engine import (
    ListwiseCrossEncoderStep,
    StageRuntime,
    TorchTrainingEngine,
    TrainerConfig,
    TrainingStageSpec,
    move_to_device,
)
from training.torch_losses import SparsePenaltyWeights

SCHEMA = "rigorousrag-authoritative-retrieval-training/v1"
RESULT_SCHEMA = "rigorousrag-authoritative-retrieval-training-result/v1"
_ARCHITECTURES = {"dense", "splade", "unicoil", "colbert", "cross_encoder"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _path(base: Path, value: Any, label: str, *, directory: bool = False) -> Path:
    selected = Path(_identifier(value, label, 4000)).expanduser()
    path = selected if selected.is_absolute() else base / selected
    resolved = path.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    if resolved.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    return resolved


def _output_path(base: Path, value: Any) -> Path:
    selected = Path(_identifier(value, "output_dir", 4000)).expanduser()
    path = selected if selected.is_absolute() else base / selected
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if resolved.is_symlink():
        raise ValueError("output_dir may not be a symlink")
    return resolved


def _tree_sha256(root: Path) -> str:
    """Digest a local artifact directory including relative names and file bytes."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root must be a regular directory")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {path}")
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_config(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("retrieval training config must contain an object")
    if value.get("schema") != SCHEMA:
        raise ValueError(f"retrieval training config schema must be {SCHEMA!r}")
    return value


def _load_tokenizer(root: Path, revision: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional execution dependency.
        raise RuntimeError("learned retrieval training requires transformers") from exc
    return AutoTokenizer.from_pretrained(
        root,
        revision=revision,
        local_files_only=True,
        trust_remote_code=False,
    )


def _architecture_model(architecture: str, model_root: Path, revision: str | None, config: Mapping[str, Any]) -> Any:
    model_config = dict(config.get("model", {}))
    if architecture == "dense":
        encoder = EncoderConfig(
            pooling=PoolingStrategy(model_config.get("pooling", "mean")),
            projection_dim=model_config.get("projection_dim"),
            normalize=bool(model_config.get("normalize", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
        untied = model_config.get("untied_document_model_root")
        untied_path = None
        if untied is not None:
            base = Path(str(untied)).expanduser()
            untied_path = str(base.resolve(strict=True))
        return DenseBiEncoder.from_local_pretrained(
            str(model_root),
            config=encoder,
            revision=revision,
            untied_document_model_name_or_path=untied_path,
            untied_document_revision=model_config.get("untied_document_revision"),
            local_files_only=True,
            trust_remote_code=False,
        )
    if architecture == "splade":
        return SpladeEncoder.from_local_pretrained(
            str(model_root),
            config=SpladeConfig(
                aggregation=str(model_config.get("aggregation", "max")),
                mask_special_token_ids=tuple(int(value) for value in model_config.get("mask_special_token_ids", ())),
            ),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    if architecture == "unicoil":
        return UniCOILEncoder.from_local_pretrained(
            str(model_root),
            config=UniCOILConfig(
                aggregation=str(model_config.get("aggregation", "max")),
                nonnegative=bool(model_config.get("nonnegative", True)),
                exclude_token_ids=tuple(int(value) for value in model_config.get("exclude_token_ids", ())),
            ),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    if architecture == "colbert":
        return ColBERTEncoder.from_local_pretrained(
            str(model_root),
            config=ColBERTConfig(
                projection_dim=int(model_config.get("projection_dim", 128)),
                dropout=float(model_config.get("dropout", 0.0)),
                normalize=bool(model_config.get("normalize", True)),
                exclude_token_ids=tuple(int(value) for value in model_config.get("exclude_token_ids", ())),
            ),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
    if architecture == "cross_encoder":
        cross = CrossEncoderReranker.from_local_pretrained(
            str(model_root),
            revision=revision,
            score_index=int(model_config.get("score_index", 0)),
            local_files_only=True,
            trust_remote_code=False,
            num_labels=int(model_config.get("num_labels", 1)),
        )
        return ListwiseReranker(cross)
    raise ValueError(f"unsupported retrieval architecture {architecture}")


def _distillation_config(loss: Mapping[str, Any], *, sparse: bool) -> DistillationConfig:
    return DistillationConfig(
        retrieval_temperature=float(loss.get("retrieval_temperature", 1.0 if sparse else 0.05)),
        teacher_temperature=float(loss.get("teacher_temperature", 1.0)),
        distillation_weight=float(loss.get("distillation_weight", 0.0)),
        minimum_teacher_candidates=int(loss.get("minimum_teacher_candidates", 2)),
    )


def _step(architecture: str, config: Mapping[str, Any]) -> Any:
    loss = dict(config.get("loss", {}))
    if architecture == "dense":
        return DistilledDenseContrastiveStep(_distillation_config(loss, sparse=False))
    if architecture in {"splade", "unicoil"}:
        penalties = SparsePenaltyWeights(**dict(loss.get("sparse_penalties", {})))
        return DistilledSparseContrastiveStep(_distillation_config(loss, sparse=True), penalties=penalties)
    if architecture == "colbert":
        return DistilledColBERTContrastiveStep(_distillation_config(loss, sparse=False))
    if architecture == "cross_encoder":
        return ListwiseCrossEncoderStep(temperature=float(loss.get("listwise_temperature", 1.0)))
    raise ValueError(f"unsupported retrieval architecture {architecture}")


def _collator_factory(architecture: str, tokenizer: Any, config: Mapping[str, Any], *, seed: int) -> Callable[[], Any]:
    data = dict(config.get("data", {}))
    if architecture == "cross_encoder":
        def make_cross() -> Any:
            return CrossEncoderCollator(
                tokenizer,
                max_length=int(data.get("pair_max_length", 512)),
                negatives_per_query=int(data.get("negatives_per_query", 8)),
                seed=seed,
            )
        return make_cross

    collator_config = CollatorConfig(
        query_max_length=int(data.get("query_max_length", 64)),
        document_max_length=int(data.get("document_max_length", 512)),
        negatives_per_query=int(data.get("negatives_per_query", 8)),
        positive_selection_seed=seed,
        pad_to_multiple_of=data.get("pad_to_multiple_of", 8),
    )

    def make_bi() -> Any:
        return BiEncoderCollator(tokenizer, collator_config)

    return make_bi


def _dataloader(dataset: Any, sampler: Any, collator: Any, *, batch_size: int) -> Any:
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("learned retrieval training requires PyTorch DataLoader") from exc
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def _validation_evaluator(
    *,
    validation_dataset: Any,
    architecture: str,
    tokenizer: Any,
    config: Mapping[str, Any],
    step: Any,
    device: Any,
    batch_size: int,
    seed: int,
) -> Callable[..., Mapping[str, float]]:
    try:
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("learned retrieval validation requires PyTorch DataLoader") from exc

    def evaluate(model: Any, *, stage_index: int, optimizer_step: int) -> Mapping[str, float]:
        # Recreate the stateful collator on every evaluation from immutable run
        # identity + optimizer step. This removes evaluation-history dependence.
        factory = _collator_factory(
            architecture,
            tokenizer,
            config,
            seed=seed + stage_index * 1_000_003 + optimizer_step,
        )
        loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=factory(),
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        total = 0.0
        count = 0
        for batch in loader:
            batch = move_to_device(batch, device)
            result = step(model, batch)
            value = float(result.loss.detach().cpu())
            if value != value or abs(value) == float("inf"):
                raise RuntimeError("validation produced a non-finite loss")
            total += value
            count += 1
        if count == 0:
            raise RuntimeError("validation dataloader produced no batches")
        return {"validation_loss": total / count}

    return evaluate


def _stage_specs(config: Mapping[str, Any]) -> tuple[TrainingStageSpec, ...]:
    raw = config.get("stages")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise ValueError("retrieval training config requires a non-empty stages array")
    stages = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"stages[{index}] must be an object")
        stage = TrainingStageSpec(**dict(row))
        if stage.evaluate_every_steps is None:
            raise ValueError("every retrieval training stage must evaluate for early stopping")
        stages.append(stage)
    return tuple(stages)


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    config = _load_config(selected)
    base = selected.parent
    architecture = _identifier(config.get("architecture"), "architecture", 100).lower()
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported retrieval architecture {architecture!r}")

    train_path = _path(base, config.get("train_data"), "train_data")
    validation_path = _path(base, config.get("validation_data"), "validation_data")
    model_root = _path(base, config.get("model_root"), "model_root", directory=True)
    tokenizer_root = _path(base, config.get("tokenizer_root", config.get("model_root")), "tokenizer_root", directory=True)
    output_dir = _output_path(base, config.get("output_dir"))

    config_sha256 = sha256_file(selected)
    train_sha256 = sha256_file(train_path)
    validation_sha256 = sha256_file(validation_path)
    model_tree_sha256 = _tree_sha256(model_root)
    tokenizer_tree_sha256 = _tree_sha256(tokenizer_root)
    dataset_manifest_digest = _digest(
        {
            "schema": "rigorousrag-retrieval-dataset-binding/v1",
            "train_sha256": train_sha256,
            "validation_sha256": validation_sha256,
        }
    )
    artifact_identity = f"{architecture}:model={model_tree_sha256}:tokenizer={tokenizer_tree_sha256}"

    result_path = output_dir / "training_result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            isinstance(previous, Mapping)
            and previous.get("schema") == RESULT_SCHEMA
            and previous.get("config_sha256") == config_sha256
            and previous.get("train_data_sha256") == train_sha256
            and previous.get("validation_data_sha256") == validation_sha256
            and previous.get("model_tree_sha256") == model_tree_sha256
            and previous.get("tokenizer_tree_sha256") == tokenizer_tree_sha256
            and previous.get("complete") is True
        ):
            return previous

    tokenizer = _load_tokenizer(tokenizer_root, config.get("tokenizer_revision"))
    model = _architecture_model(architecture, model_root, config.get("model_revision"), config)
    step = _step(architecture, config)

    train_dataset = ManifestBoundJsonlDataset(
        train_path,
        expected_sha256=train_sha256,
        dataset_manifest_digest=dataset_manifest_digest,
        split_name="train",
    )
    validation_dataset = ManifestBoundJsonlDataset(
        validation_path,
        expected_sha256=validation_sha256,
        dataset_manifest_digest=dataset_manifest_digest,
        split_name="validation",
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("retrieval train and validation splits must both be non-empty")

    data = dict(config.get("data", {}))
    batch_size = int(data.get("batch_size", 8))
    validation_batch_size = int(data.get("validation_batch_size", batch_size))
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    seed = int(config.get("seed", 0))
    stages = _stage_specs(config)

    runtimes = []
    for stage_index, _ in enumerate(stages):
        sampler = ResumableDeterministicSampler(
            len(train_dataset),
            seed=seed + stage_index * 1_000_003,
            shuffle=True,
        )
        collator = _collator_factory(
            architecture,
            tokenizer,
            config,
            seed=seed + stage_index * 1_000_003,
        )()
        runtimes.append(
            StageRuntime(
                _dataloader(train_dataset, sampler, collator, batch_size=batch_size),
                step,
                sampler=sampler,
                collator=collator,
            )
        )

    early = dict(config.get("early_stopping", {}))
    trainer_config = TrainerConfig(
        run_id=_identifier(config.get("run_id", f"retrieval-{architecture}"), "run_id", 500),
        source_commit=config["source_commit"],
        dataset_manifest_digest=dataset_manifest_digest,
        model_architecture=artifact_identity,
        stages=stages,
        device=str(config.get("device", "auto")),
        precision=str(config.get("precision", "fp32")),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        max_grad_norm=None if config.get("max_grad_norm", 1.0) is None else float(config.get("max_grad_norm", 1.0)),
        seed=seed,
        deterministic_algorithms=bool(config.get("deterministic_algorithms", False)),
        ddp=bool(config.get("ddp", False)),
        find_unused_parameters=bool(config.get("find_unused_parameters", False)),
        early_stopping_metric="validation_loss",
        early_stopping_mode="min",
        early_stopping_patience=int(early.get("patience", 10)),
        early_stopping_min_delta=float(early.get("min_delta", 0.0)),
    )
    checkpoints = CheckpointManager(output_dir / "checkpoints")
    engine = TorchTrainingEngine(model, trainer_config, checkpoints)
    resume_digest = checkpoints.resolve_pointer("latest") if (checkpoints.root / "latest.json").is_file() else None
    evaluator = _validation_evaluator(
        validation_dataset=validation_dataset,
        architecture=architecture,
        tokenizer=tokenizer,
        config=config,
        step=step,
        device=engine.device,
        batch_size=validation_batch_size,
        seed=seed + 10_000_019,
    )
    summary = engine.fit(
        tuple(runtimes),
        resume_checkpoint_digest=resume_digest,
        evaluator=evaluator,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "architecture": architecture,
        "config_sha256": config_sha256,
        "train_data_sha256": train_sha256,
        "validation_data_sha256": validation_sha256,
        "dataset_manifest_digest": dataset_manifest_digest,
        "model_tree_sha256": model_tree_sha256,
        "tokenizer_tree_sha256": tokenizer_tree_sha256,
        "model_architecture_identity": artifact_identity,
        "trainer_config_digest": trainer_config.digest,
        "summary": asdict(summary),
    }
    result["result_sha256"] = _digest(result)
    _atomic_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run or exactly resume one retrieval recipe")
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


__all__ = ["RESULT_SCHEMA", "SCHEMA", "main", "run_config"]
