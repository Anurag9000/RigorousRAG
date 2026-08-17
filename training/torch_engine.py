"""Executable staged PyTorch training engine for RigorousRAG learned components.

The engine is deliberately library-style: nothing runs on import.  When explicitly
invoked it supports deterministic seeding, CPU/CUDA/MPS selection, fp32/bf16/fp16
mixed precision, gradient accumulation/clipping, AdamW, warmup + linear/cosine
scheduling, optional DDP wrapping, hard-negative refresh hooks, evaluation/early
stopping, stage-boundary checkpoints and exact mid-stage resume.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
except Exception:  # pragma: no cover - optional training dependency.
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from training.checkpointing import (
    CheckpointManager,
    TrainerCursor,
    TrainerState,
    canonical_digest,
)
from training.torch_losses import (
    SparsePenaltyWeights,
    TensorLossBreakdown,
    in_batch_info_nce,
    listnet_loss,
    sparse_retrieval_objective,
)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("training execution requires optional PyTorch dependencies")


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _git_commit(value: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError("source_commit must be a full SHA-1 or SHA-256 Git object id")
    return selected


def _sha256(value: str, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def seed_everything(seed: int, *, deterministic_algorithms: bool = False) -> None:
    _require_torch()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be a non-negative 63-bit integer")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = "auto") -> Any:
    _require_torch()
    selected = requested.strip().lower()
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def move_to_device(value: Any, device: Any) -> Any:
    _require_torch()
    if torch.is_tensor(value):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


@dataclass(frozen=True)
class TrainingStageSpec:
    name: str
    max_optimizer_steps: int
    learning_rate: float
    weight_decay: float = 0.01
    warmup_steps: int = 0
    scheduler: str = "linear"
    checkpoint_every_steps: int = 500
    evaluate_every_steps: int | None = None
    hard_negative_refresh_every_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name is required")
        for name, minimum in (
            ("max_optimizer_steps", 1),
            ("warmup_steps", 0),
            ("checkpoint_every_steps", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.evaluate_every_steps is not None and self.evaluate_every_steps <= 0:
            raise ValueError("evaluate_every_steps must be positive or None")
        if self.hard_negative_refresh_every_steps is not None and self.hard_negative_refresh_every_steps <= 0:
            raise ValueError("hard_negative_refresh_every_steps must be positive or None")
        if _finite(self.learning_rate, "learning_rate") <= 0.0:
            raise ValueError("learning_rate must be positive")
        if _finite(self.weight_decay, "weight_decay") < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")


@dataclass(frozen=True)
class TrainerConfig:
    run_id: str
    source_commit: str
    dataset_manifest_digest: str
    model_architecture: str
    stages: tuple[TrainingStageSpec, ...]
    device: str = "auto"
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    seed: int = 0
    deterministic_algorithms: bool = False
    ddp: bool = False
    find_unused_parameters: bool = False
    early_stopping_metric: str | None = None
    early_stopping_mode: str = "max"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id is required")
        object.__setattr__(self, "source_commit", _git_commit(self.source_commit))
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            _sha256(self.dataset_manifest_digest, "dataset_manifest_digest"),
        )
        if not isinstance(self.model_architecture, str) or not self.model_architecture.strip():
            raise ValueError("model_architecture is required")
        if not self.stages or any(not isinstance(stage, TrainingStageSpec) for stage in self.stages):
            raise ValueError("stages must contain at least one TrainingStageSpec")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if isinstance(self.gradient_accumulation_steps, bool) or not isinstance(self.gradient_accumulation_steps, int) or self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm is not None and _finite(self.max_grad_norm, "max_grad_norm") <= 0.0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be min or max")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if _finite(self.early_stopping_min_delta, "early_stopping_min_delta") < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")

    @property
    def digest(self) -> str:
        from dataclasses import asdict

        return canonical_digest(asdict(self))


@dataclass
class StepResult:
    loss: Any
    metrics: dict[str, float] = field(default_factory=dict)


class BatchStep(Protocol):
    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult: ...


class Evaluator(Protocol):
    def __call__(self, model: Any, *, stage_index: int, optimizer_step: int) -> Mapping[str, float]: ...


class HardNegativeRefreshHook(Protocol):
    def __call__(self, model: Any, *, stage_index: int, optimizer_step: int) -> None: ...


@dataclass
class StageRuntime:
    dataloader: Iterable[Mapping[str, Any]]
    step: BatchStep
    sampler: Any | None = None
    collator: Any | None = None


class DenseContrastiveStep:
    def __init__(self, *, temperature: float = 0.05, label_smoothing: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.label_smoothing = float(label_smoothing)

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        query = model.encode_queries(batch["query_inputs"])
        documents = model.encode_documents(batch["document_inputs"])
        scores = query @ documents.transpose(0, 1)
        mask = batch.get("false_negative_mask")
        if mask is not None:
            scores = scores.masked_fill(mask.to(device=scores.device), torch.finfo(scores.dtype).min)
        loss = in_batch_info_nce(
            scores,
            positive_indices=batch["positive_indices"],
            temperature=self.temperature,
            label_smoothing=self.label_smoothing,
        )
        return StepResult(loss, {"retrieval_loss": float(loss.detach().cpu())})


class SparseContrastiveStep:
    def __init__(
        self,
        *,
        temperature: float = 1.0,
        penalties: SparsePenaltyWeights = SparsePenaltyWeights(),
        distillation_weight: float = 0.0,
        teacher_temperature: float = 1.0,
    ) -> None:
        self.temperature = float(temperature)
        self.penalties = penalties
        self.distillation_weight = float(distillation_weight)
        self.teacher_temperature = float(teacher_temperature)

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        query_weights = model(**batch["query_inputs"])
        document_weights = model(**batch["document_inputs"])
        scores = query_weights @ document_weights.transpose(0, 1)
        mask = batch.get("false_negative_mask")
        if mask is not None:
            scores = scores.masked_fill(mask.to(device=scores.device), torch.finfo(scores.dtype).min)
        teacher = batch.get("teacher_scores")
        student_logits = teacher_logits = None
        if teacher is not None:
            teacher = teacher.to(device=scores.device, dtype=scores.dtype)
            if torch.isfinite(teacher).all():
                student_logits, teacher_logits = scores, teacher
        result = sparse_retrieval_objective(
            scores,
            batch["positive_indices"],
            query_weights,
            document_weights,
            temperature=self.temperature,
            penalties=self.penalties,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            distillation_weight=self.distillation_weight,
            teacher_temperature=self.teacher_temperature,
        )
        metrics = {
            "retrieval_loss": float(result.retrieval.detach().cpu()),
            "query_l1": float(result.query_l1.detach().cpu()),
            "document_l1": float(result.document_l1.detach().cpu()),
            "query_flops": float(result.query_flops.detach().cpu()),
            "document_flops": float(result.document_flops.detach().cpu()),
        }
        if result.distillation is not None:
            metrics["distillation_loss"] = float(result.distillation.detach().cpu())
        return StepResult(result.total, metrics)


class ColBERTContrastiveStep:
    def __init__(self, *, temperature: float = 0.05) -> None:
        self.temperature = float(temperature)

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        query_embeddings, query_mask = model(**batch["query_inputs"])
        document_embeddings, document_mask = model(**batch["document_inputs"])
        scores = model.score_matrix(query_embeddings, query_mask, document_embeddings, document_mask)
        false_negative = batch.get("false_negative_mask")
        if false_negative is not None:
            scores = scores.masked_fill(false_negative.to(device=scores.device), torch.finfo(scores.dtype).min)
        loss = in_batch_info_nce(scores, positive_indices=batch["positive_indices"], temperature=self.temperature)
        return StepResult(loss, {"retrieval_loss": float(loss.detach().cpu())})


class ListwiseCrossEncoderStep:
    def __init__(self, *, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        scores = model.cross_encoder(**batch["pair_inputs"]) if hasattr(model, "cross_encoder") else model(**batch["pair_inputs"])
        relevance = batch["relevance"].to(device=scores.device, dtype=scores.dtype)
        losses: list[Any] = []
        offset = 0
        for size in batch["group_sizes"]:
            group_scores = scores[offset : offset + size].unsqueeze(0)
            group_relevance = relevance[offset : offset + size].unsqueeze(0)
            losses.append(listnet_loss(group_scores, group_relevance, temperature=self.temperature))
            offset += size
        loss = torch.stack(losses).mean()
        return StepResult(loss, {"listwise_loss": float(loss.detach().cpu())})


def build_adamw(model: Any, *, learning_rate: float, weight_decay: float = 0.01) -> Any:
    _require_torch()
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("bias") or "layernorm" in name.casefold() or "layer_norm" in name.casefold():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(learning_rate))


def build_stage_scheduler(optimizer: Any, stage: TrainingStageSpec) -> Any:
    _require_torch()

    def scale(step: int) -> float:
        if stage.warmup_steps > 0 and step < stage.warmup_steps:
            return max(1e-12, (step + 1) / stage.warmup_steps)
        if stage.scheduler == "constant":
            return 1.0
        progress_denominator = max(1, stage.max_optimizer_steps - stage.warmup_steps)
        progress = min(1.0, max(0.0, (step - stage.warmup_steps) / progress_denominator))
        if stage.scheduler == "linear":
            return max(0.0, 1.0 - progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _autocast(device: Any, precision: str) -> Any:
    _require_torch()
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 autocast is supported only on CUDA by this trainer")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device.type not in {"cuda", "cpu"}:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def maybe_wrap_ddp(model: Any, config: TrainerConfig, device: Any) -> Any:
    _require_torch()
    if not config.ddp:
        return model
    if dist is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("ddp=True requires an initialized torch.distributed process group")
    if device.type == "cuda":
        local_rank = int(torch.cuda.current_device())
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.find_unused_parameters,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=config.find_unused_parameters,
    )


@dataclass(frozen=True)
class TrainingSummary:
    stopped_early: bool
    completed_stages: int
    optimizer_steps: int
    global_steps: int
    latest_checkpoint_digest: str | None
    best_metric: float | None
    best_checkpoint_digest: str | None


class TorchTrainingEngine:
    def __init__(
        self,
        model: Any,
        config: TrainerConfig,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        _require_torch()
        if not isinstance(config, TrainerConfig):
            raise ValueError("config must be TrainerConfig")
        self.config = config
        self.device = resolve_device(config.device)
        seed_everything(config.seed, deterministic_algorithms=config.deterministic_algorithms)
        model.to(self.device)
        self.model = maybe_wrap_ddp(model, config, self.device)
        self.checkpoints = checkpoint_manager
        self.optimizer = build_adamw(
            self.model,
            learning_rate=config.stages[0].learning_rate,
            weight_decay=config.stages[0].weight_decay,
        )
        scaler_enabled = config.precision == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.scheduler: Any | None = None
        self.parent_checkpoint_digest: str | None = None

    def _set_stage_optimizer(self, stage: TrainingStageSpec) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = stage.learning_rate
            if group.get("weight_decay", 0.0) != 0.0:
                group["weight_decay"] = stage.weight_decay
        self.scheduler = build_stage_scheduler(self.optimizer, stage)

    def _is_primary(self) -> bool:
        return not self.config.ddp or dist is None or not dist.is_initialized() or dist.get_rank() == 0

    def _checkpoint(
        self,
        state: TrainerState,
        runtime: StageRuntime,
        *,
        stage_boundary: bool,
        metrics: Mapping[str, float] | None = None,
    ) -> str | None:
        if not self._is_primary():
            return None
        sampler_state = runtime.sampler.state_dict() if runtime.sampler is not None and hasattr(runtime.sampler, "state_dict") else {}
        collator_state = runtime.collator.state_dict() if runtime.collator is not None and hasattr(runtime.collator, "state_dict") else {}
        manifest = self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            trainer_state=state,
            sampler_state=sampler_state,
            collator_state=collator_state,
            source_commit=self.config.source_commit,
            training_config_digest=self.config.digest,
            dataset_manifest_digest=self.config.dataset_manifest_digest,
            model_architecture=self.config.model_architecture,
            parent_checkpoint_digest=self.parent_checkpoint_digest,
            stage_boundary=stage_boundary,
            metric_snapshot=metrics or {},
        )
        self.parent_checkpoint_digest = manifest.digest
        return manifest.digest

    def _resume(self, digest: str, runtime: StageRuntime) -> TrainerState:
        loaded = self.checkpoints.load(
            digest,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            expected_source_commit=self.config.source_commit,
            expected_training_config_digest=self.config.digest,
            expected_dataset_manifest_digest=self.config.dataset_manifest_digest,
            expected_model_architecture=self.config.model_architecture,
        )
        if runtime.sampler is not None and hasattr(runtime.sampler, "load_state_dict"):
            runtime.sampler.load_state_dict(loaded.sampler_state)
        if runtime.collator is not None and hasattr(runtime.collator, "load_state_dict"):
            runtime.collator.load_state_dict(loaded.collator_state)
        self.parent_checkpoint_digest = loaded.manifest.digest
        return loaded.trainer_state

    def _improved(self, value: float, best: float | None) -> bool:
        if best is None:
            return True
        delta = self.config.early_stopping_min_delta
        if self.config.early_stopping_mode == "max":
            return value > best + delta
        return value < best - delta

    def fit(
        self,
        stage_runtimes: Sequence[StageRuntime],
        *,
        resume_checkpoint_digest: str | None = None,
        evaluator: Evaluator | None = None,
        hard_negative_refresh: HardNegativeRefreshHook | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> TrainingSummary:
        if len(stage_runtimes) != len(self.config.stages):
            raise ValueError("stage_runtimes must align one-to-one with TrainerConfig.stages")
        if any(not isinstance(runtime, StageRuntime) for runtime in stage_runtimes):
            raise ValueError("stage_runtimes contains an invalid runtime")

        cursor = TrainerCursor(0, 0, 0, 0, 0, 0, 0)
        state = TrainerState(self.config.run_id, cursor)
        resume_pending = resume_checkpoint_digest
        latest_digest: str | None = None
        stopped_early = False
        completed_stages = 0

        for stage_index, (stage, runtime) in enumerate(zip(self.config.stages, stage_runtimes)):
            if stage_index < state.cursor.stage_index:
                completed_stages += 1
                continue
            self._set_stage_optimizer(stage)
            if resume_pending is not None:
                state = self._resume(resume_pending, runtime)
                resume_pending = None
                if state.cursor.stage_index != stage_index:
                    if state.cursor.stage_index > stage_index:
                        completed_stages += 1
                        continue
                    raise ValueError("resume checkpoint stage index is behind current orchestration stage")
            elif stage_index != state.cursor.stage_index:
                state = TrainerState(
                    run_id=self.config.run_id,
                    cursor=TrainerCursor(
                        stage_index=stage_index,
                        epoch=0,
                        batch_in_epoch=0,
                        global_step=state.cursor.global_step,
                        optimizer_step=0,
                        examples_seen=state.cursor.examples_seen,
                        tokens_seen=state.cursor.tokens_seen,
                    ),
                    best_metric=state.best_metric,
                    best_checkpoint_digest=state.best_checkpoint_digest,
                    early_stopping_bad_steps=state.early_stopping_bad_steps,
                    stage_name=stage.name,
                )

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            stage_optimizer_step = state.cursor.optimizer_step
            accumulation = 0
            epoch = state.cursor.epoch
            batch_in_epoch = state.cursor.batch_in_epoch

            while stage_optimizer_step < stage.max_optimizer_steps:
                produced_batch = False
                for batch in runtime.dataloader:
                    produced_batch = True
                    batch = move_to_device(batch, self.device)
                    with _autocast(self.device, self.config.precision):
                        step_result = runtime.step(self.model, batch)
                        loss = step_result.loss
                        if loss.ndim != 0 or not torch.isfinite(loss):
                            raise RuntimeError("training step produced a non-finite or non-scalar loss")
                        scaled_loss = loss / self.config.gradient_accumulation_steps
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    accumulation += 1
                    batch_in_epoch += 1
                    global_step = state.cursor.global_step + 1
                    batch_size = len(batch.get("query_ids", ())) or len(batch.get("group_sizes", ()))
                    examples_seen = state.cursor.examples_seen + int(batch_size)
                    state = TrainerState(
                        run_id=self.config.run_id,
                        cursor=TrainerCursor(
                            stage_index=stage_index,
                            epoch=epoch,
                            batch_in_epoch=batch_in_epoch,
                            global_step=global_step,
                            optimizer_step=stage_optimizer_step,
                            examples_seen=examples_seen,
                            tokens_seen=state.cursor.tokens_seen,
                        ),
                        best_metric=state.best_metric,
                        best_checkpoint_digest=state.best_checkpoint_digest,
                        early_stopping_bad_steps=state.early_stopping_bad_steps,
                        stage_name=stage.name,
                    )
                    if event_sink is not None and self._is_primary():
                        event_sink(
                            {
                                "event": "microbatch",
                                "stage": stage.name,
                                "global_step": global_step,
                                "optimizer_step": stage_optimizer_step,
                                "loss": float(loss.detach().cpu()),
                                **step_result.metrics,
                            }
                        )
                    if accumulation < self.config.gradient_accumulation_steps:
                        continue

                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    accumulation = 0
                    stage_optimizer_step += 1
                    state = TrainerState(
                        run_id=self.config.run_id,
                        cursor=TrainerCursor(
                            stage_index=stage_index,
                            epoch=epoch,
                            batch_in_epoch=batch_in_epoch,
                            global_step=global_step,
                            optimizer_step=stage_optimizer_step,
                            examples_seen=examples_seen,
                            tokens_seen=state.cursor.tokens_seen,
                        ),
                        best_metric=state.best_metric,
                        best_checkpoint_digest=state.best_checkpoint_digest,
                        early_stopping_bad_steps=state.early_stopping_bad_steps,
                        stage_name=stage.name,
                    )

                    metrics: Mapping[str, float] | None = None
                    if evaluator is not None and stage.evaluate_every_steps is not None and stage_optimizer_step % stage.evaluate_every_steps == 0:
                        self.model.eval()
                        with torch.no_grad():
                            metrics = {
                                key: _finite(value, f"evaluation metric {key}")
                                for key, value in evaluator(
                                    self.model,
                                    stage_index=stage_index,
                                    optimizer_step=stage_optimizer_step,
                                ).items()
                            }
                        self.model.train()
                        if self.config.early_stopping_metric is not None:
                            metric_name = self.config.early_stopping_metric
                            if metric_name not in metrics:
                                raise ValueError(f"evaluator omitted early-stopping metric {metric_name}")
                            value = metrics[metric_name]
                            if self._improved(value, state.best_metric):
                                state = TrainerState(
                                    run_id=state.run_id,
                                    cursor=state.cursor,
                                    best_metric=value,
                                    best_checkpoint_digest=state.best_checkpoint_digest,
                                    early_stopping_bad_steps=0,
                                    stage_name=state.stage_name,
                                )
                                best_candidate = self._checkpoint(state, runtime, stage_boundary=False, metrics=metrics)
                                if best_candidate is not None:
                                    state = TrainerState(
                                        run_id=state.run_id,
                                        cursor=state.cursor,
                                        best_metric=value,
                                        best_checkpoint_digest=best_candidate,
                                        early_stopping_bad_steps=0,
                                        stage_name=state.stage_name,
                                    )
                            else:
                                state = TrainerState(
                                    run_id=state.run_id,
                                    cursor=state.cursor,
                                    best_metric=state.best_metric,
                                    best_checkpoint_digest=state.best_checkpoint_digest,
                                    early_stopping_bad_steps=state.early_stopping_bad_steps + 1,
                                    stage_name=state.stage_name,
                                )
                                if (
                                    self.config.early_stopping_patience is not None
                                    and state.early_stopping_bad_steps >= self.config.early_stopping_patience
                                ):
                                    stopped_early = True

                    if stage_optimizer_step % stage.checkpoint_every_steps == 0:
                        latest_digest = self._checkpoint(state, runtime, stage_boundary=False, metrics=metrics)

                    if (
                        hard_negative_refresh is not None
                        and stage.hard_negative_refresh_every_steps is not None
                        and stage_optimizer_step % stage.hard_negative_refresh_every_steps == 0
                    ):
                        hard_negative_refresh(
                            self.model,
                            stage_index=stage_index,
                            optimizer_step=stage_optimizer_step,
                        )

                    if stopped_early or stage_optimizer_step >= stage.max_optimizer_steps:
                        break

                if not produced_batch:
                    raise RuntimeError("training dataloader produced no batches")
                if stopped_early or stage_optimizer_step >= stage.max_optimizer_steps:
                    break
                epoch += 1
                batch_in_epoch = 0
                state = TrainerState(
                    run_id=state.run_id,
                    cursor=TrainerCursor(
                        stage_index=stage_index,
                        epoch=epoch,
                        batch_in_epoch=0,
                        global_step=state.cursor.global_step,
                        optimizer_step=stage_optimizer_step,
                        examples_seen=state.cursor.examples_seen,
                        tokens_seen=state.cursor.tokens_seen,
                    ),
                    best_metric=state.best_metric,
                    best_checkpoint_digest=state.best_checkpoint_digest,
                    early_stopping_bad_steps=state.early_stopping_bad_steps,
                    stage_name=stage.name,
                )

            latest_digest = self._checkpoint(state, runtime, stage_boundary=True)
            completed_stages += 1
            if stopped_early:
                break
            if stage_index + 1 < len(self.config.stages):
                state = TrainerState(
                    run_id=state.run_id,
                    cursor=TrainerCursor(
                        stage_index=stage_index + 1,
                        epoch=0,
                        batch_in_epoch=0,
                        global_step=state.cursor.global_step,
                        optimizer_step=0,
                        examples_seen=state.cursor.examples_seen,
                        tokens_seen=state.cursor.tokens_seen,
                    ),
                    best_metric=state.best_metric,
                    best_checkpoint_digest=state.best_checkpoint_digest,
                    early_stopping_bad_steps=0,
                    stage_name=self.config.stages[stage_index + 1].name,
                )

        return TrainingSummary(
            stopped_early=stopped_early,
            completed_stages=completed_stages,
            optimizer_steps=state.cursor.optimizer_step,
            global_steps=state.cursor.global_step,
            latest_checkpoint_digest=latest_digest,
            best_metric=state.best_metric,
            best_checkpoint_digest=state.best_checkpoint_digest,
        )


__all__ = [
    "BatchStep",
    "ColBERTContrastiveStep",
    "DenseContrastiveStep",
    "Evaluator",
    "HardNegativeRefreshHook",
    "ListwiseCrossEncoderStep",
    "SparseContrastiveStep",
    "StageRuntime",
    "StepResult",
    "TorchTrainingEngine",
    "TrainerConfig",
    "TrainingStageSpec",
    "TrainingSummary",
    "build_adamw",
    "build_stage_scheduler",
    "move_to_device",
    "resolve_device",
    "seed_everything",
]
