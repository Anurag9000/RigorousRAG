"""Synchronized shared-batch adapter for RigorousRAG learned retrieval.

This module does not introduce new model/loss science. It reuses the authoritative
retrieval CLI's governed dataset, model constructors, collators, batch steps,
validation evaluator, stage specs and ``TorchTrainingEngine`` optimizer/checkpoint
machinery, while moving ownership of the *outer* training batch cursor to the
canonical dataset-cohort runtime.

All retrieval recipes in one cohort consume the same ``RetrievalTrainingExample``
objects for a physical microbatch. Distinct tokenizer/collator contracts are cached
once as device views; recipes whose complete view identity is equal reuse the same
GPU tensor storage. The uniform physical batch is the minimum safe recipe batch.
For each model, gradient accumulation is scaled exactly so
``original_batch * original_accumulation`` is unchanged; non-divisible recipes fail
closed.

Per-model Torch RNG state is virtualized behind a lock because dropout and similar
PyTorch modules consume device-global RNG. Models remain simultaneously resident
in VRAM and reuse the same device batch, while stochastic forward/backward sections
are serialized to retain interruption equivalence. Repositories/adapters that can
prove stream-local RNG safety may use the runtime's true concurrent CUDA streams.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from training import authoritative_retrieval_training_cli as cli
from training.checkpoint_control import set_checkpoint_pointer
from training.checkpointing import CheckpointManager, TrainerCursor, TrainerState, sha256_file
from training.data_pipeline import ManifestBoundJsonlDataset, ResumableDeterministicSampler
from training.torch_engine import StageRuntime, TorchTrainingEngine, TrainerConfig, move_to_device, seed_everything
from training.dataset_cohort_contract import model_spec_from_job
from tools.dataset_cohort_runtime_entry import load_runtime

runtime = load_runtime()

SCHEMA = "rigorousrag-retrieval-dataset-cohort/v1"
_RNG_LOCK = threading.RLock()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.replace(temporary, path)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("cohort exact resume requires PyTorch weights_only checkpoint loading") from exc


def _b64(state: torch.Tensor) -> str:
    return base64.b64encode(bytes(state.detach().cpu().tolist())).decode("ascii")


def _unb64(value: str) -> torch.Tensor:
    return torch.tensor(list(base64.b64decode(value, validate=True)), dtype=torch.uint8)


def _config_path(spec: Any) -> Path:
    raw = spec.metadata.get("recipe_config")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{spec.job_id}: recipe_config metadata is required")
    path = Path(raw).expanduser()
    return (cli._REPO_ROOT / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)


def _resolved(config_path: Path) -> dict[str, Any]:
    config = cli._load_config(config_path)
    base = config_path.parent
    architecture = cli._identifier(config.get("architecture"), "architecture", 100).lower()
    if architecture not in cli._ARCHITECTURES:
        raise ValueError(f"unsupported retrieval architecture {architecture!r}")
    if bool(config.get("ddp", False)):
        raise ValueError("dataset-cohort retrieval requires single-process model adapters; DDP is not cohort-safe")
    train_path = cli._path(base, config.get("train_data"), "train_data")
    validation_path = cli._path(base, config.get("validation_data"), "validation_data")
    model_root = cli._path(base, config.get("model_root"), "model_root", directory=True)
    tokenizer_root = cli._path(base, config.get("tokenizer_root", config.get("model_root")), "tokenizer_root", directory=True)
    output_dir = cli._output_path(base, config.get("output_dir"))
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    dataset_digest = cli._digest({"schema": "rigorousrag-retrieval-dataset-binding/v1", "train_sha256": train_sha, "validation_sha256": validation_sha})
    data = dict(config.get("data", {}))
    batch_size = int(data.get("batch_size", 8))
    validation_batch_size = int(data.get("validation_batch_size", batch_size))
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("retrieval batch sizes must be positive")
    original_accum = int(config.get("gradient_accumulation_steps", 1))
    if original_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    seed = int(config.get("seed", 0))
    stages = cli._stage_specs(config)
    view_identity = {
        "schema": "rigorousrag-retrieval-view/v1",
        "architecture": architecture,
        "tokenizer_tree_sha256": cli._tree_sha256(tokenizer_root),
        "tokenizer_revision": config.get("tokenizer_revision"),
        "seed": seed,
        "data": {
            "query_max_length": data.get("query_max_length", 64),
            "document_max_length": data.get("document_max_length", 512),
            "pair_max_length": data.get("pair_max_length", 512),
            "negatives_per_query": data.get("negatives_per_query", 8),
            "pad_to_multiple_of": data.get("pad_to_multiple_of", 8),
        },
    }
    return {
        "config": config,
        "architecture": architecture,
        "train_path": train_path,
        "validation_path": validation_path,
        "model_root": model_root,
        "tokenizer_root": tokenizer_root,
        "output_dir": output_dir,
        "train_sha": train_sha,
        "validation_sha": validation_sha,
        "dataset_digest": dataset_digest,
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "original_accum": original_accum,
        "seed": seed,
        "stages": stages,
        "view_key": "retrieval-view:" + _digest(view_identity),
        "view_identity": view_identity,
    }


def cohort_job_metadata(config_path: str | Path) -> dict[str, Any]:
    resolved = _resolved(Path(config_path).expanduser().resolve(strict=True))
    return {
        "model_family": resolved["architecture"],
        "cohort_view_key": resolved["view_key"],
        "cohort_batch_size": resolved["batch_size"],
        "cohort_adapter": "rigorousrag-retrieval",
        "cpu_capable": True,
        "gpu_capable": True,
        "cohort_contract_schema": SCHEMA,
    }


class RetrievalDatasetAdapter:
    dataset_key = "retrieval"

    def __init__(self, cohort: Any) -> None:
        if cohort.dataset_key != self.dataset_key:
            raise ValueError(f"retrieval adapter cannot serve {cohort.dataset_key!r}")
        self.cohort = cohort
        bindings = [_resolved(_config_path(spec)) for spec in cohort.models]
        first = bindings[0]
        for row in bindings[1:]:
            for key in ("train_path", "train_sha", "validation_sha", "dataset_digest"):
                if row[key] != first[key]:
                    raise ValueError(f"retrieval cohort mixes incompatible governed data identity at {key}")
        self.dataset = ManifestBoundJsonlDataset(
            first["train_path"],
            expected_sha256=first["train_sha"],
            dataset_manifest_digest=first["dataset_digest"],
            split_name="train",
        )
        if len(self.dataset) == 0:
            raise ValueError("retrieval cohort train split is empty")
        max_microbatches = 0
        for spec, row in zip(cohort.models, bindings):
            target_effective = row["batch_size"] * row["original_accum"]
            if target_effective % cohort.uniform_batch_size:
                raise ValueError(
                    f"{spec.job_id}: effective batch {target_effective} is not divisible by cohort physical batch {cohort.uniform_batch_size}"
                )
            accumulation = target_effective // cohort.uniform_batch_size
            required = sum(int(stage.max_optimizer_steps) * accumulation for stage in row["stages"])
            max_microbatches = max(max_microbatches, required)
        self.max_microbatches = max_microbatches
        self.emitted = 0
        self.sampler: ResumableDeterministicSampler | None = None
        self._pending_sampler_state: Mapping[str, Any] | None = None

    def iter_raw_batches(self, *, batch_size: int, start_batch: int, seed: int) -> Iterator[Any]:
        if batch_size != self.cohort.uniform_batch_size:
            raise ValueError("cohort physical batch size drift")
        if self.sampler is None:
            self.sampler = ResumableDeterministicSampler(len(self.dataset), seed=seed, shuffle=True)
            if self._pending_sampler_state is not None:
                self.sampler.load_state_dict(self._pending_sampler_state)
                self._pending_sampler_state = None
        if start_batch != self.emitted:
            raise ValueError(f"shared batch cursor mismatch: executor={start_batch}, dataset={self.emitted}")
        iterator = iter(self.sampler)
        while self.emitted < self.max_microbatches:
            indices: list[int] = []
            while len(indices) < batch_size:
                try:
                    indices.append(next(iterator))
                except StopIteration:
                    iterator = iter(self.sampler)
                    try:
                        indices.append(next(iterator))
                    except StopIteration as exc:
                        raise RuntimeError("retrieval dataset unexpectedly produced no examples") from exc
            self.emitted += 1
            yield [self.dataset[index] for index in indices]

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "schema": SCHEMA,
            "emitted": self.emitted,
            "max_microbatches": self.max_microbatches,
            "sampler": None if self.sampler is None else self.sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != SCHEMA:
            raise ValueError("retrieval cohort dataset state schema mismatch")
        if int(state.get("max_microbatches", -1)) != self.max_microbatches:
            raise ValueError("retrieval cohort maximum microbatch count changed")
        self.emitted = int(state.get("emitted", 0))
        sampler = state.get("sampler")
        if sampler is not None:
            if not isinstance(sampler, Mapping):
                raise ValueError("retrieval cohort sampler state is malformed")
            self._pending_sampler_state = dict(sampler)


class RetrievalModelAdapter:
    framework = "torch"

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.resolved = _resolved(_config_path(spec))
        if spec.view_key != self.resolved["view_key"]:
            raise ValueError(f"{spec.job_id}: cohort view identity drift")
        self.view_key = spec.view_key
        self.device: torch.device | None = None
        self.engine: TorchTrainingEngine | None = None
        self.checkpoints: CheckpointManager | None = None
        self.step: Any = None
        self.collator: Any = None
        self.validation_dataset: Any = None
        self.evaluator: Any = None
        self.state: TrainerState | None = None
        self.stage_index = 0
        self.accumulation = 0
        self.total_optimizer_steps = 0
        self.done = False
        self.stopped_early = False
        self.completed_stages = 0
        self.pending_best = False
        self.pending_stage_boundary = False
        self.latest_digest: str | None = None
        self.cpu_rng: torch.Tensor | None = None
        self.cuda_rng: torch.Tensor | None = None
        self.uniform_batch_size = 0
        self.effective_accum = 1

    def _require(self) -> None:
        if self.engine is None or self.state is None or self.device is None or self.checkpoints is None:
            raise RuntimeError(f"{self.spec.job_id}: cohort model adapter is not built")

    def _set_stage(self, index: int, *, reset_state: bool) -> None:
        self._require()
        assert self.engine is not None and self.state is not None
        stages = self.resolved["stages"]
        if not 0 <= index < len(stages):
            raise ValueError("stage index out of range")
        self.stage_index = index
        stage = stages[index]
        self.engine._set_stage_optimizer(stage)
        self.engine.optimizer.zero_grad(set_to_none=True)
        factory = cli._collator_factory(
            self.resolved["architecture"],
            self.tokenizer,
            self.resolved["config"],
            seed=self.resolved["seed"] + index * 1_000_003,
        )
        self.collator = factory()
        if reset_state:
            self.state = TrainerState(
                run_id=self.state.run_id,
                cursor=TrainerCursor(
                    stage_index=index,
                    epoch=0,
                    batch_in_epoch=0,
                    global_step=self.state.cursor.global_step,
                    optimizer_step=0,
                    examples_seen=self.state.cursor.examples_seen,
                    tokens_seen=self.state.cursor.tokens_seen,
                ),
                best_metric=self.state.best_metric,
                best_checkpoint_digest=self.state.best_checkpoint_digest,
                early_stopping_bad_steps=0,
                stage_name=stage.name,
            )
            self.accumulation = 0

    def build(self, *, device: str, batch_size: int) -> None:
        r = self.resolved
        target_effective = r["batch_size"] * r["original_accum"]
        if target_effective % batch_size:
            raise ValueError(
                f"{self.spec.job_id}: original effective batch {target_effective} cannot be represented by physical batch {batch_size}"
            )
        self.uniform_batch_size = batch_size
        self.effective_accum = target_effective // batch_size
        if self.effective_accum <= 0:
            raise ValueError("effective accumulation must be positive")
        requested_device = "cpu" if device == "cpu" else device
        seed_everything(r["seed"], deterministic_algorithms=bool(r["config"].get("deterministic_algorithms", False)))
        self.tokenizer = cli._load_tokenizer(r["tokenizer_root"], r["config"].get("tokenizer_revision"))
        model_cfg = dict(r["config"].get("model", {}))
        raw_untied = model_cfg.get("untied_document_model_root")
        untied = None
        untied_sha = None
        if raw_untied is not None:
            untied = cli._path(_config_path(self.spec).parent, raw_untied, "untied_document_model_root", directory=True)
            untied_sha = cli._tree_sha256(untied)
        model = cli._architecture_model(
            r["architecture"], r["model_root"], r["config"].get("model_revision"), r["config"],
            untied_document_model_root=untied,
        )
        self.step = cli._step(r["architecture"], r["config"])
        self.validation_dataset = ManifestBoundJsonlDataset(
            r["validation_path"], expected_sha256=r["validation_sha"], dataset_manifest_digest=r["dataset_digest"], split_name="validation"
        )
        if len(self.validation_dataset) == 0:
            raise ValueError("retrieval validation split is empty")
        config_sha = sha256_file(_config_path(self.spec))
        model_sha = cli._tree_sha256(r["model_root"])
        tokenizer_sha = cli._tree_sha256(r["tokenizer_root"])
        artifact_identity = (
            f"{r['architecture']}:config={config_sha}:model={model_sha}:tokenizer={tokenizer_sha}:"
            f"untied_document={untied_sha or 'tied'}:cohort_batch={batch_size}:accum={self.effective_accum}"
        )
        early = dict(r["config"].get("early_stopping", {}))
        trainer_config = TrainerConfig(
            run_id=cli._identifier(r["config"].get("run_id", f"retrieval-{r['architecture']}"), "run_id", 500) + "-cohort-v1",
            source_commit=cli._source_commit(r["config"].get("source_commit", "auto")),
            dataset_manifest_digest=r["dataset_digest"],
            model_architecture=artifact_identity,
            stages=r["stages"],
            device=requested_device,
            precision=str(r["config"].get("precision", "fp32")),
            gradient_accumulation_steps=self.effective_accum,
            max_grad_norm=None if r["config"].get("max_grad_norm", 1.0) is None else float(r["config"].get("max_grad_norm", 1.0)),
            seed=r["seed"],
            deterministic_algorithms=bool(r["config"].get("deterministic_algorithms", False)),
            ddp=False,
            find_unused_parameters=False,
            early_stopping_metric="validation_loss",
            early_stopping_mode="min",
            early_stopping_patience=int(early.get("patience", 10)),
            early_stopping_min_delta=float(early.get("min_delta", 0.0)),
        )
        cohort_output = r["output_dir"] / "dataset_cohort_v1"
        cohort_output.mkdir(parents=True, exist_ok=True)
        self.result_path = cohort_output / "training_result.json"
        self.identity = {
            "config_sha256": config_sha,
            "source_commit": trainer_config.source_commit,
            "train_data_sha256": r["train_sha"],
            "validation_data_sha256": r["validation_sha"],
            "dataset_manifest_digest": r["dataset_digest"],
            "model_tree_sha256": model_sha,
            "tokenizer_tree_sha256": tokenizer_sha,
            "untied_document_model_tree_sha256": untied_sha,
            "model_architecture_identity": artifact_identity,
        }
        self.checkpoints = CheckpointManager(cohort_output / "checkpoints")
        self.engine = TorchTrainingEngine(model, trainer_config, self.checkpoints)
        self.device = self.engine.device
        self.evaluator = cli._validation_evaluator(
            validation_dataset=self.validation_dataset,
            architecture=r["architecture"], tokenizer=self.tokenizer, config=r["config"], step=self.step,
            device=self.device, batch_size=r["validation_batch_size"], seed=r["seed"] + 10_000_019,
        )
        self.state = TrainerState(
            run_id=trainer_config.run_id,
            cursor=TrainerCursor(0, 0, 0, 0, 0, 0, 0),
            best_metric=None,
            best_checkpoint_digest=None,
            early_stopping_bad_steps=0,
            stage_name=r["stages"][0].name,
        )
        self._set_stage(0, reset_state=False)
        self.cpu_rng = torch.get_rng_state().clone()
        if self.device.type == "cuda":
            self.cuda_rng = torch.cuda.get_rng_state(self.device).clone()

    @contextlib.contextmanager
    def _model_rng(self) -> Iterator[None]:
        self._require()
        assert self.device is not None and self.cpu_rng is not None
        with _RNG_LOCK:
            outer_cpu = torch.get_rng_state().clone()
            outer_cuda = torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
            torch.set_rng_state(self.cpu_rng)
            if self.device.type == "cuda" and self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            try:
                yield
            finally:
                self.cpu_rng = torch.get_rng_state().clone()
                if self.device.type == "cuda":
                    self.cuda_rng = torch.cuda.get_rng_state(self.device).clone()
                torch.set_rng_state(outer_cpu)
                if self.device.type == "cuda" and outer_cuda is not None:
                    torch.cuda.set_rng_state(outer_cuda, self.device)

    def prepare_view(self, raw_batch: Any, *, device: str) -> Any:
        self._require()
        if self.collator is None:
            raise RuntimeError("retrieval collator is unavailable")
        batch = self.collator(raw_batch)
        assert self.device is not None
        return move_to_device(batch, self.device)

    def _improved(self, value: float, best: float | None) -> bool:
        assert self.engine is not None
        return self.engine._improved(value, best)

    def _advance_stage_or_finish(self) -> None:
        self._require()
        assert self.state is not None
        self.completed_stages = max(self.completed_stages, self.stage_index + 1)
        self.pending_stage_boundary = True
        if self.stopped_early or self.stage_index + 1 >= len(self.resolved["stages"]):
            self.done = True
            return
        self._set_stage(self.stage_index + 1, reset_state=True)

    def train_step(self, batch: Any, *, batch_index: int) -> Mapping[str, Any] | None:
        self._require()
        if self.done:
            return {"done": True}
        assert self.engine is not None and self.state is not None
        stage = self.resolved["stages"][self.stage_index]
        with self._model_rng():
            self.engine.model.train()
            autocast = getattr(__import__("training.torch_engine", fromlist=["_autocast"]), "_autocast")
            with autocast(self.engine.device, self.engine.config.precision):
                result = self.step(self.engine.model, batch)
                loss = result.loss
                if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("cohort retrieval step produced a non-finite or non-scalar loss")
                scaled = loss / self.effective_accum
            if self.engine.scaler.is_enabled():
                self.engine.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            self.accumulation += 1
            batch_size = len(batch.get("query_ids", ())) or len(batch.get("group_sizes", ()))
            self.state = TrainerState(
                run_id=self.state.run_id,
                cursor=TrainerCursor(
                    stage_index=self.stage_index,
                    epoch=0,
                    batch_in_epoch=self.state.cursor.batch_in_epoch + 1,
                    global_step=self.state.cursor.global_step + 1,
                    optimizer_step=self.state.cursor.optimizer_step,
                    examples_seen=self.state.cursor.examples_seen + int(batch_size),
                    tokens_seen=self.state.cursor.tokens_seen,
                ),
                best_metric=self.state.best_metric,
                best_checkpoint_digest=self.state.best_checkpoint_digest,
                early_stopping_bad_steps=self.state.early_stopping_bad_steps,
                stage_name=stage.name,
            )
            if self.accumulation < self.effective_accum:
                return {"done": False, "loss": float(loss.detach().cpu()), "optimizer_step": self.state.cursor.optimizer_step}
            if self.engine.scaler.is_enabled():
                self.engine.scaler.unscale_(self.engine.optimizer)
            if self.engine.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.engine.model.parameters(), self.engine.config.max_grad_norm)
            if self.engine.scaler.is_enabled():
                self.engine.scaler.step(self.engine.optimizer)
                self.engine.scaler.update()
            else:
                self.engine.optimizer.step()
            self.engine.optimizer.zero_grad(set_to_none=True)
            if self.engine.scheduler is not None:
                self.engine.scheduler.step()
            self.accumulation = 0
            optimizer_step = self.state.cursor.optimizer_step + 1
            self.total_optimizer_steps += 1
            self.state = TrainerState(
                run_id=self.state.run_id,
                cursor=TrainerCursor(
                    stage_index=self.stage_index, epoch=0, batch_in_epoch=self.state.cursor.batch_in_epoch,
                    global_step=self.state.cursor.global_step, optimizer_step=optimizer_step,
                    examples_seen=self.state.cursor.examples_seen, tokens_seen=self.state.cursor.tokens_seen,
                ),
                best_metric=self.state.best_metric,
                best_checkpoint_digest=self.state.best_checkpoint_digest,
                early_stopping_bad_steps=self.state.early_stopping_bad_steps,
                stage_name=stage.name,
            )
            metrics: Mapping[str, float] | None = None
            if stage.evaluate_every_steps is not None and optimizer_step % stage.evaluate_every_steps == 0:
                self.engine.model.eval()
                with torch.no_grad():
                    metrics = self.evaluator(self.engine.model, stage_index=self.stage_index, optimizer_step=optimizer_step)
                self.engine.model.train()
                value = float(metrics["validation_loss"])
                if not math.isfinite(value):
                    raise RuntimeError("validation_loss is non-finite")
                if self._improved(value, self.state.best_metric):
                    self.state = TrainerState(
                        run_id=self.state.run_id, cursor=self.state.cursor, best_metric=value,
                        best_checkpoint_digest=self.state.best_checkpoint_digest, early_stopping_bad_steps=0,
                        stage_name=self.state.stage_name,
                    )
                    self.pending_best = True
                else:
                    bad = self.state.early_stopping_bad_steps + 1
                    self.state = TrainerState(
                        run_id=self.state.run_id, cursor=self.state.cursor, best_metric=self.state.best_metric,
                        best_checkpoint_digest=self.state.best_checkpoint_digest, early_stopping_bad_steps=bad,
                        stage_name=self.state.stage_name,
                    )
                    patience = self.engine.config.early_stopping_patience
                    if patience is not None and bad >= patience:
                        self.stopped_early = True
            if self.stopped_early or optimizer_step >= stage.max_optimizer_steps:
                self._advance_stage_or_finish()
            return {
                "done": self.done,
                "loss": float(loss.detach().cpu()),
                "optimizer_step": optimizer_step,
                "validation": None if metrics is None else dict(metrics),
            }

    def _runtime_for_checkpoint(self) -> StageRuntime:
        return StageRuntime((), self.step, sampler=None, collator=self.collator)

    def _gradient_payload(self) -> dict[str, Any]:
        assert self.engine is not None
        module = self.engine.model.module if hasattr(self.engine.model, "module") else self.engine.model
        return {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }

    def save_checkpoint(self, path: Path) -> Mapping[str, Any] | None:
        self._require()
        assert self.engine is not None and self.state is not None and self.checkpoints is not None
        path.mkdir(parents=True, exist_ok=True)
        with self._model_rng():
            digest = self.engine._checkpoint(
                self.state,
                self._runtime_for_checkpoint(),
                stage_boundary=self.pending_stage_boundary,
            )
        if digest is None:
            raise RuntimeError("cohort retrieval checkpoint unexpectedly produced no digest")
        self.latest_digest = digest
        if self.pending_best:
            set_checkpoint_pointer(self.checkpoints, "best", digest)
            self.state = TrainerState(
                run_id=self.state.run_id, cursor=self.state.cursor, best_metric=self.state.best_metric,
                best_checkpoint_digest=digest, early_stopping_bad_steps=self.state.early_stopping_bad_steps,
                stage_name=self.state.stage_name,
            )
            self.pending_best = False
        self.pending_stage_boundary = False
        gradients_path = path / "gradients.pt"
        temporary = gradients_path.with_suffix(".tmp")
        torch.save(self._gradient_payload(), temporary)
        os.replace(temporary, gradients_path)
        adapter_state = {
            "schema": SCHEMA,
            "job_id": self.spec.job_id,
            "native_checkpoint_digest": digest,
            "stage_index": self.stage_index,
            "accumulation": self.accumulation,
            "total_optimizer_steps": self.total_optimizer_steps,
            "done": self.done,
            "stopped_early": self.stopped_early,
            "completed_stages": self.completed_stages,
            "pending_best": self.pending_best,
            "pending_stage_boundary": self.pending_stage_boundary,
            "effective_accum": self.effective_accum,
            "uniform_batch_size": self.uniform_batch_size,
            "cpu_rng": _b64(self.cpu_rng),
            "cuda_rng": None if self.cuda_rng is None else _b64(self.cuda_rng),
            "gradients_sha256": sha256_file(gradients_path),
        }
        _atomic_json(path / "adapter_state.json", adapter_state)
        if self.done:
            self._write_result()
        return {
            "schema": SCHEMA,
            "native_checkpoint_digest": digest,
            "adapter_state_sha256": sha256_file(path / "adapter_state.json"),
            "gradients_sha256": adapter_state["gradients_sha256"],
            "done": self.done,
        }

    def load_checkpoint(self, path: Path, metadata: Mapping[str, Any]) -> None:
        self._require()
        if not metadata:
            return
        state_path = path / "adapter_state.json"
        gradients_path = path / "gradients.pt"
        if sha256_file(state_path) != metadata.get("adapter_state_sha256"):
            raise RuntimeError("cohort adapter-state digest mismatch")
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA or raw.get("job_id") != self.spec.job_id:
            raise RuntimeError("cohort adapter-state identity mismatch")
        if int(raw["uniform_batch_size"]) != self.uniform_batch_size or int(raw["effective_accum"]) != self.effective_accum:
            raise RuntimeError("cohort batch/accumulation contract changed across resume")
        self.stage_index = int(raw["stage_index"])
        assert self.state is not None and self.engine is not None and self.checkpoints is not None
        self._set_stage(self.stage_index, reset_state=False)
        digest = str(raw["native_checkpoint_digest"])
        loaded = self.checkpoints.load(
            digest,
            model=self.engine.model,
            optimizer=self.engine.optimizer,
            scheduler=self.engine.scheduler,
            scaler=self.engine.scaler if self.engine.scaler.is_enabled() else None,
            expected_source_commit=self.engine.config.source_commit,
            expected_training_config_digest=self.engine.config.digest,
            expected_dataset_manifest_digest=self.engine.config.dataset_manifest_digest,
            expected_model_architecture=self.engine.config.model_architecture,
            restore_rng=False,
        )
        self.state = loaded.trainer_state
        if loaded.collator_state and hasattr(self.collator, "load_state_dict"):
            self.collator.load_state_dict(loaded.collator_state)
        if sha256_file(gradients_path) != raw["gradients_sha256"] or raw["gradients_sha256"] != metadata.get("gradients_sha256"):
            raise RuntimeError("cohort gradient-state digest mismatch")
        gradients = _torch_load(gradients_path)
        module = self.engine.model.module if hasattr(self.engine.model, "module") else self.engine.model
        by_name = dict(module.named_parameters())
        for name, value in gradients.items():
            if name not in by_name:
                raise RuntimeError(f"checkpoint gradient names unknown parameter {name!r}")
            by_name[name].grad = value.to(by_name[name].device, dtype=by_name[name].dtype)
        self.accumulation = int(raw["accumulation"])
        self.total_optimizer_steps = int(raw["total_optimizer_steps"])
        self.done = bool(raw["done"])
        self.stopped_early = bool(raw["stopped_early"])
        self.completed_stages = int(raw["completed_stages"])
        self.pending_best = bool(raw["pending_best"])
        self.pending_stage_boundary = bool(raw["pending_stage_boundary"])
        self.cpu_rng = _unb64(raw["cpu_rng"])
        self.cuda_rng = None if raw.get("cuda_rng") is None else _unb64(raw["cuda_rng"])
        self.latest_digest = digest

    def _write_result(self) -> None:
        assert self.engine is not None and self.state is not None
        result = {
            "schema": cli.RESULT_SCHEMA,
            "complete": True,
            "architecture": self.resolved["architecture"],
            "step_variant": str(self.resolved["config"].get("step_variant", "base")).strip().lower(),
            **self.identity,
            "trainer_config_digest": self.engine.config.digest,
            "dataset_cohort_schema": SCHEMA,
            "uniform_physical_batch_size": self.uniform_batch_size,
            "effective_gradient_accumulation_steps": self.effective_accum,
            "summary": {
                "stopped_early": self.stopped_early,
                "completed_stages": self.completed_stages,
                "optimizer_steps": self.total_optimizer_steps,
                "global_steps": self.state.cursor.global_step,
                "latest_checkpoint_digest": self.latest_digest,
                "best_metric": self.state.best_metric,
                "best_checkpoint_digest": self.state.best_checkpoint_digest,
            },
        }
        result["result_sha256"] = cli._digest(result)
        cli._atomic_json(self.result_path, result)

    def close(self) -> None:
        if self.done and self.engine is not None:
            self._write_result()
        self.validation_dataset = None
        self.collator = None
        self.step = None
        self.evaluator = None
        self.tokenizer = None
        self.engine = None
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()


def build_registry(cohort: Any | None = None) -> Any:
    registry = runtime.AdapterRegistry()
    registry.register_model("rigorousrag-retrieval", RetrievalModelAdapter)
    registry.register_dataset("retrieval", RetrievalDatasetAdapter)
    return registry


def specs_from_jobs(jobs: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return tuple(model_spec_from_job(job) for job in jobs if str(job.get("dataset")) == "retrieval" and str(job.get("phase")) == "training")


__all__ = ["RetrievalDatasetAdapter", "RetrievalModelAdapter", "SCHEMA", "build_registry", "cohort_job_metadata", "specs_from_jobs"]
