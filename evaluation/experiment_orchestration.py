"""Repository-owned repeated experiment, shadow comparison and ablation orchestration.

The orchestration source is complete but inert until a caller supplies benchmark runners
and examples.  It gives later real-stack execution one deterministic contract for:

* current-vs-shadow paired runs across explicit seeds/repetitions;
* exact query-order and dataset-manifest binding;
* resource observation hooks instead of invented resource numbers;
* ablation matrices and historical regression baselines; and
* immutable run/series digests suitable for the statistical promotion layer.

No model, dataset, network or benchmark is executed on import.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

_MAX_EXAMPLES = 10_000_000
_MAX_RUNS = 100_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StackRole(str, Enum):
    CURRENT = "current"
    SHADOW = "shadow"
    ABLATION = "ablation"


@dataclass(frozen=True)
class BenchmarkExample:
    query_id: str
    payload_digest: str
    group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        digest = _identifier(self.payload_digest, "payload_digest", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("payload_digest must be SHA-256")
        object.__setattr__(self, "payload_digest", digest)
        if self.group_id is not None:
            object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))


@dataclass(frozen=True)
class ResourceObservation:
    wall_clock_ms: float | None = None
    peak_memory_bytes: int | None = None
    device_peak_memory_bytes: int | None = None
    storage_bytes: int | None = None
    provider_cost_units: float | None = None
    measured_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("wall_clock_ms", "provider_cost_units"):
            value = getattr(self, name)
            if value is not None:
                selected = _finite(value, name)
                if selected < 0.0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, selected)
        for name in ("peak_memory_bytes", "device_peak_memory_bytes", "storage_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(
            self,
            "measured_fields",
            tuple(_identifier(value, "measured resource field", 300) for value in self.measured_fields),
        )
        allowed = {
            "wall_clock_ms",
            "peak_memory_bytes",
            "device_peak_memory_bytes",
            "storage_bytes",
            "provider_cost_units",
        }
        if not set(self.measured_fields) <= allowed:
            raise ValueError("measured_fields contains an unsupported resource field")
        for field_name in self.measured_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"measured field {field_name} has no observation")


@dataclass(frozen=True)
class BenchmarkOutput:
    query_id: str
    metrics: Mapping[str, float]
    output_digest: str
    resources: ResourceObservation = ResourceObservation()
    trace_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        if not isinstance(self.metrics, Mapping) or not self.metrics or len(self.metrics) > 10_000:
            raise ValueError("metrics must be a non-empty bounded mapping")
        object.__setattr__(
            self,
            "metrics",
            {_identifier(key, "metric", 300): _finite(value, "metric value") for key, value in self.metrics.items()},
        )
        for name in ("output_digest", "trace_digest"):
            value = getattr(self, name)
            if value is not None:
                digest = _identifier(value, name, 64).lower()
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    raise ValueError(f"{name} must be SHA-256")
                object.__setattr__(self, name, digest)
        if not isinstance(self.resources, ResourceObservation):
            raise ValueError("resources must be ResourceObservation")


class BenchmarkRunner(Protocol):
    @property
    def stack_id(self) -> str: ...

    @property
    def stack_manifest_digest(self) -> str: ...

    def run_example(self, example: BenchmarkExample, *, seed: int) -> BenchmarkOutput: ...


class ResourceObserver(Protocol):
    def begin(self) -> Any: ...

    def end(self, token: Any) -> ResourceObservation: ...


class WallClockObserver:
    """Minimal measured wall-clock observer; optional and only active when explicitly invoked."""

    def begin(self) -> int:
        return time.perf_counter_ns()

    def end(self, token: Any) -> ResourceObservation:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("wall-clock observer token is invalid")
        elapsed_ms = max(0.0, (time.perf_counter_ns() - token) / 1_000_000.0)
        return ResourceObservation(wall_clock_ms=elapsed_ms, measured_fields=("wall_clock_ms",))


@dataclass(frozen=True)
class ExperimentSchedule:
    experiment_id: str
    dataset_manifest_digest: str
    seeds: tuple[int, ...]
    repetitions_per_seed: int = 1
    fail_fast: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _identifier(self.experiment_id, "experiment_id"))
        digest = _identifier(self.dataset_manifest_digest, "dataset_manifest_digest", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("dataset_manifest_digest must be SHA-256")
        object.__setattr__(self, "dataset_manifest_digest", digest)
        if not self.seeds or len(self.seeds) > _MAX_RUNS:
            raise ValueError("seeds must be non-empty and bounded")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1 for seed in self.seeds):
            raise ValueError("seeds must be non-negative 63-bit integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if isinstance(self.repetitions_per_seed, bool) or not isinstance(self.repetitions_per_seed, int) or not 1 <= self.repetitions_per_seed <= _MAX_RUNS:
            raise ValueError("repetitions_per_seed is invalid")
        if len(self.seeds) * self.repetitions_per_seed > _MAX_RUNS:
            raise ValueError("scheduled run count exceeds safety bound")
        if not isinstance(self.fail_fast, bool):
            raise ValueError("fail_fast must be boolean")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class RunIdentity:
    seed: int
    repetition: int


@dataclass(frozen=True)
class StackRun:
    role: StackRole
    stack_id: str
    stack_manifest_digest: str
    identity: RunIdentity
    outputs: tuple[BenchmarkOutput, ...]
    error_type: str | None = None

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class PairedExperimentSeries:
    schedule: ExperimentSchedule
    query_contract_digest: str
    current_runs: tuple[StackRun, ...]
    shadow_runs: tuple[StackRun, ...]

    def __post_init__(self) -> None:
        if len(self.current_runs) != len(self.shadow_runs):
            raise ValueError("current and shadow run counts differ")
        for current, shadow in zip(self.current_runs, self.shadow_runs):
            if current.identity != shadow.identity:
                raise ValueError("paired current/shadow run identities differ")
            current_ids = tuple(output.query_id for output in current.outputs)
            shadow_ids = tuple(output.query_id for output in shadow.outputs)
            if current_ids != shadow_ids:
                raise ValueError("paired current/shadow query order differs")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


def query_contract_digest(examples: Sequence[BenchmarkExample]) -> str:
    if not examples or len(examples) > _MAX_EXAMPLES:
        raise ValueError("examples must be non-empty and bounded")
    query_ids = [example.query_id for example in examples]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique within a benchmark contract")
    return canonical_digest([asdict(example) for example in examples])


def _validate_stack_digest(value: str, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def execute_stack_run(
    runner: BenchmarkRunner,
    examples: Sequence[BenchmarkExample],
    *,
    role: StackRole,
    identity: RunIdentity,
    observer_factory: Callable[[], ResourceObserver] | None = None,
) -> StackRun:
    """Execute one explicitly requested stack run while enforcing query-order contracts."""

    stack_id = _identifier(runner.stack_id, "stack_id")
    manifest = _validate_stack_digest(runner.stack_manifest_digest, "stack_manifest_digest")
    outputs: list[BenchmarkOutput] = []
    for example in examples:
        observer = observer_factory() if observer_factory is not None else None
        token = observer.begin() if observer is not None else None
        output = runner.run_example(example, seed=identity.seed)
        measured = observer.end(token) if observer is not None else None
        if output.query_id != example.query_id:
            raise ValueError("runner changed query_id or query order")
        if measured is not None:
            output = BenchmarkOutput(
                query_id=output.query_id,
                metrics=output.metrics,
                output_digest=output.output_digest,
                resources=measured,
                trace_digest=output.trace_digest,
            )
        outputs.append(output)
    return StackRun(StackRole(role), stack_id, manifest, identity, tuple(outputs))


def execute_paired_series(
    current: BenchmarkRunner,
    shadow: BenchmarkRunner,
    examples: Sequence[BenchmarkExample],
    schedule: ExperimentSchedule,
    *,
    observer_factory: Callable[[], ResourceObserver] | None = None,
) -> PairedExperimentSeries:
    """Execute the exact repeated current/shadow schedule when a caller explicitly invokes it."""

    contract = query_contract_digest(examples)
    current_runs: list[StackRun] = []
    shadow_runs: list[StackRun] = []
    for seed in schedule.seeds:
        for repetition in range(schedule.repetitions_per_seed):
            identity = RunIdentity(seed, repetition)
            current_runs.append(
                execute_stack_run(
                    current,
                    examples,
                    role=StackRole.CURRENT,
                    identity=identity,
                    observer_factory=observer_factory,
                )
            )
            shadow_runs.append(
                execute_stack_run(
                    shadow,
                    examples,
                    role=StackRole.SHADOW,
                    identity=identity,
                    observer_factory=observer_factory,
                )
            )
    return PairedExperimentSeries(schedule, contract, tuple(current_runs), tuple(shadow_runs))


@dataclass(frozen=True)
class AblationVariant:
    variant_id: str
    disabled_components: tuple[str, ...] = ()
    overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant_id", _identifier(self.variant_id, "variant_id"))
        object.__setattr__(
            self,
            "disabled_components",
            tuple(_identifier(value, "disabled component") for value in self.disabled_components),
        )
        if len(set(self.disabled_components)) != len(self.disabled_components):
            raise ValueError("disabled_components must be unique")
        if not isinstance(self.overrides, Mapping) or len(self.overrides) > 10_000:
            raise ValueError("overrides must be a bounded mapping")
        object.__setattr__(
            self,
            "overrides",
            {
                _identifier(key, "override key", 500): _identifier(value, "override value", 5_000)
                for key, value in self.overrides.items()
            },
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class HistoricalRegressionBaseline:
    baseline_id: str
    source_commit: str
    dataset_manifest_digest: str
    stack_manifest_digest: str
    metric_means: Mapping[str, float]
    allowed_regressions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_id", _identifier(self.baseline_id, "baseline_id"))
        for name in ("source_commit", "dataset_manifest_digest", "stack_manifest_digest"):
            object.__setattr__(self, name, _validate_stack_digest(getattr(self, name), name))
        if not self.metric_means or not isinstance(self.metric_means, Mapping):
            raise ValueError("metric_means must be non-empty")
        means = {_identifier(key, "metric", 300): _finite(value, "metric mean") for key, value in self.metric_means.items()}
        tolerances = {
            _identifier(key, "metric", 300): _finite(value, "allowed regression")
            for key, value in self.allowed_regressions.items()
        }
        if any(value < 0.0 for value in tolerances.values()):
            raise ValueError("allowed regressions must be non-negative")
        if not set(tolerances) <= set(means):
            raise ValueError("allowed_regressions references unknown metrics")
        object.__setattr__(self, "metric_means", means)
        object.__setattr__(self, "allowed_regressions", tolerances)


def compare_historical_regression(
    observed_metric_means: Mapping[str, Any],
    baseline: HistoricalRegressionBaseline,
    *,
    higher_is_better: Mapping[str, bool],
) -> tuple[str, ...]:
    """Return deterministic regression reason strings; empty means within configured tolerances."""

    reasons: list[str] = []
    for metric, reference in baseline.metric_means.items():
        if metric not in observed_metric_means:
            reasons.append(f"missing metric {metric}")
            continue
        if metric not in higher_is_better:
            reasons.append(f"missing direction for metric {metric}")
            continue
        observed = _finite(observed_metric_means[metric], f"observed {metric}")
        tolerance = baseline.allowed_regressions.get(metric, 0.0)
        regression = reference - observed if higher_is_better[metric] else observed - reference
        if regression > tolerance:
            reasons.append(f"{metric} regression {regression:.6g} exceeds tolerance {tolerance:.6g}")
    return tuple(reasons)


__all__ = [
    "AblationVariant",
    "BenchmarkExample",
    "BenchmarkOutput",
    "BenchmarkRunner",
    "ExperimentSchedule",
    "HistoricalRegressionBaseline",
    "PairedExperimentSeries",
    "ResourceObservation",
    "ResourceObserver",
    "RunIdentity",
    "StackRole",
    "StackRun",
    "WallClockObserver",
    "canonical_digest",
    "compare_historical_regression",
    "execute_paired_series",
    "execute_stack_run",
    "query_contract_digest",
]
