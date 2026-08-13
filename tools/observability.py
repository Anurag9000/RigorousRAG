"""Dependency-free request tracing, stage metrics, and cost telemetry."""

from __future__ import annotations

import contextlib
import contextvars
import json
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional


_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "rigorousrag_trace_id", default=None
)


def current_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def new_trace_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    stage: str
    started_at: float
    duration_ms: float
    success: bool
    input_items: int = 0
    output_items: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    attributes: Mapping[str, str] = field(default_factory=dict)
    error_type: Optional[str] = None


@dataclass(frozen=True)
class MetricsSnapshot:
    events: int
    failed_events: int
    total_duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    by_stage: Mapping[str, Mapping[str, float]]


class TraceRecorder:
    """Thread-safe in-memory recorder with JSONL and Prometheus exports."""

    def __init__(self, *, max_events: int = 10000) -> None:
        if isinstance(max_events, bool) or int(max_events) != max_events or max_events <= 0:
            raise ValueError("max_events must be a positive integer.")
        self.max_events = int(max_events)
        self._events: List[TraceEvent] = []
        self._lock = threading.RLock()

    def record(self, event: TraceEvent) -> None:
        if not event.trace_id or not event.stage:
            raise ValueError("trace_id and stage are required.")
        if not math.isfinite(event.duration_ms) or event.duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative.")
        if not math.isfinite(event.estimated_cost_usd) or event.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be finite and non-negative.")
        with self._lock:
            self._events.append(event)
            overflow = len(self._events) - self.max_events
            if overflow > 0:
                del self._events[:overflow]

    def events(self, *, trace_id: Optional[str] = None) -> List[TraceEvent]:
        with self._lock:
            values = list(self._events)
        if trace_id is None:
            return values
        return [event for event in values if event.trace_id == trace_id]

    @contextlib.contextmanager
    def span(
        self,
        stage: str,
        *,
        trace_id: Optional[str] = None,
        input_items: int = 0,
        attributes: Optional[Mapping[str, object]] = None,
    ) -> Iterator["Span"]:
        selected_trace = trace_id or current_trace_id() or new_trace_id()
        token = _current_trace_id.set(selected_trace)
        span = Span(
            recorder=self,
            trace_id=selected_trace,
            stage=stage,
            input_items=max(int(input_items), 0),
            attributes={str(key): str(value) for key, value in (attributes or {}).items()},
        )
        try:
            yield span
        except Exception as exc:
            span.success = False
            span.error_type = type(exc).__name__
            raise
        finally:
            span.finish()
            _current_trace_id.reset(token)

    def snapshot(self) -> MetricsSnapshot:
        values = self.events()
        grouped: Dict[str, Dict[str, float]] = {}
        for event in values:
            stage = grouped.setdefault(
                event.stage,
                {"events": 0.0, "failures": 0.0, "duration_ms": 0.0, "cost_usd": 0.0},
            )
            stage["events"] += 1.0
            stage["failures"] += 0.0 if event.success else 1.0
            stage["duration_ms"] += event.duration_ms
            stage["cost_usd"] += event.estimated_cost_usd
        for stage in grouped.values():
            count = stage["events"] or 1.0
            stage["mean_duration_ms"] = stage["duration_ms"] / count
        return MetricsSnapshot(
            events=len(values),
            failed_events=sum(not event.success for event in values),
            total_duration_ms=sum(event.duration_ms for event in values),
            prompt_tokens=sum(event.prompt_tokens for event in values),
            completion_tokens=sum(event.completion_tokens for event in values),
            estimated_cost_usd=sum(event.estimated_cost_usd for event in values),
            by_stage=grouped,
        )

    def write_jsonl(self, path: str | Path) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = self.events()
        with destination.open("w", encoding="utf-8") as handle:
            for event in values:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return len(values)

    def prometheus_text(self, *, prefix: str = "rigorousrag") -> str:
        snapshot = self.snapshot()
        lines = [
            f"{prefix}_events_total {snapshot.events}",
            f"{prefix}_events_failed_total {snapshot.failed_events}",
            f"{prefix}_duration_milliseconds_total {snapshot.total_duration_ms:.6f}",
            f"{prefix}_prompt_tokens_total {snapshot.prompt_tokens}",
            f"{prefix}_completion_tokens_total {snapshot.completion_tokens}",
            f"{prefix}_estimated_cost_usd_total {snapshot.estimated_cost_usd:.12f}",
        ]
        for stage_name, values in sorted(snapshot.by_stage.items()):
            label = stage_name.replace("\\", "_").replace('"', "_")
            lines.append(
                f'{prefix}_stage_events_total{{stage="{label}"}} {int(values["events"])}'
            )
            lines.append(
                f'{prefix}_stage_duration_milliseconds_total{{stage="{label}"}} '
                f'{values["duration_ms"]:.6f}'
            )
        return "\n".join(lines) + "\n"


class Span:
    def __init__(
        self,
        *,
        recorder: TraceRecorder,
        trace_id: str,
        stage: str,
        input_items: int,
        attributes: Mapping[str, str],
    ) -> None:
        self.recorder = recorder
        self.trace_id = trace_id
        self.stage = str(stage)
        self.input_items = input_items
        self.output_items = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = 0.0
        self.attributes = dict(attributes)
        self.success = True
        self.error_type: Optional[str] = None
        self._started_wall = time.time()
        self._started = time.perf_counter()
        self._finished = False

    def add_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        for value in (prompt_tokens, completion_tokens):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("token counts must be non-negative integers.")
        if not math.isfinite(float(estimated_cost_usd)) or estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be finite and non-negative.")
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.estimated_cost_usd += float(estimated_cost_usd)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        duration_ms = (time.perf_counter() - self._started) * 1000.0
        self.recorder.record(
            TraceEvent(
                trace_id=self.trace_id,
                stage=self.stage,
                started_at=self._started_wall,
                duration_ms=duration_ms,
                success=self.success,
                input_items=self.input_items,
                output_items=max(int(self.output_items), 0),
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                estimated_cost_usd=self.estimated_cost_usd,
                attributes=dict(self.attributes),
                error_type=self.error_type,
            )
        )


DEFAULT_RECORDER = TraceRecorder()
