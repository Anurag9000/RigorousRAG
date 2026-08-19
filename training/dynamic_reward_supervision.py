"""Governed realized-retrieval-gain supervision for recorded dynamic RAG trajectories."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.dynamic_record_identity import dynamic_step_pair

_MAX_BYTES = 2 * 1024 * 1024 * 1024
_MAX_RECORDS = 100_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class RealizedRetrievalGainProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def gains(self, steps: Sequence[LegalDynamicRagEpisodeStep]) -> Sequence[float]: ...


class SidecarRealizedRetrievalGainProvider:
    """Strict sidecar of measured post-action utility deltas keyed by exact episode/step pairs."""
    def __init__(self, path: str | Path, *, expected_sha256: str, metric_contract_sha256: str) -> None:
        source = safe_advanced_path(path, label="realized retrieval gain sidecar", must_exist=True, require_file=True)
        if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
            raise ValueError("realized retrieval gain sidecar exceeds byte safety bound")
        actual = _file_sha(source)
        if actual != _sha(expected_sha256, "expected gain sidecar sha256"):
            raise ValueError("realized retrieval gain sidecar digest mismatch")
        self.metric_contract_sha256 = _sha(metric_contract_sha256, "metric_contract_sha256")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
        except Exception as exc:
            raise ValueError("realized retrieval gain sidecar is not strict JSON") from exc
        if not isinstance(payload, Mapping) or set(payload) != {"schema", "gains"} or payload.get("schema") != "rigorousrag-realized-retrieval-gains/v1":
            raise ValueError("unsupported realized retrieval gain sidecar schema")
        raw = payload.get("gains")
        if not isinstance(raw, list) or len(raw) > _MAX_RECORDS:
            raise ValueError("realized retrieval gains must be a bounded array")
        gains: dict[tuple[str, str], float] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping) or set(item) != {"episode_id", "step_id", "gain"}:
                raise ValueError(f"realized retrieval gain record {index} is malformed")
            key = dynamic_step_pair(item["episode_id"], item["step_id"])
            if key in gains:
                raise ValueError(f"duplicate realized retrieval gain identity: {key!r}")
            gains[key] = _finite(item["gain"], f"gain[{key!r}]")
        self.path = str(source)
        self.content_sha256 = actual
        self._gains = gains

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-realized-retrieval-gain-provider/v1",
            "content_sha256": self.content_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "semantics": "measured_post_action_utility_delta_keyed_by_exact_episode_step_pair",
        })

    def gains(self, steps: Sequence[LegalDynamicRagEpisodeStep]) -> Sequence[float]:
        result = []
        for step in steps:
            key = dynamic_step_pair(step.episode_id, step.step_id)
            if key not in self._gains:
                raise ValueError(f"realized retrieval gain sidecar lacks step {key!r}")
            result.append(self._gains[key])
        return tuple(result)


@dataclass(frozen=True)
class RealizedRetrievalGainReceipt:
    provider_sha256: str
    record_count: int
    input_records_sha256: str
    output_records_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("provider_sha256", "input_records_sha256", "output_records_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self._payload()) != self.receipt_sha256:
            raise ValueError("realized retrieval gain receipt digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-realized-retrieval-gain-receipt/v1",
            "provider_sha256": self.provider_sha256,
            "record_count": self.record_count,
            "input_records_sha256": self.input_records_sha256,
            "output_records_sha256": self.output_records_sha256,
        }


def _record_identity(step: LegalDynamicRagEpisodeStep, *, include_gain: bool) -> Mapping[str, Any]:
    value = {
        "episode_id": step.episode_id,
        "step_id": step.step_id,
        "context_sha256": hashlib.sha256(step.context.encode("utf-8")).hexdigest(),
        "action": step.action.value,
        "valid_actions": [action.value for action in step.valid_actions],
    }
    if include_gain:
        value["realized_retrieval_gain"] = step.realized_retrieval_gain
    return value


def apply_realized_retrieval_gains(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    provider: RealizedRetrievalGainProvider,
) -> tuple[tuple[LegalDynamicRagEpisodeStep, ...], RealizedRetrievalGainReceipt]:
    if not steps or any(not isinstance(step, LegalDynamicRagEpisodeStep) for step in steps):
        raise ValueError("gain application requires LegalDynamicRagEpisodeStep values")
    provider_sha = _sha(getattr(provider, "contract_sha256", None), "gain provider contract_sha256")
    gains = tuple(_finite(value, "realized retrieval gain") for value in provider.gains(steps))
    if len(gains) != len(steps):
        raise ValueError("gain provider returned the wrong number of values")
    input_sha = _digest([_record_identity(step, include_gain=False) for step in steps])
    updated = []
    for step, gain in zip(steps, gains):
        metadata = dict(step.metadata); metadata["realized_retrieval_gain_provider_sha256"] = provider_sha
        updated.append(replace(step, realized_retrieval_gain=gain, metadata=metadata))
    output_sha = _digest([_record_identity(step, include_gain=True) for step in updated])
    unsigned = {
        "schema": "rigorousrag-realized-retrieval-gain-receipt/v1",
        "provider_sha256": provider_sha,
        "record_count": len(updated),
        "input_records_sha256": input_sha,
        "output_records_sha256": output_sha,
    }
    receipt = RealizedRetrievalGainReceipt(provider_sha, len(updated), input_sha, output_sha, _digest(unsigned))
    return tuple(updated), receipt


__all__ = [
    "RealizedRetrievalGainProvider",
    "RealizedRetrievalGainReceipt",
    "SidecarRealizedRetrievalGainProvider",
    "apply_realized_retrieval_gains",
]
