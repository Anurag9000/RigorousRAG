"""Compatibility preparation of recorded dynamic-RAG decisions.

The authoritative final-training workflow uses the two-phase manifest-bound pipeline in
``dynamic_manifest_bound_hidden_cache`` so hidden-cache identity binds the final dataset
manifest. This compatibility helper remains useful for non-final experiments and legacy
callers; it now shares the same collision-resistant step keys, exact-pair sidecar identities
and read-only sealing semantics. No model executes on import.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_data import TextSpan
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.dynamic_record_identity import dynamic_hidden_cache_key, dynamic_step_pair

_MAX_ANNOTATIONS = 100_000_000
_MAX_SIDECAR_BYTES = 2 * 1024 * 1024 * 1024


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("dynamic trajectory preparation requires optional PyTorch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
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


class BoundGeneratorHiddenStateProvider(Protocol):
    generator_sha256: str
    tokenizer_sha256: str
    @property
    def contract_sha256(self) -> str: ...
    def encode(self, texts: list[str]) -> Mapping[str, Any]: ...


class InformationNeedAnnotationProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def spans(self, step: LegalDynamicRagEpisodeStep) -> Sequence[TextSpan]: ...


class SidecarInformationNeedAnnotationProvider:
    """Strict immutable annotation sidecar keyed by exact episode/step tuples.

    An entry with an empty ``spans`` list is an explicit negative label, distinct from an
    absent entry. The external JSON keeps episode_id and step_id as separate fields; internal
    lookup therefore never relies on delimiter concatenation.
    """
    def __init__(self, path: str | Path, *, expected_sha256: str) -> None:
        source = safe_advanced_path(path, label="information-need annotation sidecar", must_exist=True, require_file=True)
        if source.stat().st_size <= 0 or source.stat().st_size > _MAX_SIDECAR_BYTES:
            raise ValueError("information-need annotation sidecar exceeds byte safety bound")
        actual = _file_sha(source)
        expected = _sha(expected_sha256, "expected annotation sidecar sha256")
        if actual != expected:
            raise ValueError("information-need annotation sidecar digest mismatch")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
        except Exception as exc:
            raise ValueError("information-need annotation sidecar is not strict JSON") from exc
        if not isinstance(payload, Mapping) or set(payload) != {"schema", "annotations"} or payload.get("schema") != "rigorousrag-information-need-annotations/v1":
            raise ValueError("unsupported information-need annotation sidecar schema")
        raw_annotations = payload.get("annotations")
        if not isinstance(raw_annotations, list) or len(raw_annotations) > _MAX_ANNOTATIONS:
            raise ValueError("information-need annotations must be a bounded array")
        annotations: dict[tuple[str, str], tuple[TextSpan, ...]] = {}
        for index, raw in enumerate(raw_annotations):
            if not isinstance(raw, Mapping) or set(raw) != {"episode_id", "step_id", "spans"} or not isinstance(raw.get("spans"), list):
                raise ValueError(f"information-need annotation {index} is malformed")
            key = dynamic_step_pair(raw["episode_id"], raw["step_id"])
            if key in annotations:
                raise ValueError(f"duplicate information-need annotation identity: {key!r}")
            spans = []
            for span_raw in raw["spans"]:
                if not isinstance(span_raw, Mapping) or set(span_raw) != {"start", "end"}:
                    raise ValueError("information-need span must be a closed {start,end} object")
                spans.append(TextSpan(start=span_raw["start"], end=span_raw["end"]))
            annotations[key] = tuple(spans)
        self.path = str(source)
        self.content_sha256 = actual
        self._annotations = annotations

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-information-need-annotation-provider/v1",
            "content_sha256": self.content_sha256,
            "semantics": "character_spans_over_exact_record_context_with_explicit_empty_negative_exact_pair_keying",
        })

    def spans(self, step: LegalDynamicRagEpisodeStep) -> Sequence[TextSpan]:
        key = dynamic_step_pair(step.episode_id, step.step_id)
        if key not in self._annotations:
            raise ValueError(f"information-need annotation sidecar lacks step {key!r}")
        spans = self._annotations[key]
        if any(span.end > len(step.context) for span in spans):
            raise ValueError(f"information-need annotation for {key!r} lies outside exact step context")
        return spans


@dataclass(frozen=True)
class DynamicTrajectoryPreparationReceipt:
    cache_identity_sha256: str
    hidden_provider_sha256: str
    annotation_provider_sha256: str | None
    record_count: int
    keyset_sha256: str
    entryset_sha256: str
    prepared_records_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("cache_identity_sha256", "hidden_provider_sha256", "keyset_sha256", "entryset_sha256", "prepared_records_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.annotation_provider_sha256 is not None:
            object.__setattr__(self, "annotation_provider_sha256", _sha(self.annotation_provider_sha256, "annotation_provider_sha256"))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self._payload()) != self.receipt_sha256:
            raise ValueError("dynamic trajectory preparation receipt digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-trajectory-preparation-receipt/v1",
            "cache_identity_sha256": self.cache_identity_sha256,
            "hidden_provider_sha256": self.hidden_provider_sha256,
            "annotation_provider_sha256": self.annotation_provider_sha256,
            "record_count": self.record_count,
            "keyset_sha256": self.keyset_sha256,
            "entryset_sha256": self.entryset_sha256,
            "prepared_records_sha256": self.prepared_records_sha256,
        }


def _normalized_hidden_tensors(encoded: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_torch()
    required = {"token_hidden", "state_hidden", "attention_mask"}
    if not isinstance(encoded, Mapping) or not required.issubset(encoded):
        raise ValueError("hidden-state provider must return token_hidden/state_hidden/attention_mask")
    token_hidden = encoded["token_hidden"]
    state_hidden = encoded["state_hidden"]
    attention = encoded["attention_mask"]
    if not all(torch.is_tensor(value) for value in (token_hidden, state_hidden, attention)):
        raise ValueError("hidden-state provider outputs must be tensors")
    token_hidden = token_hidden.detach().cpu(); state_hidden = state_hidden.detach().cpu(); attention = attention.detach().cpu()
    if token_hidden.ndim != 3 or token_hidden.size(0) != 1 or state_hidden.ndim != 2 or state_hidden.size(0) != 1 or attention.ndim != 2 or attention.size(0) != 1:
        raise ValueError("hidden-state provider must return batch-one [1,T,H]/[1,H]/[1,T]")
    if token_hidden.size(1) != attention.size(1) or token_hidden.size(2) != state_hidden.size(1):
        raise ValueError("hidden-state provider tensor shapes are inconsistent")
    visible = attention[0].to(dtype=torch.bool)
    if not bool(visible.any().item()):
        raise ValueError("hidden-state provider returned no visible tokens")
    return {
        "token_hidden": token_hidden[0].contiguous(),
        "state_hidden": state_hidden[0].contiguous(),
        "attention_mask": attention[0].contiguous(),
    }


def _record_digest(step: LegalDynamicRagEpisodeStep) -> str:
    return _digest({
        "episode_id": step.episode_id,
        "step_id": step.step_id,
        "context_sha256": hashlib.sha256(step.context.encode("utf-8")).hexdigest(),
        "features": dict(step.features),
        "action": step.action.value,
        "valid_actions": [action.value for action in step.valid_actions],
        "behavior_action_probability": step.behavior_action_probability,
        "need_spans": [asdict(span) for span in step.need_spans],
        "hidden_state_cache_key": step.hidden_state_cache_key,
        "metadata": dict(step.metadata),
    })


def prepare_dynamic_trajectory_supervision(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    hidden_provider: BoundGeneratorHiddenStateProvider,
    cache: AuthoritativeSafetensorSupervisionCache,
    annotation_provider: InformationNeedAnnotationProvider | None,
    require_need_annotations: bool = True,
) -> tuple[tuple[LegalDynamicRagEpisodeStep, ...], DynamicTrajectoryPreparationReceipt]:
    """Compatibility one-pass preparation; final training should use the two-phase path."""
    if not steps or any(not isinstance(step, LegalDynamicRagEpisodeStep) for step in steps):
        raise ValueError("preparation requires LegalDynamicRagEpisodeStep values")
    if not isinstance(cache, AuthoritativeSafetensorSupervisionCache):
        raise ValueError("cache must be AuthoritativeSafetensorSupervisionCache")
    if cache.is_sealed:
        raise ValueError("trajectory preparation cache must be writable and unsealed")
    provider_sha = _sha(getattr(hidden_provider, "contract_sha256", None), "hidden provider contract_sha256")
    generator_sha = _sha(getattr(hidden_provider, "generator_sha256", None), "hidden provider generator_sha256")
    tokenizer_sha = _sha(getattr(hidden_provider, "tokenizer_sha256", None), "hidden provider tokenizer_sha256")
    identity = cache.identity
    if identity.cache_kind != "generator_hidden_states":
        raise ValueError("trajectory preparation requires cache_kind=generator_hidden_states")
    if identity.producer_sha256 != generator_sha or identity.tokenizer_sha256 != tokenizer_sha:
        raise ValueError("hidden-state cache producer/tokenizer identity differs from provider")
    annotation_sha = None
    if annotation_provider is not None:
        annotation_sha = _sha(getattr(annotation_provider, "contract_sha256", None), "annotation provider contract_sha256")
    elif require_need_annotations:
        raise ValueError("need-selection preparation requires an explicit annotation provider; empty labels must be explicit")

    prepared = []
    keys = []
    entry_digests = []
    seen: set[str] = set()
    for step in steps:
        key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
        if key in seen:
            raise ValueError(f"duplicate hidden-state cache key: {key}")
        seen.add(key)
        spans = tuple(annotation_provider.spans(step)) if annotation_provider is not None else tuple(step.need_spans)
        if any(not isinstance(span, TextSpan) or span.end > len(step.context) for span in spans):
            raise ValueError(f"invalid information-need spans for {dynamic_step_pair(step.episode_id, step.step_id)!r}")
        tensors = _normalized_hidden_tensors(hidden_provider.encode([step.context]))
        entry_sha = cache.put(key, tensors)
        metadata = dict(step.metadata)
        metadata["hidden_provider_sha256"] = provider_sha
        metadata["hidden_cache_identity_sha256"] = identity.digest
        if annotation_sha is not None:
            metadata["need_annotation_provider_sha256"] = annotation_sha
        prepared.append(replace(step, hidden_state_cache_key=key, need_spans=spans, metadata=metadata))
        keys.append(key); entry_digests.append(entry_sha)

    cache.seal()
    records_sha = _digest([_record_digest(step) for step in prepared])
    unsigned = {
        "schema": "rigorousrag-dynamic-trajectory-preparation-receipt/v1",
        "cache_identity_sha256": identity.digest,
        "hidden_provider_sha256": provider_sha,
        "annotation_provider_sha256": annotation_sha,
        "record_count": len(prepared),
        "keyset_sha256": _digest(keys),
        "entryset_sha256": _digest(entry_digests),
        "prepared_records_sha256": records_sha,
    }
    receipt = DynamicTrajectoryPreparationReceipt(
        cache_identity_sha256=identity.digest,
        hidden_provider_sha256=provider_sha,
        annotation_provider_sha256=annotation_sha,
        record_count=len(prepared),
        keyset_sha256=unsigned["keyset_sha256"],
        entryset_sha256=unsigned["entryset_sha256"],
        prepared_records_sha256=records_sha,
        receipt_sha256=_digest(unsigned),
    )
    return tuple(prepared), receipt


__all__ = [
    "BoundGeneratorHiddenStateProvider",
    "DynamicTrajectoryPreparationReceipt",
    "InformationNeedAnnotationProvider",
    "SidecarInformationNeedAnnotationProvider",
    "prepare_dynamic_trajectory_supervision",
]
