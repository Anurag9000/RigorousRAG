"""Two-phase information-need/hidden-state supervision for dynamic RAG.

The final hidden-state cache identity must bind the final training ``DatasetManifest``. That
manifest cannot exist until records already contain deterministic hidden-state cache keys.
This module breaks the cycle without weakening identity checks:

1. ``plan_dynamic_hidden_supervision`` attaches reviewed need spans and collision-resistant
   deterministic cache keys but writes no hidden tensors;
2. trajectories may then receive reward/GAE targets and be published into final splits;
3. ``materialize_manifest_bound_hidden_cache`` executes the admitted generator later and
   writes the exact keys into a cache whose identity is bound to the final manifest SHA, then
   seals the completed cache read-only.

No model executes on import.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_data import TextSpan
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication
from training.dynamic_record_identity import dynamic_hidden_cache_key
from training.dynamic_trajectory_preparation import BoundGeneratorHiddenStateProvider, InformationNeedAnnotationProvider

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _normalized_hidden(encoded: Mapping[str, Any]) -> Mapping[str, Any]:
    if torch is None:
        raise RuntimeError("manifest-bound hidden cache materialization requires optional PyTorch")
    required = {"token_hidden", "state_hidden", "attention_mask"}
    if not isinstance(encoded, Mapping) or not required.issubset(encoded):
        raise ValueError("hidden provider must return token_hidden/state_hidden/attention_mask")
    token_hidden, state_hidden, attention = encoded["token_hidden"], encoded["state_hidden"], encoded["attention_mask"]
    if not all(torch.is_tensor(value) for value in (token_hidden, state_hidden, attention)):
        raise ValueError("hidden provider outputs must be tensors")
    token_hidden, state_hidden, attention = token_hidden.detach().cpu(), state_hidden.detach().cpu(), attention.detach().cpu()
    if token_hidden.ndim != 3 or token_hidden.size(0) != 1 or state_hidden.ndim != 2 or state_hidden.size(0) != 1 or attention.ndim != 2 or attention.size(0) != 1:
        raise ValueError("hidden provider must return batch-one [1,T,H]/[1,H]/[1,T]")
    if token_hidden.size(1) != attention.size(1) or token_hidden.size(2) != state_hidden.size(1):
        raise ValueError("hidden provider tensor shapes are inconsistent")
    if not bool(attention[0].to(dtype=torch.bool).any().item()):
        raise ValueError("hidden provider returned no visible token")
    return {"token_hidden": token_hidden[0].contiguous(), "state_hidden": state_hidden[0].contiguous(), "attention_mask": attention[0].contiguous()}


@dataclass(frozen=True)
class DynamicHiddenSupervisionPlanReceipt:
    hidden_provider_sha256: str
    annotation_provider_sha256: str | None
    record_count: int
    keyset_sha256: str
    planned_records_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("hidden_provider_sha256", "keyset_sha256", "planned_records_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.annotation_provider_sha256 is not None:
            object.__setattr__(self, "annotation_provider_sha256", _sha(self.annotation_provider_sha256, "annotation_provider_sha256"))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("dynamic hidden-supervision plan receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-dynamic-hidden-supervision-plan-receipt/v1", "hidden_provider_sha256": self.hidden_provider_sha256, "annotation_provider_sha256": self.annotation_provider_sha256, "record_count": self.record_count, "keyset_sha256": self.keyset_sha256, "planned_records_sha256": self.planned_records_sha256}


@dataclass(frozen=True)
class ManifestBoundHiddenCacheReceipt:
    dataset_manifest_sha256: str
    cache_identity_sha256: str
    cache_contract_sha256: str
    hidden_provider_sha256: str
    record_count: int
    keyset_sha256: str
    entryset_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "cache_identity_sha256", "cache_contract_sha256", "hidden_provider_sha256", "keyset_sha256", "entryset_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("manifest-bound hidden-cache receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-manifest-bound-hidden-cache-receipt/v1", "dataset_manifest_sha256": self.dataset_manifest_sha256, "cache_identity_sha256": self.cache_identity_sha256, "cache_contract_sha256": self.cache_contract_sha256, "hidden_provider_sha256": self.hidden_provider_sha256, "record_count": self.record_count, "keyset_sha256": self.keyset_sha256, "entryset_sha256": self.entryset_sha256}


def _planned_record_digest(step: LegalDynamicRagEpisodeStep) -> str:
    return _digest({"episode_id": step.episode_id, "step_id": step.step_id, "context_sha256": hashlib.sha256(step.context.encode("utf-8")).hexdigest(), "need_spans": [asdict(span) for span in step.need_spans], "hidden_state_cache_key": step.hidden_state_cache_key, "metadata": dict(step.metadata)})


def plan_dynamic_hidden_supervision(
    steps: Sequence[LegalDynamicRagEpisodeStep], *, hidden_provider: BoundGeneratorHiddenStateProvider, annotation_provider: InformationNeedAnnotationProvider | None, require_need_annotations: bool = True,
) -> tuple[tuple[LegalDynamicRagEpisodeStep, ...], DynamicHiddenSupervisionPlanReceipt]:
    selected = tuple(steps)
    if not selected or any(not isinstance(step, LegalDynamicRagEpisodeStep) for step in selected):
        raise ValueError("steps must contain LegalDynamicRagEpisodeStep values")
    provider_sha = _sha(getattr(hidden_provider, "contract_sha256", None), "hidden provider contract_sha256")
    annotation_sha = None if annotation_provider is None else _sha(getattr(annotation_provider, "contract_sha256", None), "annotation provider contract_sha256")
    if annotation_provider is None and require_need_annotations:
        raise ValueError("need-selection planning requires an explicit annotation provider")
    planned = []
    keys = []
    seen: set[str] = set()
    for step in selected:
        key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
        if key in seen:
            raise ValueError(f"duplicate hidden-state cache key {key}")
        seen.add(key)
        spans = tuple(annotation_provider.spans(step)) if annotation_provider is not None else tuple(step.need_spans)
        if any(not isinstance(span, TextSpan) or span.end > len(step.context) for span in spans):
            raise ValueError(f"invalid information-need spans for {step.episode_id}:{step.step_id}")
        metadata = dict(step.metadata)
        metadata["hidden_provider_sha256"] = provider_sha
        if annotation_sha is not None:
            metadata["need_annotation_provider_sha256"] = annotation_sha
        planned.append(replace(step, hidden_state_cache_key=key, need_spans=spans, metadata=metadata))
        keys.append(key)
    records_sha = _digest([_planned_record_digest(step) for step in planned])
    unsigned = {"schema": "rigorousrag-dynamic-hidden-supervision-plan-receipt/v1", "hidden_provider_sha256": provider_sha, "annotation_provider_sha256": annotation_sha, "record_count": len(planned), "keyset_sha256": _digest(keys), "planned_records_sha256": records_sha}
    return tuple(planned), DynamicHiddenSupervisionPlanReceipt(provider_sha, annotation_sha, len(planned), unsigned["keyset_sha256"], records_sha, _digest(unsigned))


def materialize_manifest_bound_hidden_cache(
    dataset: VerifiedDynamicDatasetPublication, *, hidden_provider: BoundGeneratorHiddenStateProvider, cache: AuthoritativeSafetensorSupervisionCache,
) -> ManifestBoundHiddenCacheReceipt:
    if not isinstance(dataset, VerifiedDynamicDatasetPublication):
        raise ValueError("dataset must be VerifiedDynamicDatasetPublication")
    if not isinstance(cache, AuthoritativeSafetensorSupervisionCache):
        raise ValueError("cache must be AuthoritativeSafetensorSupervisionCache")
    if cache.is_sealed:
        raise ValueError("hidden cache must be writable and unsealed before materialization")
    provider_sha = _sha(getattr(hidden_provider, "contract_sha256", None), "hidden provider contract_sha256")
    generator_sha = _sha(getattr(hidden_provider, "generator_sha256", None), "hidden provider generator_sha256")
    tokenizer_sha = _sha(getattr(hidden_provider, "tokenizer_sha256", None), "hidden provider tokenizer_sha256")
    identity = cache.identity
    if identity.cache_kind != "generator_hidden_states":
        raise ValueError("cache_kind must be generator_hidden_states")
    if identity.dataset_manifest_sha256 != dataset.manifest.manifest_digest:
        raise ValueError("hidden cache identity must bind the final dynamic dataset manifest")
    if identity.producer_sha256 != generator_sha or identity.tokenizer_sha256 != tokenizer_sha:
        raise ValueError("hidden cache producer/tokenizer identity differs from provider")
    keys: list[str] = []
    entries: list[str] = []
    count = 0
    seen: set[str] = set()
    for split in dataset.manifest.splits:
        records = dataset.split(split.name)
        for index in range(len(records)):
            step = records[index]
            key = step.hidden_state_cache_key
            expected_key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
            if key is None:
                raise ValueError(f"published step {step.episode_id}:{step.step_id} lacks hidden_state_cache_key")
            if key != expected_key:
                raise ValueError("published hidden-state key differs from canonical deterministic key")
            if key in seen:
                raise ValueError(f"duplicate hidden-state cache key across final splits: {key}")
            if step.metadata.get("hidden_provider_sha256") != provider_sha:
                raise ValueError("published step hidden-provider identity differs from materializer")
            seen.add(key)
            tensors = _normalized_hidden(hidden_provider.encode([step.context]))
            entries.append(cache.put(key, tensors))
            keys.append(key)
            count += 1
    contract = cache.seal()
    unsigned = {"schema": "rigorousrag-manifest-bound-hidden-cache-receipt/v1", "dataset_manifest_sha256": dataset.manifest.manifest_digest, "cache_identity_sha256": identity.digest, "cache_contract_sha256": contract, "hidden_provider_sha256": provider_sha, "record_count": count, "keyset_sha256": _digest(keys), "entryset_sha256": _digest(entries)}
    return ManifestBoundHiddenCacheReceipt(dataset.manifest.manifest_digest, identity.digest, contract, provider_sha, count, unsigned["keyset_sha256"], unsigned["entryset_sha256"], _digest(unsigned))


__all__ = ["DynamicHiddenSupervisionPlanReceipt", "ManifestBoundHiddenCacheReceipt", "materialize_manifest_bound_hidden_cache", "plan_dynamic_hidden_supervision"]
