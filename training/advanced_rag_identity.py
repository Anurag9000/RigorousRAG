"""Content identities for every non-weight input that can change advanced RAG training."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def provider_identity_sha256(provider: Any | None, *, label: str) -> str | None:
    """Return the strongest reproducible identity exposed by a provider.

    Mutable supervision caches may expose an explicit ``seal()`` transition. In that case the
    identity operation is also the read-authority boundary: exact current contents are frozen
    before their content contract is captured, so later reads are checked against the same
    snapshot that entered the run/checkpoint identity. Generic providers without a seal remain
    unchanged.
    """
    if provider is None:
        return None
    seal = getattr(provider, "seal", None)
    if callable(seal):
        sealed = seal()
        if not isinstance(sealed, str):
            raise ValueError(f"{label} seal() must return a SHA-256 content contract")
        return _sha(sealed, label)
    identity = getattr(provider, "identity", None)
    for candidate in (
        getattr(provider, "contract_sha256", None),
        getattr(provider, "binding_sha256", None),
        getattr(identity, "digest", None),
    ):
        if isinstance(candidate, str):
            return _sha(candidate, label)
    raise ValueError(f"{label} provider must expose a content identity")


def dataclass_sha256(value: Any, *, label: str) -> str:
    if not is_dataclass(value):
        raise ValueError(f"{label} must be a dataclass")
    return _digest({"schema": f"rigorousrag-{label}/v1", "value": asdict(value)})


def trainability_sha256(trainability: Mapping[str, Any]) -> str:
    normalized = {}
    for stage, policy in sorted(trainability.items()):
        prefixes = getattr(policy, "trainable_prefixes", None)
        if prefixes is None:
            raise ValueError("trainability policy must expose trainable_prefixes")
        normalized[str(stage)] = list(prefixes)
    return _digest({"schema": "rigorousrag-stage-trainability/v1", "stages": normalized})


@dataclass(frozen=True)
class AdvancedTrainingInputIdentity:
    kind: str
    plan_sha256: str
    training_split_sha256: str
    validation_split_sha256: str
    tokenizer_sha256: str
    execution_config_sha256: str
    collator_config_sha256: str
    trainability_sha256: str
    teacher_cache_sha256: str | None = None
    reference_cache_sha256: str | None = None
    retriever_supervision_sha256: str | None = None
    hidden_state_cache_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced training input kind")
        for name in (
            "plan_sha256",
            "training_split_sha256",
            "validation_split_sha256",
            "tokenizer_sha256",
            "execution_config_sha256",
            "collator_config_sha256",
            "trainability_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in (
            "teacher_cache_sha256",
            "reference_cache_sha256",
            "retriever_supervision_sha256",
            "hidden_state_cache_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))

    @property
    def input_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-advanced-training-input-identity/v1", **asdict(self)})

    def bound_run_id(self, run_id: str) -> str:
        selected = str(run_id).strip()
        if not selected:
            raise ValueError("run_id is required")
        return f"{selected}:inputs:{self.input_sha256}"


__all__ = [
    "AdvancedTrainingInputIdentity",
    "dataclass_sha256",
    "provider_identity_sha256",
    "trainability_sha256",
]
