"""Reproducible research-capsule manifests.

A capsule records immutable inputs, generations, capabilities, policies, queries/results,
reports and environment fingerprints needed to replay a research run.  Secret values and
raw private evidence are explicitly excluded; external artifacts are referenced by
content digest and owner-scoped identifier.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_MAX_REFS = 100_000
_ALLOWED_KINDS = frozenset({
    "source", "generation", "index", "model", "policy", "query", "result", "report",
    "graph", "dataset", "config", "code", "container", "environment", "other",
})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CapsuleReference:
    ref_id: str
    kind: str
    content_sha256: str
    version: str = ""
    owner_scope_sha256: str = ""
    uri_hint: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_id", _text(self.ref_id, "ref_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported capsule reference kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "version", _text(self.version, "version", 256, allow_empty=True))
        if self.owner_scope_sha256:
            object.__setattr__(self, "owner_scope_sha256", _sha(self.owner_scope_sha256, "owner_scope_sha256"))
        object.__setattr__(self, "uri_hint", _text(self.uri_hint, "uri_hint", 2000, allow_empty=True))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        safe: dict[str, str] = {}
        for key, value in self.metadata.items():
            name = _text(str(key), "metadata key", 100).lower()
            if any(secret in name for secret in ("secret", "password", "token", "api_key", "credential")):
                raise ValueError("capsule metadata may not contain secret-like fields")
            safe[name] = _text(str(value), "metadata value", 1000, allow_empty=True)
        object.__setattr__(self, "metadata", safe)


@dataclass(frozen=True)
class ReplayStep:
    step_id: str
    operation: str
    input_ref_ids: tuple[str, ...]
    output_ref_ids: tuple[str, ...]
    capability_ref_id: str = ""
    policy_ref_id: str = ""
    deterministic: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _text(self.step_id, "step_id", 256))
        object.__setattr__(self, "operation", _text(self.operation, "operation", 256))
        for name in ("input_ref_ids", "output_ref_ids"):
            values = getattr(self, name)
            if len(values) > 10_000:
                raise ValueError(f"{name} exceeds the item limit")
            object.__setattr__(self, name, tuple(dict.fromkeys(_text(item, name, 256) for item in values)))
        object.__setattr__(self, "capability_ref_id", _text(self.capability_ref_id, "capability_ref_id", 256, allow_empty=True))
        object.__setattr__(self, "policy_ref_id", _text(self.policy_ref_id, "policy_ref_id", 256, allow_empty=True))
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class ResearchCapsule:
    capsule_id: str
    project_id: str
    run_id: str
    code_revision: str
    references: tuple[CapsuleReference, ...]
    replay_steps: tuple[ReplayStep, ...]
    created_at: float = field(default_factory=time.time)
    schema_version: str = "1.0.0"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("capsule_id", "project_id", "run_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        object.__setattr__(self, "code_revision", _text(self.code_revision, "code_revision", 256))
        if len(self.references) > _MAX_REFS or len(self.replay_steps) > _MAX_REFS:
            raise ValueError("research capsule exceeds its item limits")
        reference_ids = {item.ref_id for item in self.references}
        if len(reference_ids) != len(self.references):
            raise ValueError("capsule reference IDs must be unique")
        step_ids = {item.step_id for item in self.replay_steps}
        if len(step_ids) != len(self.replay_steps):
            raise ValueError("replay step IDs must be unique")
        produced: set[str] = set()
        for step in self.replay_steps:
            referenced = set(step.input_ref_ids) | set(step.output_ref_ids)
            for optional in (step.capability_ref_id, step.policy_ref_id):
                if optional:
                    referenced.add(optional)
            if not referenced.issubset(reference_ids):
                raise ValueError("replay step references an unknown capsule artifact")
            duplicate_outputs = produced & set(step.output_ref_ids)
            if duplicate_outputs:
                raise ValueError("multiple replay steps claim the same output reference")
            produced.update(step.output_ref_ids)
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version", 32))
        if len(self.notes) > 1000:
            raise ValueError("notes exceed the item limit")
        object.__setattr__(self, "notes", tuple(_text(item, "note", 5000) for item in self.notes))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    def replayability(self) -> Mapping[str, Any]:
        deterministic = sum(1 for step in self.replay_steps if step.deterministic)
        seeded = sum(1 for step in self.replay_steps if step.seed is not None)
        unresolved = [ref.ref_id for ref in self.references if not ref.content_sha256]
        return {
            "steps": len(self.replay_steps),
            "deterministic_steps": deterministic,
            "seeded_steps": seeded,
            "fully_content_addressed": not unresolved,
            "unresolved_references": unresolved,
            "note": "Content addressing enables replay checks but cannot guarantee deterministic external providers.",
        }


def capsule_manifest_bytes(capsule: ResearchCapsule) -> bytes:
    if not isinstance(capsule, ResearchCapsule):
        raise TypeError("capsule must be ResearchCapsule")
    payload = asdict(capsule)
    payload["fingerprint"] = capsule.fingerprint
    return _canonical(payload)


__all__ = ["CapsuleReference", "ReplayStep", "ResearchCapsule", "capsule_manifest_bytes"]
