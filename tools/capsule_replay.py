"""Verified replay orchestration for content-addressed research capsules."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol
from tools.research_capsule import CapsuleReference, ReplayStep, ResearchCapsule

class CapsuleResolver(Protocol):
    def resolve(self, reference: CapsuleReference) -> bytes: ...

class ReplayOperation(Protocol):
    def run(self, step: ReplayStep, inputs: Mapping[str, bytes]) -> Mapping[str, bytes]: ...

@dataclass(frozen=True)
class ReplayStepReceipt:
    step_id: str
    status: str
    output_sha256: Mapping[str, str]
    deterministic: bool

@dataclass(frozen=True)
class CapsuleReplayReceipt:
    capsule_id: str
    capsule_fingerprint: str
    steps: tuple[ReplayStepReceipt, ...]
    reproducible: bool
    reasons: tuple[str, ...]

def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def replay_capsule(capsule: ResearchCapsule, *, resolver: CapsuleResolver, operations: Mapping[str, ReplayOperation]) -> CapsuleReplayReceipt:
    refs = {ref.ref_id: ref for ref in capsule.references}
    materialized: dict[str, bytes] = {}
    reasons: list[str] = []
    for ref in capsule.references:
        # Outputs are resolved lazily only when no replay step produces them.
        if not any(ref.ref_id in step.output_ref_ids for step in capsule.replay_steps):
            payload = resolver.resolve(ref)
            if not isinstance(payload, bytes) or _sha(payload) != ref.content_sha256:
                raise RuntimeError(f"capsule reference identity mismatch: {ref.ref_id}")
            materialized[ref.ref_id] = payload
    receipts: list[ReplayStepReceipt] = []
    for step in capsule.replay_steps:
        operation = operations.get(step.operation)
        if operation is None:
            raise RuntimeError(f"replay operation unavailable: {step.operation}")
        inputs: dict[str, bytes] = {}
        for ref_id in step.input_ref_ids:
            if ref_id not in materialized:
                payload = resolver.resolve(refs[ref_id])
                if not isinstance(payload, bytes) or _sha(payload) != refs[ref_id].content_sha256:
                    raise RuntimeError(f"capsule input identity mismatch: {ref_id}")
                materialized[ref_id] = payload
            inputs[ref_id] = materialized[ref_id]
        output = operation.run(step, inputs)
        if set(output) != set(step.output_ref_ids):
            raise RuntimeError(f"replay step {step.step_id} returned unexpected outputs")
        output_digests: dict[str, str] = {}
        for ref_id, payload in output.items():
            if not isinstance(payload, bytes):
                raise RuntimeError("replay operations must return bytes")
            digest = _sha(payload)
            output_digests[ref_id] = digest
            expected = refs[ref_id].content_sha256
            if digest != expected:
                reasons.append(f"output_mismatch:{step.step_id}:{ref_id}")
            materialized[ref_id] = payload
        receipts.append(ReplayStepReceipt(step.step_id, "matched" if all(output_digests[r] == refs[r].content_sha256 for r in output_digests) else "mismatch", output_digests, step.deterministic))
        if not step.deterministic:
            reasons.append(f"nondeterministic_step:{step.step_id}")
    return CapsuleReplayReceipt(capsule.capsule_id, capsule.fingerprint, tuple(receipts), not any(reason.startswith("output_mismatch:") for reason in reasons), tuple(reasons))

__all__ = ["CapsuleReplayReceipt", "CapsuleResolver", "ReplayOperation", "ReplayStepReceipt", "replay_capsule"]
