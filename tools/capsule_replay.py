"""Verified replay orchestration for content-addressed research capsules.

Verification and execution are intentionally separate. A digest authority can prove that
an immutable reference is still available at the expected identity without materializing
private bytes. Actual replay still requires a resolver capable of returning the input
bytes needed by the replay operation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol

from tools.research_capsule import CapsuleReference, ReplayStep, ResearchCapsule


class CapsuleResolver(Protocol):
    def resolve(self, reference: CapsuleReference) -> bytes: ...


class CapsuleDigestAuthority(Protocol):
    def digest(self, reference: CapsuleReference) -> str | None: ...


class ReplayOperation(Protocol):
    def run(self, step: ReplayStep, inputs: Mapping[str, bytes]) -> Mapping[str, bytes]: ...


@dataclass(frozen=True)
class CapsuleReferenceVerification:
    ref_id: str
    expected_sha256: str
    actual_sha256: str
    status: str


@dataclass(frozen=True)
class CapsuleVerificationReceipt:
    capsule_id: str
    capsule_fingerprint: str
    references: tuple[CapsuleReferenceVerification, ...]
    verified: bool
    unavailable_ref_ids: tuple[str, ...]
    mismatched_ref_ids: tuple[str, ...]


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


def _valid_sha(value: object) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def verify_capsule(
    capsule: ResearchCapsule,
    *,
    authority: CapsuleDigestAuthority | CapsuleResolver,
) -> CapsuleVerificationReceipt:
    """Verify every capsule reference against an external durable authority.

    If ``authority.digest`` exists, no private material needs to be fetched. Otherwise the
    legacy byte resolver is used and the digest is computed locally. Returning the digest
    copied from the reference itself is not sufficient; concrete authorities are expected
    to derive it from the durable object they govern.
    """

    if not isinstance(capsule, ResearchCapsule):
        raise TypeError("capsule must be ResearchCapsule")
    digest_fn = getattr(authority, "digest", None)
    resolve_fn = getattr(authority, "resolve", None)
    if not callable(digest_fn) and not callable(resolve_fn):
        raise TypeError("authority must implement digest or resolve")

    receipts: list[CapsuleReferenceVerification] = []
    unavailable: list[str] = []
    mismatched: list[str] = []
    for reference in capsule.references:
        actual = ""
        try:
            if callable(digest_fn):
                actual = _valid_sha(digest_fn(reference))
            else:
                payload = resolve_fn(reference)
                if not isinstance(payload, bytes):
                    raise TypeError("capsule resolver must return bytes")
                actual = _sha(payload)
        except (KeyError, FileNotFoundError, LookupError):
            actual = ""
        if not actual:
            status = "unavailable"
            unavailable.append(reference.ref_id)
        elif actual != reference.content_sha256:
            status = "mismatch"
            mismatched.append(reference.ref_id)
        else:
            status = "matched"
        receipts.append(
            CapsuleReferenceVerification(
                ref_id=reference.ref_id,
                expected_sha256=reference.content_sha256,
                actual_sha256=actual,
                status=status,
            )
        )
    return CapsuleVerificationReceipt(
        capsule_id=capsule.capsule_id,
        capsule_fingerprint=capsule.fingerprint,
        references=tuple(receipts),
        verified=not unavailable and not mismatched,
        unavailable_ref_ids=tuple(unavailable),
        mismatched_ref_ids=tuple(mismatched),
    )


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
        receipts.append(
            ReplayStepReceipt(
                step.step_id,
                "matched" if all(output_digests[r] == refs[r].content_sha256 for r in output_digests) else "mismatch",
                output_digests,
                step.deterministic,
            )
        )
        if not step.deterministic:
            reasons.append(f"nondeterministic_step:{step.step_id}")
    return CapsuleReplayReceipt(
        capsule.capsule_id,
        capsule.fingerprint,
        tuple(receipts),
        not any(reason.startswith("output_mismatch:") for reason in reasons),
        tuple(reasons),
    )


__all__ = [
    "CapsuleDigestAuthority",
    "CapsuleReferenceVerification",
    "CapsuleReplayReceipt",
    "CapsuleResolver",
    "CapsuleVerificationReceipt",
    "ReplayOperation",
    "ReplayStepReceipt",
    "replay_capsule",
    "verify_capsule",
]
