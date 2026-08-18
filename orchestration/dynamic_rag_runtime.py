"""Bounded server-owned runtime for generation-time dynamic retrieval.

A learned policy may score *closed* actions, but it must not gain direct tool authority.
This module therefore owns the iterative control loop and keeps generation, retrieval-query
release, retrieval execution, evidence admission and verification behind explicit injected
protocols.

This is a research/runtime composition primitive, not a replacement for
``authoritative_generation``.  Any final answer intended for production publication must
still pass the repository's authoritative DLP, closed-schema grounding, runtime-fence and
publication paths.  Likewise, retrieval implementations remain responsible for their
normal owner/domain/SSRF/source-trust boundaries.

No model, retriever or network resource is loaded on import.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from training.dynamic_retrieval_policy import (
    DynamicRetrievalAction,
    DynamicRetrievalBudget,
    DynamicRetrievalFeatures,
    DynamicRetrievalRuntimeState,
    allowed_actions,
    transition_runtime_state,
)

_HEX = frozenset("0123456789abcdef")
_MAX_ITERATIONS = 1_000_000
_MAX_TEXT_CHARS = 10_000_000
_MAX_EVIDENCE = 1_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _bounded_text(value: Any, label: str, maximum: int = _MAX_TEXT_CHARS, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if (not allow_empty and not value) or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is empty, oversized or contains NUL")
    return value


@dataclass(frozen=True)
class DynamicGenerationChunk:
    text: str
    generated_tokens: int
    finish_reason: str
    chunk_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _bounded_text(self.text, "generated chunk"))
        object.__setattr__(self, "generated_tokens", _bounded_int(self.generated_tokens, "generated_tokens", 1, 10_000_000))
        object.__setattr__(self, "finish_reason", _identifier(self.finish_reason, "finish_reason", 100))
        provided = _sha(self.chunk_sha256, "chunk_sha256")
        expected = _digest({
            "schema": "rigorousrag-dynamic-generation-chunk/v1",
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "generated_tokens": self.generated_tokens,
            "finish_reason": self.finish_reason,
        })
        if provided != expected:
            raise ValueError("generation chunk digest mismatch")
        object.__setattr__(self, "chunk_sha256", provided)

    @classmethod
    def build(cls, *, text: str, generated_tokens: int, finish_reason: str) -> "DynamicGenerationChunk":
        selected_text = _bounded_text(text, "generated chunk")
        selected_tokens = _bounded_int(generated_tokens, "generated_tokens", 1, 10_000_000)
        selected_reason = _identifier(finish_reason, "finish_reason", 100)
        payload = {
            "schema": "rigorousrag-dynamic-generation-chunk/v1",
            "text_sha256": hashlib.sha256(selected_text.encode("utf-8")).hexdigest(),
            "generated_tokens": selected_tokens,
            "finish_reason": selected_reason,
        }
        return cls(selected_text, selected_tokens, selected_reason, _digest(payload))


@dataclass(frozen=True)
class DynamicRetrievedEvidence:
    evidence_id: str
    evidence_sha256: str
    source_group_sha256: str
    retrieval_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id", 1_000))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        object.__setattr__(self, "source_group_sha256", _sha(self.source_group_sha256, "source_group_sha256"))
        object.__setattr__(self, "retrieval_score", _finite(self.retrieval_score, "retrieval_score"))


@dataclass(frozen=True)
class VerificationObservation:
    support_score: float
    contradiction_score: float
    verifier_sha256: str

    def __post_init__(self) -> None:
        for name in ("support_score", "contradiction_score"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "verifier_sha256", _sha(self.verifier_sha256, "verifier_sha256"))


@dataclass(frozen=True)
class DynamicRuntimeSnapshot:
    request_sha256: str
    generated_text: str
    evidence: tuple[DynamicRetrievedEvidence, ...]
    verification: VerificationObservation | None
    state: DynamicRetrievalRuntimeState
    iteration: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_sha256", _sha(self.request_sha256, "request_sha256"))
        object.__setattr__(self, "generated_text", _bounded_text(self.generated_text, "generated_text", allow_empty=True))
        evidence = tuple(self.evidence)
        if len(evidence) > _MAX_EVIDENCE or any(not isinstance(item, DynamicRetrievedEvidence) for item in evidence):
            raise ValueError("evidence must be a bounded DynamicRetrievedEvidence sequence")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("dynamic runtime evidence ids must be unique")
        object.__setattr__(self, "evidence", evidence)
        if self.verification is not None and not isinstance(self.verification, VerificationObservation):
            raise ValueError("verification must be VerificationObservation or None")
        if not isinstance(self.state, DynamicRetrievalRuntimeState):
            raise ValueError("state must be DynamicRetrievalRuntimeState")
        object.__setattr__(self, "iteration", _bounded_int(self.iteration, "iteration", 0, _MAX_ITERATIONS))

    @property
    def snapshot_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-runtime-snapshot/v1",
            "request_sha256": self.request_sha256,
            "generated_text_sha256": hashlib.sha256(self.generated_text.encode("utf-8")).hexdigest(),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "evidence_sha256": item.evidence_sha256,
                    "source_group_sha256": item.source_group_sha256,
                    "retrieval_score": item.retrieval_score,
                }
                for item in self.evidence
            ],
            "verification": None if self.verification is None else {
                "support_score": self.verification.support_score,
                "contradiction_score": self.verification.contradiction_score,
                "verifier_sha256": self.verification.verifier_sha256,
            },
            "state": {
                "generated_tokens": self.state.generated_tokens,
                "retrievals": self.state.retrievals,
                "verifications": self.state.verifications,
                "consecutive_retrievals": self.state.consecutive_retrievals,
                "terminal": self.state.terminal,
            },
            "iteration": self.iteration,
        })


@runtime_checkable
class DynamicFeatureProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures: ...


@runtime_checkable
class DynamicPolicyProvider(Protocol):
    @property
    def artifact_sha256(self) -> str: ...
    @property
    def contract_sha256(self) -> str: ...
    def action_scores(self, features: DynamicRetrievalFeatures, *, snapshot_sha256: str) -> Mapping[DynamicRetrievalAction, float]: ...


@runtime_checkable
class DynamicGenerationProvider(Protocol):
    @property
    def artifact_sha256(self) -> str: ...
    @property
    def contract_sha256(self) -> str: ...
    def generate_chunk(self, snapshot: DynamicRuntimeSnapshot, *, maximum_tokens: int) -> DynamicGenerationChunk: ...


@runtime_checkable
class InformationNeedQueryProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def build_query(self, snapshot: DynamicRuntimeSnapshot) -> str: ...


@runtime_checkable
class RetrievalQueryRelease(Protocol):
    @property
    def policy_sha256(self) -> str: ...
    def release(self, query: str, *, snapshot_sha256: str) -> str | None: ...


@runtime_checkable
class DynamicRetrievalProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def retrieve(self, released_query: str, *, snapshot_sha256: str) -> Sequence[DynamicRetrievedEvidence]: ...


@runtime_checkable
class DynamicEvidenceAdmission(Protocol):
    @property
    def policy_sha256(self) -> str: ...
    def admit(self, candidates: Sequence[DynamicRetrievedEvidence], *, snapshot_sha256: str) -> Sequence[DynamicRetrievedEvidence]: ...


@runtime_checkable
class DynamicVerificationProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def verify(self, snapshot: DynamicRuntimeSnapshot) -> VerificationObservation: ...


@dataclass(frozen=True)
class DynamicRagRuntimePolicy:
    budget: DynamicRetrievalBudget
    maximum_iterations: int = 256
    maximum_generated_characters: int = 1_000_000
    maximum_evidence_items: int = 256
    stop_on_generator_finish: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.budget, DynamicRetrievalBudget):
            raise ValueError("budget must be DynamicRetrievalBudget")
        object.__setattr__(self, "maximum_iterations", _bounded_int(self.maximum_iterations, "maximum_iterations", 1, _MAX_ITERATIONS))
        object.__setattr__(self, "maximum_generated_characters", _bounded_int(self.maximum_generated_characters, "maximum_generated_characters", 1, _MAX_TEXT_CHARS))
        object.__setattr__(self, "maximum_evidence_items", _bounded_int(self.maximum_evidence_items, "maximum_evidence_items", 1, _MAX_EVIDENCE))
        if not isinstance(self.stop_on_generator_finish, bool):
            raise ValueError("stop_on_generator_finish must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-runtime-policy/v1",
            "budget_sha256": self.budget.budget_sha256,
            "maximum_iterations": self.maximum_iterations,
            "maximum_generated_characters": self.maximum_generated_characters,
            "maximum_evidence_items": self.maximum_evidence_items,
            "stop_on_generator_finish": self.stop_on_generator_finish,
        })


@dataclass(frozen=True)
class DynamicRagRuntimeResult:
    request_sha256: str
    policy_sha256: str
    generated_text: str
    evidence: tuple[DynamicRetrievedEvidence, ...]
    final_state: DynamicRetrievalRuntimeState
    terminal_action: DynamicRetrievalAction
    iterations: int
    trace_sha256s: tuple[str, ...]
    provider_contract_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_sha256", _sha(self.request_sha256, "request_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        object.__setattr__(self, "generated_text", _bounded_text(self.generated_text, "generated_text", allow_empty=True))
        evidence = tuple(self.evidence)
        if len(evidence) > _MAX_EVIDENCE or any(not isinstance(item, DynamicRetrievedEvidence) for item in evidence):
            raise ValueError("result evidence is invalid")
        object.__setattr__(self, "evidence", evidence)
        if not isinstance(self.final_state, DynamicRetrievalRuntimeState) or not self.final_state.terminal:
            raise ValueError("final_state must be terminal DynamicRetrievalRuntimeState")
        if not isinstance(self.terminal_action, DynamicRetrievalAction):
            object.__setattr__(self, "terminal_action", DynamicRetrievalAction(self.terminal_action))
        if self.terminal_action not in {DynamicRetrievalAction.STOP, DynamicRetrievalAction.ABSTAIN}:
            raise ValueError("terminal_action must be stop or abstain")
        object.__setattr__(self, "iterations", _bounded_int(self.iterations, "iterations", 1, _MAX_ITERATIONS))
        trace = tuple(_sha(value, "trace sha256") for value in self.trace_sha256s)
        if len(trace) != self.iterations:
            raise ValueError("trace length must equal iterations")
        object.__setattr__(self, "trace_sha256s", trace)
        object.__setattr__(self, "provider_contract_sha256", _sha(self.provider_contract_sha256, "provider_contract_sha256"))
        provided = _sha(self.result_sha256, "result_sha256")
        if provided != _digest(self._payload()):
            raise ValueError("dynamic RAG runtime result digest mismatch")
        object.__setattr__(self, "result_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-runtime-result/v1",
            "request_sha256": self.request_sha256,
            "policy_sha256": self.policy_sha256,
            "generated_text_sha256": hashlib.sha256(self.generated_text.encode("utf-8")).hexdigest(),
            "evidence_sha256s": [item.evidence_sha256 for item in self.evidence],
            "final_state": {
                "generated_tokens": self.final_state.generated_tokens,
                "retrievals": self.final_state.retrievals,
                "verifications": self.final_state.verifications,
                "consecutive_retrievals": self.final_state.consecutive_retrievals,
                "terminal": self.final_state.terminal,
            },
            "terminal_action": self.terminal_action.value,
            "iterations": self.iterations,
            "trace_sha256s": self.trace_sha256s,
            "provider_contract_sha256": self.provider_contract_sha256,
        }


def _provider_contract_digest(*values: str) -> str:
    return _digest({"schema": "rigorousrag-dynamic-runtime-provider-contract/v1", "component_sha256s": list(values)})


def _choose_action(scores: Mapping[DynamicRetrievalAction, float], allowed: Sequence[DynamicRetrievalAction]) -> DynamicRetrievalAction:
    permitted = tuple(allowed)
    if not permitted:
        raise RuntimeError("dynamic runtime has no permitted action")
    normalized: dict[DynamicRetrievalAction, float] = {}
    for raw_action, raw_score in scores.items():
        action = raw_action if isinstance(raw_action, DynamicRetrievalAction) else DynamicRetrievalAction(raw_action)
        if action in normalized:
            raise ValueError("dynamic policy emitted duplicate action score")
        normalized[action] = _finite(raw_score, "action score")
    if set(normalized) != set(DynamicRetrievalAction):
        raise ValueError("dynamic policy must score the complete closed action vocabulary")
    # Deterministic tie-break by enum value; server selects the action, never the provider.
    return min(permitted, key=lambda action: (-normalized[action], action.value))


def run_dynamic_rag(
    *,
    request_sha256: str,
    runtime_policy: DynamicRagRuntimePolicy,
    feature_provider: DynamicFeatureProvider,
    policy_provider: DynamicPolicyProvider,
    generation_provider: DynamicGenerationProvider,
    query_provider: InformationNeedQueryProvider,
    query_release: RetrievalQueryRelease,
    retrieval_provider: DynamicRetrievalProvider,
    evidence_admission: DynamicEvidenceAdmission,
    verification_provider: DynamicVerificationProvider,
    initial_evidence: Sequence[DynamicRetrievedEvidence] = (),
) -> DynamicRagRuntimeResult:
    """Execute a bounded closed-action dynamic-RAG episode."""

    request = _sha(request_sha256, "request_sha256")
    if not isinstance(runtime_policy, DynamicRagRuntimePolicy):
        raise ValueError("runtime_policy must be DynamicRagRuntimePolicy")
    providers = (
        feature_provider, policy_provider, generation_provider, query_provider,
        query_release, retrieval_provider, evidence_admission, verification_provider,
    )
    if not all(isinstance(provider, Protocol.__class__) is False for provider in ()):  # pragma: no cover - no-op typing guard.
        raise AssertionError("unreachable")
    # Validate and bind all provider identities up front. Runtime-checkable Protocols provide
    # structural validation without importing implementation packages.
    if not isinstance(feature_provider, DynamicFeatureProvider):
        raise ValueError("feature_provider contract is invalid")
    if not isinstance(policy_provider, DynamicPolicyProvider):
        raise ValueError("policy_provider contract is invalid")
    if not isinstance(generation_provider, DynamicGenerationProvider):
        raise ValueError("generation_provider contract is invalid")
    if not isinstance(query_provider, InformationNeedQueryProvider):
        raise ValueError("query_provider contract is invalid")
    if not isinstance(query_release, RetrievalQueryRelease):
        raise ValueError("query_release contract is invalid")
    if not isinstance(retrieval_provider, DynamicRetrievalProvider):
        raise ValueError("retrieval_provider contract is invalid")
    if not isinstance(evidence_admission, DynamicEvidenceAdmission):
        raise ValueError("evidence_admission contract is invalid")
    if not isinstance(verification_provider, DynamicVerificationProvider):
        raise ValueError("verification_provider contract is invalid")

    component_digests = (
        _sha(feature_provider.contract_sha256, "feature contract"),
        _sha(policy_provider.artifact_sha256, "policy artifact"),
        _sha(policy_provider.contract_sha256, "policy contract"),
        _sha(generation_provider.artifact_sha256, "generation artifact"),
        _sha(generation_provider.contract_sha256, "generation contract"),
        _sha(query_provider.contract_sha256, "query-builder contract"),
        _sha(query_release.policy_sha256, "query-release policy"),
        _sha(retrieval_provider.contract_sha256, "retrieval contract"),
        _sha(evidence_admission.policy_sha256, "evidence-admission policy"),
        _sha(verification_provider.contract_sha256, "verification contract"),
    )
    provider_contract = _provider_contract_digest(*component_digests)

    starting = tuple(initial_evidence)
    if len(starting) > runtime_policy.maximum_evidence_items or any(not isinstance(item, DynamicRetrievedEvidence) for item in starting):
        raise ValueError("initial evidence is invalid or exceeds runtime budget")
    evidence_by_id = {item.evidence_id: item for item in starting}
    if len(evidence_by_id) != len(starting):
        raise ValueError("initial evidence ids must be unique")

    state = DynamicRetrievalRuntimeState()
    generated = ""
    verification: VerificationObservation | None = None
    trace: list[str] = []
    terminal_action: DynamicRetrievalAction | None = None

    for iteration in range(runtime_policy.maximum_iterations):
        snapshot = DynamicRuntimeSnapshot(
            request_sha256=request,
            generated_text=generated,
            evidence=tuple(evidence_by_id.values()),
            verification=verification,
            state=state,
            iteration=iteration,
        )
        features = feature_provider.features(snapshot)
        if not isinstance(features, DynamicRetrievalFeatures):
            raise RuntimeError("feature provider returned invalid DynamicRetrievalFeatures")
        permitted = allowed_actions(state, runtime_policy.budget, verification_enabled=True)
        scores = policy_provider.action_scores(features, snapshot_sha256=snapshot.snapshot_sha256)
        if not isinstance(scores, Mapping):
            raise RuntimeError("dynamic policy provider returned invalid action scores")
        action = _choose_action(scores, permitted)
        event_payload: dict[str, Any] = {
            "schema": "rigorousrag-dynamic-runtime-event/v1",
            "iteration": iteration,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "action": action.value,
        }

        if action == DynamicRetrievalAction.CONTINUE:
            remaining = runtime_policy.budget.max_generation_tokens - state.generated_tokens
            if remaining <= 0:
                raise RuntimeError("continue selected with no generation budget")
            chunk = generation_provider.generate_chunk(snapshot, maximum_tokens=remaining)
            if not isinstance(chunk, DynamicGenerationChunk):
                raise RuntimeError("generation provider returned invalid chunk")
            if chunk.generated_tokens > remaining:
                raise RuntimeError("generation provider exceeded server token budget")
            if len(generated) + len(chunk.text) > runtime_policy.maximum_generated_characters:
                raise RuntimeError("generation provider exceeded server character budget")
            generated += chunk.text
            state = transition_runtime_state(state, action, runtime_policy.budget, generated_tokens_delta=chunk.generated_tokens)
            event_payload["chunk_sha256"] = chunk.chunk_sha256
            if runtime_policy.stop_on_generator_finish and chunk.finish_reason.casefold() in {"stop", "eos", "end", "completed"}:
                terminal_action = DynamicRetrievalAction.STOP
                state = DynamicRetrievalRuntimeState(
                    generated_tokens=state.generated_tokens,
                    retrievals=state.retrievals,
                    verifications=state.verifications,
                    consecutive_retrievals=0,
                    terminal=True,
                )
                event_payload["implicit_terminal_action"] = terminal_action.value

        elif action == DynamicRetrievalAction.RETRIEVE:
            raw_query = _bounded_text(query_provider.build_query(snapshot), "information-need query", 100_000)
            released_query = query_release.release(raw_query, snapshot_sha256=snapshot.snapshot_sha256)
            if released_query is None:
                terminal_action = DynamicRetrievalAction.ABSTAIN
                state = transition_runtime_state(state, terminal_action, runtime_policy.budget)
                event_payload["query_release"] = "blocked"
            else:
                released = _bounded_text(released_query, "released retrieval query", 100_000)
                candidates = tuple(retrieval_provider.retrieve(released, snapshot_sha256=snapshot.snapshot_sha256))
                if len(candidates) > runtime_policy.maximum_evidence_items or any(not isinstance(item, DynamicRetrievedEvidence) for item in candidates):
                    raise RuntimeError("retrieval provider returned an invalid/oversized candidate set")
                admitted = tuple(evidence_admission.admit(candidates, snapshot_sha256=snapshot.snapshot_sha256))
                if len(admitted) > runtime_policy.maximum_evidence_items or any(not isinstance(item, DynamicRetrievedEvidence) for item in admitted):
                    raise RuntimeError("evidence admission returned an invalid/oversized evidence set")
                candidate_map = {item.evidence_id: item for item in candidates}
                if len(candidate_map) != len(candidates):
                    raise RuntimeError("retrieval candidate ids must be unique")
                if any(item.evidence_id not in candidate_map or candidate_map[item.evidence_id] != item for item in admitted):
                    raise RuntimeError("evidence admission may only select unchanged retrieval candidates")
                for item in admitted:
                    previous = evidence_by_id.get(item.evidence_id)
                    if previous is not None and previous != item:
                        raise RuntimeError("evidence identity collision changed immutable evidence")
                    evidence_by_id[item.evidence_id] = item
                if len(evidence_by_id) > runtime_policy.maximum_evidence_items:
                    raise RuntimeError("cumulative evidence budget exceeded")
                state = transition_runtime_state(state, action, runtime_policy.budget)
                event_payload["released_query_sha256"] = hashlib.sha256(released.encode("utf-8")).hexdigest()
                event_payload["admitted_evidence_sha256s"] = sorted(item.evidence_sha256 for item in admitted)

        elif action == DynamicRetrievalAction.VERIFY:
            verification = verification_provider.verify(snapshot)
            if not isinstance(verification, VerificationObservation):
                raise RuntimeError("verification provider returned invalid observation")
            state = transition_runtime_state(state, action, runtime_policy.budget)
            event_payload["verification_sha256"] = _digest({
                "support_score": verification.support_score,
                "contradiction_score": verification.contradiction_score,
                "verifier_sha256": verification.verifier_sha256,
            })

        elif action in {DynamicRetrievalAction.ABSTAIN, DynamicRetrievalAction.STOP}:
            state = transition_runtime_state(state, action, runtime_policy.budget)
            terminal_action = action
        else:  # pragma: no cover - enum exhaustiveness.
            raise AssertionError("unknown dynamic retrieval action")

        trace.append(_digest(event_payload))
        if terminal_action is not None:
            break
    else:
        # Iteration budget is an authority boundary, not a hint. Exhaustion terminates by
        # abstention rather than allowing an unbounded or silently truncated episode.
        terminal_action = DynamicRetrievalAction.ABSTAIN
        state = DynamicRetrievalRuntimeState(
            generated_tokens=state.generated_tokens,
            retrievals=state.retrievals,
            verifications=state.verifications,
            consecutive_retrievals=state.consecutive_retrievals,
            terminal=True,
        )
        trace[-1] = _digest({
            "schema": "rigorousrag-dynamic-runtime-event/v1",
            "previous_event_sha256": trace[-1],
            "iteration_budget_exhausted": True,
            "terminal_action": terminal_action.value,
        })

    assert terminal_action is not None
    payload = {
        "schema": "rigorousrag-dynamic-runtime-result/v1",
        "request_sha256": request,
        "policy_sha256": runtime_policy.policy_sha256,
        "generated_text_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
        "evidence_sha256s": [item.evidence_sha256 for item in evidence_by_id.values()],
        "final_state": {
            "generated_tokens": state.generated_tokens,
            "retrievals": state.retrievals,
            "verifications": state.verifications,
            "consecutive_retrievals": state.consecutive_retrievals,
            "terminal": state.terminal,
        },
        "terminal_action": terminal_action.value,
        "iterations": len(trace),
        "trace_sha256s": tuple(trace),
        "provider_contract_sha256": provider_contract,
    }
    return DynamicRagRuntimeResult(
        request_sha256=request,
        policy_sha256=runtime_policy.policy_sha256,
        generated_text=generated,
        evidence=tuple(evidence_by_id.values()),
        final_state=state,
        terminal_action=terminal_action,
        iterations=len(trace),
        trace_sha256s=tuple(trace),
        provider_contract_sha256=provider_contract,
        result_sha256=_digest(payload),
    )


__all__ = [
    "DynamicEvidenceAdmission", "DynamicFeatureProvider", "DynamicGenerationChunk",
    "DynamicGenerationProvider", "DynamicPolicyProvider", "DynamicRagRuntimePolicy",
    "DynamicRagRuntimeResult", "DynamicRetrievedEvidence", "DynamicRetrievalProvider",
    "DynamicRuntimeSnapshot", "DynamicVerificationProvider", "InformationNeedQueryProvider",
    "RetrievalQueryRelease", "VerificationObservation", "run_dynamic_rag",
]
