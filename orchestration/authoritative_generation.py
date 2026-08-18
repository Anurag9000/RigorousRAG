"""Authoritative source-only serving transaction for grounded RAG generation.

The repository contains strong lower-level primitives for runtime-stack authority,
retrieved-content trust, DLP release and closed-schema output validation.  This module makes
their ordering non-optional for the highest-authority serving path:

1. assert the exact active runtime stack and fencing token;
2. bind the selected generator provider to the promoted generator component;
3. compute retrieved-content trust decisions inside the transaction;
4. re-bind packed evidence to immutable generation/provenance identities;
5. apply content-bound model-input DLP release;
6. render only released system/user messages;
7. invoke an injected generator exactly once;
8. re-assert the runtime fence before publication;
9. validate strict closed-schema JSON and server-owned citation ids;
10. emit a digest-only transaction receipt.

The module performs no model loading, network I/O or durable background work itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from orchestration.runtime_stack_authority import RuntimeComponent, RuntimeStackArtifact
from security.data_release import SensitiveDataScan
from security.model_output_authority import (
    ClosedOutputSchema,
    GroundedAnswerPolicy,
    GroundedModelOutput,
    validate_grounded_model_output,
    validate_model_output,
)
from security.retrieved_content_trust import (
    InjectionSignal,
    RetrievedContentTrustPolicy,
    RetrievedEvidenceMaterialization,
    decide_retrieved_content_trust,
)
from tools.authoritative_generation_release import (
    AuthoritativeGenerationReleaseReceipt,
    AuthoritativeReleasedGeneration,
    model_input_sha256,
    release_authoritative_generation_context,
    render_authoritative_released_messages,
)
from tools.evidence_context_packing import ContextEvidenceCandidate, ContextPackingReceipt
from tools.governed_generation_release import GenerationReleasePolicies
from tools.trusted_generation_context import ChatMessage, TrustedGenerationContext, build_trusted_generation_context

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@runtime_checkable
class RuntimeAuthorityReader(Protocol):
    def assert_runtime_authority(
        self,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        stack_sha256: str,
        fencing_token: int,
    ) -> RuntimeStackArtifact:
        ...


@runtime_checkable
class GroundedGeneratorProvider(Protocol):
    """Injected generator already loaded/configured by deployment code.

    Artifact and contract digests are read both before and after invocation to make provider
    replacement/drift observable at the source authority boundary.
    """

    @property
    def artifact_sha256(self) -> str:
        ...

    @property
    def contract_sha256(self) -> str:
        ...

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        request_sha256: str,
        output_schema_sha256: str,
    ) -> str | bytes:
        ...


@dataclass(frozen=True)
class ServingGenerationContract:
    """Exact generation-side contract promoted as ``RuntimeStackArtifact.generation_contract_sha256``."""

    generator_component_contract_sha256: str
    trust_policy_sha256: str
    release_policies_sha256: str
    output_schema_sha256: str
    grounded_answer_policy_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "generator_component_contract_sha256",
            "trust_policy_sha256",
            "release_policies_sha256",
            "output_schema_sha256",
            "grounded_answer_policy_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-serving-generation-contract/v1", **asdict(self)})

    @classmethod
    def build(
        cls,
        *,
        generator_component_contract_sha256: str,
        trust_policy: RetrievedContentTrustPolicy,
        release_policies: GenerationReleasePolicies,
        output_schema: ClosedOutputSchema,
        grounded_answer_policy: GroundedAnswerPolicy,
    ) -> "ServingGenerationContract":
        if not isinstance(trust_policy, RetrievedContentTrustPolicy):
            raise ValueError("trust_policy must be RetrievedContentTrustPolicy")
        if not isinstance(release_policies, GenerationReleasePolicies):
            raise ValueError("release_policies must be GenerationReleasePolicies")
        if not isinstance(output_schema, ClosedOutputSchema):
            raise ValueError("output_schema must be ClosedOutputSchema")
        if not isinstance(grounded_answer_policy, GroundedAnswerPolicy):
            raise ValueError("grounded_answer_policy must be GroundedAnswerPolicy")
        _assert_grounded_schema(output_schema, grounded_answer_policy)
        return cls(
            generator_component_contract_sha256=generator_component_contract_sha256,
            trust_policy_sha256=trust_policy.policy_sha256,
            release_policies_sha256=release_policies.policies_sha256,
            output_schema_sha256=output_schema.schema_sha256,
            grounded_answer_policy_sha256=grounded_answer_policy.policy_sha256,
        )


@dataclass(frozen=True)
class AuthoritativeGenerationReceipt:
    scope_sha256: str
    stack_sha256: str
    fencing_token: int
    generation_contract_sha256: str
    generator_component_sha256: str
    generator_artifact_sha256: str
    trusted_context_sha256: str
    release_receipt_sha256: str
    model_input_sha256: str | None
    raw_output_sha256: str | None
    validated_output_sha256: str | None
    grounded_output_sha256: str | None
    action: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "scope_sha256",
            "stack_sha256",
            "generation_contract_sha256",
            "generator_component_sha256",
            "generator_artifact_sha256",
            "trusted_context_sha256",
            "release_receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "fencing_token", _positive_int(self.fencing_token, "fencing_token"))
        for name in ("model_input_sha256", "raw_output_sha256", "validated_output_sha256", "grounded_output_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))
        if self.action not in {"blocked", "published"}:
            raise ValueError("action must be blocked or published")
        if self.action == "blocked":
            if any(
                value is not None
                for value in (
                    self.model_input_sha256,
                    self.raw_output_sha256,
                    self.validated_output_sha256,
                    self.grounded_output_sha256,
                )
            ):
                raise ValueError("blocked serving receipt may not contain model/output digests")
        else:
            if any(
                value is None
                for value in (
                    self.model_input_sha256,
                    self.raw_output_sha256,
                    self.validated_output_sha256,
                    self.grounded_output_sha256,
                )
            ):
                raise ValueError("published serving receipt requires complete model/output digests")
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("serving receipt digest mismatch")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-generation-receipt/v1",
            "scope_sha256": self.scope_sha256,
            "stack_sha256": self.stack_sha256,
            "fencing_token": self.fencing_token,
            "generation_contract_sha256": self.generation_contract_sha256,
            "generator_component_sha256": self.generator_component_sha256,
            "generator_artifact_sha256": self.generator_artifact_sha256,
            "trusted_context_sha256": self.trusted_context_sha256,
            "release_receipt_sha256": self.release_receipt_sha256,
            "model_input_sha256": self.model_input_sha256,
            "raw_output_sha256": self.raw_output_sha256,
            "validated_output_sha256": self.validated_output_sha256,
            "grounded_output_sha256": self.grounded_output_sha256,
            "action": self.action,
        }


@dataclass(frozen=True)
class AuthoritativeGenerationResult:
    receipt: AuthoritativeGenerationReceipt
    release_receipt: AuthoritativeGenerationReleaseReceipt
    grounded_output: GroundedModelOutput | None

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, AuthoritativeGenerationReceipt):
            raise ValueError("receipt must be AuthoritativeGenerationReceipt")
        if not isinstance(self.release_receipt, AuthoritativeGenerationReleaseReceipt):
            raise ValueError("release_receipt must be AuthoritativeGenerationReleaseReceipt")
        if self.receipt.release_receipt_sha256 != self.release_receipt.receipt_sha256:
            raise ValueError("serving/release receipt identity mismatch")
        if self.receipt.action == "blocked":
            if self.grounded_output is not None or self.release_receipt.action != "blocked":
                raise ValueError("blocked serving result cannot contain grounded output")
        else:
            if not isinstance(self.grounded_output, GroundedModelOutput) or self.release_receipt.action != "released":
                raise ValueError("published serving result requires released context and grounded output")
            if self.grounded_output.grounded_output_sha256 != self.receipt.grounded_output_sha256:
                raise ValueError("grounded output differs from serving receipt")


def _assert_grounded_schema(schema: ClosedOutputSchema, policy: GroundedAnswerPolicy) -> None:
    fields = {value.name: value for value in schema.fields}
    required = {
        policy.answer_field: "string",
        policy.citation_field: "string_list",
        policy.abstain_field: "boolean",
    }
    for name, kind in required.items():
        spec = fields.get(name)
        if spec is None or spec.field_type != kind or not spec.required:
            raise ValueError(f"output schema must contain required grounded field {name!r} of type {kind}")
    reason = fields.get(policy.abstention_reason_field)
    if reason is not None and reason.field_type != "string":
        raise ValueError("abstention reason field must be string when present")


def _generator_component(stack: RuntimeStackArtifact, component_id: str) -> RuntimeComponent:
    selected_id = _text(component_id, "generator_component_id")
    matches = [
        value for value in stack.components
        if value.kind == "generator" and value.component_id == selected_id
    ]
    if len(matches) != 1:
        raise RuntimeError("authoritative runtime stack must contain exactly one selected generator component")
    return matches[0]


def _provider_digest(provider: GroundedGeneratorProvider, attribute: str) -> str:
    try:
        value = getattr(provider, attribute)
    except Exception as exc:  # provider property failures are authority failures.
        raise RuntimeError(f"generator provider {attribute} is unavailable") from exc
    return _sha(value, f"generator provider {attribute}")


def _assert_provider_binding(
    provider: GroundedGeneratorProvider,
    component: RuntimeComponent,
) -> None:
    if not isinstance(provider, GroundedGeneratorProvider):
        raise ValueError("generator must implement GroundedGeneratorProvider")
    if _provider_digest(provider, "artifact_sha256") != component.artifact_sha256:
        raise RuntimeError("generator provider artifact differs from promoted runtime component")
    if _provider_digest(provider, "contract_sha256") != component.contract_sha256:
        raise RuntimeError("generator provider contract differs from promoted runtime component")


def _scope_sha256(owner_id: str, service_id: str, domain_id: str) -> str:
    return _digest(
        {
            "schema": "rigorousrag-serving-scope/v1",
            "owner_id": _text(owner_id, "owner_id"),
            "service_id": _text(service_id, "service_id"),
            "domain_id": _text(domain_id, "domain_id"),
        }
    )


def _signals_by_identity(
    materializations: Sequence[RetrievedEvidenceMaterialization],
    additional_signals: Mapping[str, Sequence[InjectionSignal]] | None,
) -> Mapping[str, tuple[InjectionSignal, ...]]:
    supplied = {} if additional_signals is None else dict(additional_signals)
    identities = {value.identity.identity_sha256 for value in materializations}
    normalized: dict[str, tuple[InjectionSignal, ...]] = {}
    for key, values in supplied.items():
        selected_key = _sha(key, "additional signal evidence identity")
        if selected_key not in identities:
            raise ValueError("additional trust signals reference evidence outside materializations")
        rows = tuple(values)
        if len(rows) > 100 or any(not isinstance(value, InjectionSignal) for value in rows):
            raise ValueError("additional trust signals must be a bounded InjectionSignal sequence")
        if len({value.signal_sha256 for value in rows}) != len(rows):
            raise ValueError("additional trust signals contain duplicate identities")
        normalized[selected_key] = rows
    return normalized


def _serving_receipt(
    *,
    scope_sha256: str,
    stack: RuntimeStackArtifact,
    fencing_token: int,
    contract: ServingGenerationContract,
    generator_component: RuntimeComponent,
    trusted_context: TrustedGenerationContext,
    release_receipt: AuthoritativeGenerationReleaseReceipt,
    model_input_sha: str | None,
    raw_output_sha: str | None,
    validated_output_sha: str | None,
    grounded_output_sha: str | None,
    action: str,
) -> AuthoritativeGenerationReceipt:
    payload = {
        "schema": "rigorousrag-authoritative-generation-receipt/v1",
        "scope_sha256": scope_sha256,
        "stack_sha256": stack.stack_sha256,
        "fencing_token": fencing_token,
        "generation_contract_sha256": contract.contract_sha256,
        "generator_component_sha256": generator_component.component_sha256,
        "generator_artifact_sha256": generator_component.artifact_sha256,
        "trusted_context_sha256": trusted_context.context_sha256,
        "release_receipt_sha256": release_receipt.receipt_sha256,
        "model_input_sha256": model_input_sha,
        "raw_output_sha256": raw_output_sha,
        "validated_output_sha256": validated_output_sha,
        "grounded_output_sha256": grounded_output_sha,
        "action": action,
    }
    return AuthoritativeGenerationReceipt(**payload, receipt_sha256=_digest(payload))


def serve_grounded_generation(
    *,
    runtime_authority: RuntimeAuthorityReader,
    owner_id: str,
    service_id: str,
    domain_id: str,
    stack_sha256: str,
    fencing_token: int,
    generator_component_id: str,
    generator: GroundedGeneratorProvider,
    serving_contract: ServingGenerationContract,
    packing_receipt: ContextPackingReceipt,
    original_candidates: Sequence[ContextEvidenceCandidate],
    materializations: Sequence[RetrievedEvidenceMaterialization],
    trust_policy: RetrievedContentTrustPolicy,
    release_policies: GenerationReleasePolicies,
    output_schema: ClosedOutputSchema,
    grounded_answer_policy: GroundedAnswerPolicy,
    system_instruction: str,
    user_query: str,
    additional_trust_signals: Mapping[str, Sequence[InjectionSignal]] | None = None,
    system_external_scans: Sequence[SensitiveDataScan] = (),
    query_external_scans: Sequence[SensitiveDataScan] = (),
    evidence_external_scans: Mapping[str, Sequence[SensitiveDataScan]] | None = None,
) -> AuthoritativeGenerationResult:
    """Execute the fail-closed source serving transaction.

    A blocked DLP release returns a digest-only blocked result without invoking the
    generator.  Any stack/fence/provider drift after model invocation raises before output
    validation/publication.
    """

    if not isinstance(runtime_authority, RuntimeAuthorityReader):
        raise ValueError("runtime_authority must implement RuntimeAuthorityReader")
    if not isinstance(serving_contract, ServingGenerationContract):
        raise ValueError("serving_contract must be ServingGenerationContract")
    if not isinstance(packing_receipt, ContextPackingReceipt):
        raise ValueError("packing_receipt must be ContextPackingReceipt")
    if not isinstance(trust_policy, RetrievedContentTrustPolicy):
        raise ValueError("trust_policy must be RetrievedContentTrustPolicy")
    if not isinstance(release_policies, GenerationReleasePolicies):
        raise ValueError("release_policies must be GenerationReleasePolicies")
    if not isinstance(output_schema, ClosedOutputSchema) or not isinstance(grounded_answer_policy, GroundedAnswerPolicy):
        raise ValueError("output schema/grounded policy types are invalid")
    _assert_grounded_schema(output_schema, grounded_answer_policy)

    requested_stack_sha = _sha(stack_sha256, "stack_sha256")
    fence = _positive_int(fencing_token, "fencing_token")
    stack = runtime_authority.assert_runtime_authority(
        owner_id=owner_id,
        service_id=service_id,
        domain_id=domain_id,
        stack_sha256=requested_stack_sha,
        fencing_token=fence,
    )
    if stack.stack_sha256 != requested_stack_sha:
        raise RuntimeError("runtime authority returned a different stack than requested")
    component = _generator_component(stack, generator_component_id)
    _assert_provider_binding(generator, component)

    expected_contract = ServingGenerationContract.build(
        generator_component_contract_sha256=component.contract_sha256,
        trust_policy=trust_policy,
        release_policies=release_policies,
        output_schema=output_schema,
        grounded_answer_policy=grounded_answer_policy,
    )
    if expected_contract != serving_contract:
        raise RuntimeError("serving inputs differ from supplied generation contract")
    if stack.generation_contract_sha256 != serving_contract.contract_sha256:
        raise RuntimeError("serving generation contract is not the promoted runtime contract")

    mats = tuple(materializations)
    if not mats or any(not isinstance(value, RetrievedEvidenceMaterialization) for value in mats):
        raise ValueError("materializations must be non-empty RetrievedEvidenceMaterialization values")
    if len({value.identity.identity_sha256 for value in mats}) != len(mats):
        raise ValueError("materializations contain duplicate evidence identities")
    signals = _signals_by_identity(mats, additional_trust_signals)
    trust_decisions = tuple(
        decide_retrieved_content_trust(
            value,
            policy=trust_policy,
            additional_signals=signals.get(value.identity.identity_sha256, ()),
        )
        for value in mats
    )

    trusted_context = build_trusted_generation_context(
        packing_receipt,
        original_candidates=original_candidates,
        materializations=mats,
        trust_decisions=trust_decisions,
        system_instruction=system_instruction,
        user_query=user_query,
    )
    release_receipt, released = release_authoritative_generation_context(
        trusted_context,
        system_instruction=system_instruction,
        user_query=user_query,
        policies=release_policies,
        system_external_scans=system_external_scans,
        query_external_scans=query_external_scans,
        evidence_external_scans=evidence_external_scans,
    )
    scope = _scope_sha256(owner_id, service_id, domain_id)
    if released is None:
        blocked_receipt = _serving_receipt(
            scope_sha256=scope,
            stack=stack,
            fencing_token=fence,
            contract=serving_contract,
            generator_component=component,
            trusted_context=trusted_context,
            release_receipt=release_receipt,
            model_input_sha=None,
            raw_output_sha=None,
            validated_output_sha=None,
            grounded_output_sha=None,
            action="blocked",
        )
        return AuthoritativeGenerationResult(blocked_receipt, release_receipt, None)

    if not isinstance(released, AuthoritativeReleasedGeneration):
        raise RuntimeError("released generation authority returned an invalid result")
    messages = render_authoritative_released_messages(released)
    input_sha = model_input_sha256(released, messages)
    request_sha = _digest(
        {
            "schema": "rigorousrag-generator-request/v1",
            "scope_sha256": scope,
            "stack_sha256": stack.stack_sha256,
            "fencing_token": fence,
            "generation_contract_sha256": serving_contract.contract_sha256,
            "generator_component_sha256": component.component_sha256,
            "model_input_sha256": input_sha,
        }
    )

    raw_output = generator.generate(
        messages,
        request_sha256=request_sha,
        output_schema_sha256=output_schema.schema_sha256,
    )
    if not isinstance(raw_output, (str, bytes)):
        raise RuntimeError("generator returned an unsupported output type")

    # Fence/provider drift after a long-running model call must fail before publication.
    post_stack = runtime_authority.assert_runtime_authority(
        owner_id=owner_id,
        service_id=service_id,
        domain_id=domain_id,
        stack_sha256=requested_stack_sha,
        fencing_token=fence,
    )
    if post_stack.stack_sha256 != stack.stack_sha256:
        raise RuntimeError("runtime stack changed during generation")
    _assert_provider_binding(generator, component)

    validated = validate_model_output(
        raw_output,
        schema=output_schema,
        model_artifact_sha256=component.artifact_sha256,
        context_sha256=input_sha,
    )
    allowed_citation_ids = tuple(value.evidence_id for value in released.evidence)
    grounded = validate_grounded_model_output(
        validated,
        policy=grounded_answer_policy,
        allowed_citation_ids=allowed_citation_ids,
    )
    published_receipt = _serving_receipt(
        scope_sha256=scope,
        stack=stack,
        fencing_token=fence,
        contract=serving_contract,
        generator_component=component,
        trusted_context=trusted_context,
        release_receipt=release_receipt,
        model_input_sha=input_sha,
        raw_output_sha=validated.raw_output_sha256,
        validated_output_sha=validated.output_sha256,
        grounded_output_sha=grounded.grounded_output_sha256,
        action="published",
    )
    return AuthoritativeGenerationResult(published_receipt, release_receipt, grounded)


__all__ = [
    "AuthoritativeGenerationReceipt",
    "AuthoritativeGenerationResult",
    "GroundedGeneratorProvider",
    "RuntimeAuthorityReader",
    "ServingGenerationContract",
    "serve_grounded_generation",
]
