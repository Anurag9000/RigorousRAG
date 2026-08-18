from __future__ import annotations

import hashlib

import pytest

from security.artifact_attestation import (
    ArtifactAdmissionPolicy,
    ArtifactAttestationStatement,
    ArtifactSubject,
    AttestationPredicate,
    VerifiedAttestation,
    decide_artifact_admission,
)
from security.data_release import DataReleasePolicy, SensitiveCategoryRule, release_text
from security.model_output_authority import (
    ClosedOutputSchema,
    GroundedAnswerPolicy,
    OutputFieldSpec,
    parse_strict_model_json,
    validate_grounded_model_output,
    validate_model_output,
)
from security.operator_authorization import (
    OperatorAuthorizationPolicy,
    OperatorAuthorizationRequest,
    PrincipalRoleBinding,
    RolePermission,
    VerifiedPrincipalAssertion,
    assert_operator_authorization,
    authorize_operator_action,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assertion(*, methods=("password", "webauthn"), valid_until=100.0):
    return VerifiedPrincipalAssertion(
        principal_sha256=_sha("principal"),
        issuer_id="idp",
        verification_provider_id="verified-idp-adapter",
        assertion_sha256=_sha("assertion"),
        authenticated_at=10.0,
        valid_until=valid_until,
        methods=tuple(methods),
    )


def _operator_policy() -> OperatorAuthorizationPolicy:
    return OperatorAuthorizationPolicy(
        "ops-v1",
        bindings=(PrincipalRoleBinding(_sha("principal"), "runtime-admin", ("owner-a",), ("science",)),),
        permissions=(RolePermission("runtime-admin", ("runtime.promote",), ("runtime_stack",), require_mfa=True, require_reason_digest=True),),
        decision_ttl_seconds=30.0,
    )


def test_operator_authorization_requires_scope_mfa_and_reason_digest() -> None:
    request = OperatorAuthorizationRequest("owner-a", "science", "runtime.promote", "runtime_stack", _sha("stack"))
    denied = authorize_operator_action(_assertion(methods=("password",)), request, policy=_operator_policy(), now=20.0)
    assert denied.authorized is False
    assert "mfa_required" in denied.reason_codes

    reasoned = OperatorAuthorizationRequest("owner-a", "science", "runtime.promote", "runtime_stack", _sha("stack"), _sha("change reason"))
    allowed = authorize_operator_action(_assertion(), reasoned, policy=_operator_policy(), now=20.0)
    assert allowed.authorized is True
    assert_operator_authorization(allowed, reasoned, policy_sha256=_operator_policy().policy_sha256, now=21.0)

    wrong_scope = OperatorAuthorizationRequest("owner-b", "science", "runtime.promote", "runtime_stack", _sha("stack"), _sha("reason"))
    denied_scope = authorize_operator_action(_assertion(), wrong_scope, policy=_operator_policy(), now=20.0)
    assert denied_scope.authorized is False
    assert "principal_has_no_role_binding_for_scope" in denied_scope.reason_codes


def test_operator_authorization_receipt_expires() -> None:
    request = OperatorAuthorizationRequest("owner-a", "science", "runtime.promote", "runtime_stack", _sha("stack"), _sha("reason"))
    decision = authorize_operator_action(_assertion(valid_until=25.0), request, policy=_operator_policy(), now=20.0)
    assert decision.authorized
    with pytest.raises(RuntimeError, match="stale or not yet valid"):
        assert_operator_authorization(decision, request, policy_sha256=_operator_policy().policy_sha256, now=26.0)


def test_native_dlp_redacts_secret_and_email_before_model_input() -> None:
    policy = DataReleasePolicy(
        "model-input",
        "model_input",
        (
            SensitiveCategoryRule("secret", "redact"),
            SensitiveCategoryRule("email", "redact"),
        ),
    )
    decision, released = release_text("email=a@example.com api_key=abcdefgh12345678", policy=policy)
    assert decision.action == "redact"
    assert released is not None
    assert "a@example.com" not in released.text
    assert "abcdefgh12345678" not in released.text
    assert "[REDACTED:email]" in released.text
    assert "[REDACTED:secret]" in released.text


def test_missing_required_external_dlp_attestation_blocks_release() -> None:
    policy = DataReleasePolicy(
        "external-provider",
        "external_provider",
        (SensitiveCategoryRule("secret", "block"),),
        required_detector_ids=("enterprise-dlp",),
    )
    decision, released = release_text("ordinary text", policy=policy)
    assert decision.action == "block"
    assert released is None
    assert "required_detector_attestation_missing" in decision.reason_codes


def test_model_output_parser_rejects_duplicate_keys_and_reserved_authority_fields() -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        parse_strict_model_json('{"answer":"a","answer":"b"}')
    with pytest.raises(ValueError, match="reserved"):
        ClosedOutputSchema("bad", "1", (OutputFieldSpec("tool_calls", "string_list"),))


def test_grounded_output_allows_only_server_owned_citations() -> None:
    schema = ClosedOutputSchema(
        "grounded-answer",
        "1",
        (
            OutputFieldSpec("answer", "string", maximum_length=1000),
            OutputFieldSpec("citation_ids", "string_list", maximum_items=10),
            OutputFieldSpec("abstain", "boolean"),
            OutputFieldSpec("abstention_reason", "string", required=False, maximum_length=500),
        ),
    )
    raw = '{"answer":"supported","citation_ids":["ev-1"],"abstain":false}'
    validated = validate_model_output(raw, schema=schema, model_artifact_sha256=_sha("model"), context_sha256=_sha("context"))
    grounded = validate_grounded_model_output(validated, policy=GroundedAnswerPolicy(), allowed_citation_ids=("ev-1", "ev-2"))
    assert grounded.citation_ids == ("ev-1",)

    bad = validate_model_output(
        '{"answer":"unsupported","citation_ids":["invented"],"abstain":false}',
        schema=schema,
        model_artifact_sha256=_sha("model"),
        context_sha256=_sha("context"),
    )
    with pytest.raises(ValueError, match="server-owned"):
        validate_grounded_model_output(bad, policy=GroundedAnswerPolicy(), allowed_citation_ids=("ev-1",))


def _attestation(*, predicates=("build_provenance", "sbom", "dependency_lock"), produced_at=10.0):
    subject = ArtifactSubject("retriever-image", "container", _sha("artifact"))
    statement = ArtifactAttestationStatement.build(
        subject=subject,
        builder_id="trusted-builder",
        source_revision="a" * 40,
        build_config_sha256=_sha("build-config"),
        dependency_lock_sha256=_sha("lock"),
        sbom_sha256=_sha("sbom"),
        predicates=tuple(AttestationPredicate(kind, _sha(kind)) for kind in predicates),
        produced_at=produced_at,
    )
    verified = VerifiedAttestation(
        statement.statement_sha256,
        subject.subject_sha256,
        "trusted-key",
        "trusted-verifier",
        _sha("verifier-version"),
        _sha("verification-evidence"),
        11.0,
    )
    return statement, verified


def _admission_policy() -> ArtifactAdmissionPolicy:
    return ArtifactAdmissionPolicy(
        "admission-v1",
        ("build_provenance", "sbom", "dependency_lock"),
        ("trusted-builder",),
        ("trusted-key",),
        ("trusted-verifier",),
        maximum_attestation_age_seconds=100.0,
    )


def test_artifact_admission_binds_source_lock_builder_key_and_predicates() -> None:
    statement, verified = _attestation()
    decision = decide_artifact_admission(
        statement,
        verified,
        policy=_admission_policy(),
        now=20.0,
        expected_artifact_sha256=_sha("artifact"),
        expected_source_revision="a" * 40,
        expected_dependency_lock_sha256=_sha("lock"),
    )
    assert decision.admitted is True

    missing, verified_missing = _attestation(predicates=("build_provenance", "dependency_lock"))
    denied = decide_artifact_admission(
        missing,
        verified_missing,
        policy=_admission_policy(),
        now=20.0,
        expected_artifact_sha256=_sha("artifact"),
        expected_source_revision="a" * 40,
        expected_dependency_lock_sha256=_sha("lock"),
    )
    assert denied.admitted is False
    assert "missing_required_predicate:sbom" in denied.reason_codes
