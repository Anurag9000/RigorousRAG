"""Strict construction of reproducibility capsules from durable research authorities."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from tools.research_capsule import CapsuleReference, ReplayStep, ResearchCapsule
from tools.research_result_provenance import session_binding
from tools.research_result_store import StoredResearchResult
from tools.research_workspace import ResearchProject, ResearchSession, ResearchTurn

_RUNTIME_KEY = "_rigorousrag_runtime"
_CITATION_DIGEST_KEYS = (
    "content_sha256",
    "evidence_sha256",
    "chunk_sha256",
    "document_sha256",
    "source_sha256",
)


def _sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def _code_revision(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    revision = value.strip().lower()
    if len(revision) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in revision):
        return ""
    return revision


def _turn_for_result(
    session: ResearchSession,
    result: StoredResearchResult,
) -> tuple[int, ResearchTurn] | None:
    matches = [
        (index, turn)
        for index, turn in enumerate(session.turns)
        if turn.result_sha256 == result.result_id
    ]
    if len(matches) != 1:
        return None
    index, turn = matches[0]
    return (index, turn) if turn.query_sha256 == result.query_sha256 else None


def _citation_digest(citation: Any) -> str:
    metadata = getattr(citation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    for key in _CITATION_DIGEST_KEYS:
        digest = _sha(metadata.get(key))
        if digest:
            return digest
    return ""


@dataclass(frozen=True)
class CapsuleBuildAssessment:
    manifest_ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    bindings: Mapping[str, Any]


@dataclass(frozen=True)
class CapsuleBuildContext:
    project: ResearchProject
    session: ResearchSession
    result: StoredResearchResult
    code_revision: str


def assess_capsule(context: CapsuleBuildContext) -> CapsuleBuildAssessment:
    if not isinstance(context.project, ResearchProject):
        raise TypeError("project must be ResearchProject")
    if not isinstance(context.session, ResearchSession):
        raise TypeError("session must be ResearchSession")
    if not isinstance(context.result, StoredResearchResult):
        raise TypeError("result must be StoredResearchResult")

    project = context.project
    session = context.session
    result = context.result
    blockers: list[str] = []
    warnings: list[str] = []

    if project.owner_id != session.owner_id or session.project_id != project.project_id:
        blockers.append("project_session_scope_mismatch")
    match = _turn_for_result(session, result)
    turn_index = match[0] if match is not None else -1
    turn = match[1] if match is not None else None
    if turn is None:
        blockers.append("result_not_uniquely_bound_to_session_turn")

    result_session = session_binding(result.metadata)
    execution_fingerprint = ""
    if result_session is None:
        blockers.append("result_session_provenance_missing")
    else:
        execution_fingerprint = result_session["session_fingerprint_before"]
        if result_session["project_id"] != project.project_id:
            blockers.append("result_project_provenance_mismatch")
        if result_session["session_id"] != session.session_id:
            blockers.append("result_session_provenance_mismatch")
        if turn is not None:
            prefix = ResearchSession(
                owner_id=session.owner_id,
                project_id=session.project_id,
                session_id=session.session_id,
                turns=session.turns[:turn_index],
                created_at=session.created_at,
                closed_at=None,
            )
            if execution_fingerprint != prefix.fingerprint:
                blockers.append("result_session_snapshot_fingerprint_mismatch")
            if result_session["plan_sha256"] != _sha(turn.plan_sha256):
                blockers.append("result_turn_plan_fingerprint_mismatch")
            if result_session["policy_sha256"] != _sha(turn.policy_sha256):
                blockers.append("result_turn_policy_fingerprint_mismatch")

    revision = _code_revision(context.code_revision)
    if not revision:
        blockers.append("exact_code_revision_unavailable")

    runtime = result.metadata.get(_RUNTIME_KEY)
    if not isinstance(runtime, Mapping):
        runtime = {}
        blockers.append("server_runtime_binding_missing")
    runtime_config = _sha(runtime.get("runtime_config_sha256"))
    capability_registry = _sha(runtime.get("capability_registry_sha256"))
    model_identifier_sha = _sha(runtime.get("model_identifier_sha256"))
    if not runtime_config:
        blockers.append("runtime_config_fingerprint_missing")
    if not capability_registry:
        blockers.append("capability_registry_fingerprint_missing")
    if not model_identifier_sha:
        blockers.append("model_identifier_fingerprint_missing")

    selected = runtime.get("selected_capabilities") if isinstance(runtime, Mapping) else None
    if not isinstance(selected, Mapping):
        selected = {}
        blockers.append("selected_capability_bindings_missing")
    capability_bindings: dict[str, Mapping[str, str]] = {}
    for role, raw in sorted(selected.items(), key=lambda item: str(item[0])):
        if not isinstance(raw, Mapping):
            blockers.append(f"capability_binding_invalid:{role}")
            continue
        fingerprint = _sha(raw.get("fingerprint"))
        capability_id = str(raw.get("capability_id") or "").strip()
        version = str(raw.get("version") or "").strip()
        if not fingerprint or not capability_id:
            blockers.append(f"capability_binding_incomplete:{role}")
            continue
        capability_bindings[str(role)] = {
            "capability_id": capability_id,
            "version": version,
            "fingerprint": fingerprint,
        }

    policy_sha = _sha(turn.policy_sha256) if turn is not None else ""
    if not policy_sha:
        blockers.append("session_policy_fingerprint_missing")
    runtime_policy = capability_bindings.get("policy", {})
    runtime_policy_sha = _sha(runtime_policy.get("fingerprint")) if runtime_policy else ""
    if runtime_policy_sha and policy_sha and runtime_policy_sha != policy_sha:
        blockers.append("session_runtime_policy_fingerprint_mismatch")

    citation_digests: list[str] = []
    for index, citation in enumerate(result.citations):
        digest = _citation_digest(citation)
        citation_digests.append(digest)
        if not digest:
            blockers.append(f"citation_content_fingerprint_missing:{index}")

    if result.model:
        warnings.append(
            "external_model_identifier_is_bound_but_model_weights_are_not_locally_content_addressed"
        )
    warnings.append(
        "external_provider_execution_may_remain_nondeterministic_even_when_all_manifest_bindings_match"
    )

    bindings: dict[str, Any] = {
        "project_fingerprint": project.fingerprint,
        # A replayable session identity is the immutable pre-execution snapshot. The live
        # session may continue accumulating turns or later be closed without invalidating
        # the historical run that this capsule describes.
        "session_fingerprint": execution_fingerprint,
        "session_current_fingerprint": session.fingerprint,
        "session_execution_fingerprint": execution_fingerprint,
        "result_id": result.result_id,
        "query_sha256": result.query_sha256,
        "policy_sha256": policy_sha,
        "plan_sha256": turn.plan_sha256 if turn is not None else "",
        "runtime_config_sha256": runtime_config,
        "capability_registry_sha256": capability_registry,
        "model_identifier": result.model,
        "model_identifier_sha256": model_identifier_sha,
        "code_revision": revision,
        "corpora": [
            {
                "corpus_id": corpus.corpus_id,
                "generation_sha256": corpus.generation_sha256,
                "retrieval_profile_sha256": corpus.retrieval_profile_sha256,
            }
            for corpus in project.corpora
        ],
        "citation_content_sha256": citation_digests,
        "selected_capabilities": capability_bindings,
    }
    unique_blockers = tuple(dict.fromkeys(blockers))
    return CapsuleBuildAssessment(
        manifest_ready=not unique_blockers,
        blockers=unique_blockers,
        warnings=tuple(dict.fromkeys(warnings)),
        bindings=bindings,
    )


def build_capsule(context: CapsuleBuildContext) -> ResearchCapsule:
    assessment = assess_capsule(context)
    if not assessment.manifest_ready:
        raise RuntimeError(
            "research capsule bindings are incomplete: " + ", ".join(assessment.blockers)
        )

    project = context.project
    session = context.session
    result = context.result
    bindings = assessment.bindings
    references: list[CapsuleReference] = [
        CapsuleReference("project", "other", project.fingerprint, version=project.project_id),
        CapsuleReference(
            "session",
            "other",
            str(bindings["session_execution_fingerprint"]),
            version=session.session_id,
            metadata={"snapshot": "pre_execution"},
        ),
        CapsuleReference("query", "query", result.query_sha256),
        CapsuleReference("result", "result", result.result_id),
        CapsuleReference(
            "runtime-config", "config", str(bindings["runtime_config_sha256"])
        ),
        CapsuleReference(
            "capability-registry", "config", str(bindings["capability_registry_sha256"])
        ),
        CapsuleReference(
            "model-identifier",
            "model",
            str(bindings["model_identifier_sha256"]),
            version=result.model,
            metadata={
                "binding": "provider_model_identifier",
                "weights_content_addressed": "false",
            },
        ),
        CapsuleReference("policy", "policy", str(bindings["policy_sha256"])),
    ]

    for corpus in project.corpora:
        references.append(
            CapsuleReference(
                f"corpus:{corpus.corpus_id}:generation",
                "generation",
                corpus.generation_sha256,
                version=corpus.corpus_id,
            )
        )
        references.append(
            CapsuleReference(
                f"corpus:{corpus.corpus_id}:retrieval-profile",
                "config",
                corpus.retrieval_profile_sha256,
                version=corpus.corpus_id,
            )
        )

    for role, raw in sorted(bindings["selected_capabilities"].items()):
        references.append(
            CapsuleReference(
                f"capability:{role}",
                "other",
                str(raw["fingerprint"]),
                version=str(raw.get("version") or ""),
                metadata={
                    "capability_id": str(raw["capability_id"]),
                    "role": str(role),
                },
            )
        )

    for index, (citation_id, digest) in enumerate(
        zip(result.citation_ids, bindings["citation_content_sha256"])
    ):
        references.append(
            CapsuleReference(
                f"citation:{index}",
                "source",
                str(digest),
                metadata={"citation_id": citation_id},
            )
        )

    input_refs = tuple(
        ref.ref_id for ref in references if ref.ref_id not in {"result", "project"}
    )
    step = ReplayStep(
        step_id="research-query",
        operation="research_query",
        input_ref_ids=input_refs,
        output_ref_ids=("result",),
        capability_ref_id="model-identifier",
        policy_ref_id="policy",
        deterministic=False,
    )
    return ResearchCapsule(
        capsule_id=f"capsule_{uuid.uuid4().hex}",
        project_id=project.project_id,
        run_id=result.result_id,
        code_revision=str(bindings["code_revision"]),
        references=tuple(references),
        replay_steps=(step,),
        notes=assessment.warnings,
    )


__all__ = [
    "CapsuleBuildAssessment",
    "CapsuleBuildContext",
    "assess_capsule",
    "build_capsule",
]
