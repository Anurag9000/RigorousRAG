"""Server-owned provenance enrichment for finalized research answers.

All durable research-result creation paths should pass through this module before result
identity is computed. It binds exact runtime/capability identities, optional project/
session execution context, and content-addresses authoritative citation passages without
treating URLs or source identifiers as evidence content.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from tools.models import AgentAnswer, Citation
from tools.research_workspace import ResearchSession
from tools.runtime_composition import RuntimeComposition

_RUNTIME_METADATA_KEY = "_rigorousrag_runtime"
_SESSION_METADATA_KEY = "_rigorousrag_session"


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _maybe_sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def content_address_citations(citations: Sequence[Citation]) -> tuple[Citation, ...]:
    """Attach a digest of the exact evidence passage to every addressable citation."""

    output: list[Citation] = []
    for citation in citations:
        if not isinstance(citation, Citation):
            raise TypeError("citations must contain Citation objects")
        copy = citation.model_copy(deep=True)
        evidence = copy.quote or copy.snippet or ""
        if evidence:
            metadata = dict(copy.metadata or {})
            metadata["evidence_sha256"] = sha256_text(evidence)
            copy.metadata = metadata
        output.append(copy)
    return tuple(output)


def runtime_binding(
    composition: RuntimeComposition,
    *,
    model: str,
    strategy: str,
) -> Mapping[str, Any]:
    if not isinstance(composition, RuntimeComposition):
        raise TypeError("composition must be RuntimeComposition")
    selected: dict[str, Mapping[str, str]] = {}
    for role, capability_id in sorted(composition.selected_capabilities.items()):
        descriptor = composition.capabilities.active(capability_id)
        if descriptor is None:
            continue
        selected[str(role)] = {
            "capability_id": descriptor.capability_id,
            "version": descriptor.version,
            "fingerprint": descriptor.fingerprint,
        }
    return {
        "schema_version": "1.0.0",
        "runtime_config_sha256": composition.config.fingerprint,
        "capability_registry_sha256": composition.capabilities.fingerprint,
        "retrieval_strategy": strategy,
        "model_identifier": model,
        "model_identifier_sha256": sha256_text(model) if model else "",
        "model_artifact_content_addressed": False,
        "selected_capabilities": selected,
    }


def turn_fingerprints(metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Derive authoritative plan/policy hashes from persisted server metadata."""

    if not isinstance(metadata, Mapping):
        return "", ""
    plan_sha = _maybe_sha(metadata.get("plan_fingerprint") or metadata.get("plan_sha256"))
    policy_sha = _maybe_sha(metadata.get("policy_fingerprint") or metadata.get("policy_sha256"))
    if not policy_sha:
        runtime = metadata.get(_RUNTIME_METADATA_KEY)
        selected = runtime.get("selected_capabilities") if isinstance(runtime, Mapping) else None
        policy = selected.get("policy") if isinstance(selected, Mapping) else None
        if isinstance(policy, Mapping):
            policy_sha = _maybe_sha(policy.get("fingerprint"))
    return plan_sha, policy_sha


def session_binding(metadata: Mapping[str, Any]) -> Mapping[str, str] | None:
    """Return a validated immutable session binding from result metadata, if present."""

    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(_SESSION_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return None
    project_id = raw.get("project_id")
    session_id = raw.get("session_id")
    session_fingerprint = _maybe_sha(raw.get("session_fingerprint_before"))
    plan_sha = _maybe_sha(raw.get("plan_sha256"))
    policy_sha = _maybe_sha(raw.get("policy_sha256"))
    if not isinstance(project_id, str) or not project_id.strip():
        return None
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    if not session_fingerprint:
        return None
    return {
        "project_id": project_id.strip(),
        "session_id": session_id.strip(),
        "session_fingerprint_before": session_fingerprint,
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
    }


def bind_answer_to_session(answer: AgentAnswer, session: ResearchSession) -> AgentAnswer:
    """Cryptographically bind a pending result to its authoritative session context.

    The binding is inserted before ``ResearchResultStore.put`` computes the immutable
    result ID. It captures the session fingerprint *before* the turn is appended, which
    makes concurrent session mutation detectable without pretending the post-append
    session fingerprint was known at execution time.
    """

    if not isinstance(answer, AgentAnswer):
        raise TypeError("answer must be AgentAnswer")
    if not isinstance(session, ResearchSession):
        raise TypeError("session must be ResearchSession")
    metadata = dict(answer.metadata or {})
    plan_sha, policy_sha = turn_fingerprints(metadata)
    metadata[_SESSION_METADATA_KEY] = {
        "schema_version": "1.0.0",
        "project_id": session.project_id,
        "session_id": session.session_id,
        "session_fingerprint_before": session.fingerprint,
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
    }
    return answer.model_copy(update={"metadata": metadata})


def carry_session_binding(answer: AgentAnswer, source_metadata: Mapping[str, Any]) -> AgentAnswer:
    """Carry a previously authenticated session binding into a replacement result."""

    if not isinstance(answer, AgentAnswer):
        raise TypeError("answer must be AgentAnswer")
    binding = session_binding(source_metadata)
    if binding is None:
        return answer
    metadata = dict(answer.metadata or {})
    metadata[_SESSION_METADATA_KEY] = {
        "schema_version": "1.0.0",
        **dict(binding),
    }
    return answer.model_copy(update={"metadata": metadata})


def finalize_answer_provenance(
    answer: AgentAnswer,
    composition: RuntimeComposition,
    *,
    model: str,
    strategy: str,
) -> AgentAnswer:
    """Return a copy enriched only with server-owned immutable provenance fields."""

    if not isinstance(answer, AgentAnswer):
        raise TypeError("answer must be AgentAnswer")
    metadata = dict(answer.metadata or {})
    metadata[_RUNTIME_METADATA_KEY] = dict(
        runtime_binding(composition, model=model, strategy=strategy)
    )
    citations = content_address_citations(tuple(answer.citations or ()))
    return answer.model_copy(update={"metadata": metadata, "citations": list(citations)})


__all__ = [
    "bind_answer_to_session",
    "carry_session_binding",
    "content_address_citations",
    "finalize_answer_provenance",
    "runtime_binding",
    "session_binding",
    "sha256_text",
    "turn_fingerprints",
]
