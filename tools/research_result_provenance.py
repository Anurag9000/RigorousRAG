"""Server-owned provenance enrichment for finalized research answers.

All durable research-result creation paths should pass through this module before result
identity is computed. It binds exact runtime/capability identities and content-addresses
the authoritative citation passage without treating URLs or source identifiers as content.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from tools.models import AgentAnswer, Citation
from tools.runtime_composition import RuntimeComposition

_RUNTIME_METADATA_KEY = "_rigorousrag_runtime"


def sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    "content_address_citations",
    "finalize_answer_provenance",
    "runtime_binding",
    "sha256_text",
]
