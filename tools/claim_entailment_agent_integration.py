"""Optional fail-closed claim-entailment post-gate for ``SearchAgent`` answers.

The integration is inert until an application sets ``agent.entailment_provider``. When
configured, it post-processes only the server-owned citations already selected by the
agent, removes unsupported/contradicted claims, and narrows published citations to those
that actually supported retained claims. Provider failure therefore cannot add support.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from tools.claim_entailment import (
    ClaimGatePolicy,
    assess_claims,
    citation_identity,
    segment_atomic_claims,
    supported_answer_text,
)

_MARKER = "_claim_entailment_agent_gate_installed"
_ORIGINAL = "_claim_entailment_original_run"


def install_claim_entailment_agent_gate(module: ModuleType) -> None:
    if getattr(module, _MARKER, False):
        return
    search_agent = getattr(module, "SearchAgent", None)
    answer_type = getattr(module, "AgentAnswer", None)
    if search_agent is None or answer_type is None or not hasattr(search_agent, "run"):
        raise RuntimeError("search_agent_legacy does not expose the expected SearchAgent boundary")

    original_run = getattr(search_agent, _ORIGINAL, None)
    if original_run is None:
        original_run = search_agent.run
        setattr(search_agent, _ORIGINAL, original_run)

    def run_with_claim_gate(self: Any, query: str):
        answer = original_run(self, query)
        provider = getattr(self, "entailment_provider", None)
        if provider is None:
            metadata = dict(getattr(answer, "metadata", {}) or {})
            metadata.setdefault("claim_entailment_gate", "not_configured")
            answer.metadata = metadata
            return answer
        citations = list(getattr(answer, "citations", ()) or ())
        text = str(getattr(answer, "answer", "") or "").strip()
        if not text or not citations:
            metadata = dict(getattr(answer, "metadata", {}) or {})
            metadata["claim_entailment_gate"] = "no_authoritative_evidence"
            answer.metadata = metadata
            return answer
        policy = getattr(self, "claim_gate_policy", None)
        if policy is None:
            policy = ClaimGatePolicy()
        if not isinstance(policy, ClaimGatePolicy):
            raise RuntimeError("claim_gate_policy must be ClaimGatePolicy")
        claims = segment_atomic_claims(text, max_claims=64)
        if not claims:
            return answer
        result = assess_claims(claims, citations, provider, policy=policy)
        supported_text = supported_answer_text(result)
        if supported_text:
            answer.answer = supported_text
        else:
            answer.answer = (
                "The retrieved evidence did not semantically support a publishable "
                "answer under the configured claim-entailment policy."
            )
        allowed_ids = set(result.authoritative_citation_ids)
        answer.citations = [citation for citation in citations if citation_identity(citation) in allowed_ids]
        warnings = list(getattr(answer, "warnings", ()) or ())
        if result.rejected_claim_ids:
            warnings.append(
                f"Claim entailment gate removed {len(result.rejected_claim_ids)} unsupported or contradicted claim(s)."
            )
        metadata = dict(getattr(answer, "metadata", {}) or {})
        metadata.update(
            {
                "claim_entailment_gate": "applied",
                "claim_gate_fingerprint": result.fingerprint,
                "claims_assessed": len(result.assessments),
                "claims_supported": len(result.supported_claim_ids),
                "claims_rejected": len(result.rejected_claim_ids),
                "claims_contradicted": len(result.contradicted_claim_ids),
            }
        )
        answer.warnings = warnings
        answer.metadata = metadata
        return answer

    search_agent.run = run_with_claim_gate
    setattr(module, _MARKER, True)


__all__ = ["install_claim_entailment_agent_gate"]
