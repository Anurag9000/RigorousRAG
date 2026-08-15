"""Install final evidence-admissibility policy on the live research agent.

This gate runs after source-status filtering and optional semantic entailment. It can
remove citations and cited claims that violate reviewed source-trust policy. If the
configured trust registry is unavailable, source-dependent publication fails closed.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from tools.evidence_admissibility import (
    EvidenceAdmissibilityPolicy,
    evaluate_answer_admissibility,
)
from tools.models import AgentAnswer

_PROMPT_LINE = (
    "- Final evidence admissibility is enforced by the server. Causal and treatment "
    "claims may require reviewed eligible evidence; never reconstruct claims or citation "
    "markers removed by the server policy gate."
)


def install_evidence_admissibility_agent_gate(module: ModuleType) -> ModuleType:
    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module")
    if getattr(module, "_evidence_admissibility_agent_gate_installed", False):
        return module
    agent_class = getattr(module, "SearchAgent", None)
    original_run = getattr(agent_class, "run", None)
    if not isinstance(agent_class, type) or not callable(original_run):
        raise RuntimeError("search agent run boundary is unavailable")

    def run(self: Any, query: str) -> AgentAnswer:
        answer = original_run(self, query)
        if not isinstance(answer, AgentAnswer):
            return answer
        trust_reader = getattr(self, "source_trust_store", None)
        policy = getattr(self, "evidence_admissibility_policy", None)
        if trust_reader is None and policy is None:
            return answer
        if policy is not None and not isinstance(policy, EvidenceAdmissibilityPolicy):
            return AgentAnswer(
                answer="Evidence admissibility could not be verified; source-dependent synthesis is withheld.",
                citations=[],
                warnings=[*answer.warnings, "Evidence admissibility policy configuration is invalid."],
                metadata={**dict(answer.metadata or {}), "admissibility_gate": {"status": "configuration_error"}},
            )
        try:
            result = evaluate_answer_admissibility(
                answer.answer,
                answer.citations,
                owner_id=getattr(self, "owner_id", ""),
                trust_reader=trust_reader,
                policy=policy,
            )
        except Exception as exc:
            return AgentAnswer(
                answer="Evidence admissibility could not be verified; source-dependent synthesis is withheld.",
                citations=[],
                warnings=[
                    *answer.warnings,
                    f"Evidence admissibility gate failed closed ({type(exc).__name__}).",
                ],
                metadata={
                    **dict(answer.metadata or {}),
                    "admissibility_gate": {"status": "verification_unavailable"},
                },
            )

        warnings = list(answer.warnings)
        if result.rejected_claim_ids or result.rejected_citation_labels:
            warnings.append(
                "The evidence-admissibility policy removed "
                f"{len(result.rejected_claim_ids)} claim(s) and "
                f"{len(result.rejected_citation_labels)} citation(s)."
            )
        trust_revision_ids = tuple(
            dict.fromkeys(
                item.trust_revision_id
                for item in result.citation_decisions
                if item.trust_revision_id
            )
        )[:100]
        reviewed_sources = tuple(
            dict.fromkeys(
                item.source_id
                for item in result.citation_decisions
                if item.reviewed and item.trust_revision_id
            )
        )[:100]
        evaluated_sources = tuple(
            dict.fromkeys(item.source_id for item in result.citation_decisions if item.source_id)
        )[:100]
        metadata = dict(answer.metadata or {})
        metadata["admissibility_gate"] = {
            "status": "applied",
            "policy_sha256": result.policy_sha256,
            "result_fingerprint": result.fingerprint,
            "trust_revision_ids": list(trust_revision_ids),
            "reviewed_source_ids": list(reviewed_sources),
            "evaluated_source_ids": list(evaluated_sources),
            "rejected_claim_count": len(result.rejected_claim_ids),
            "rejected_citation_count": len(result.rejected_citation_labels),
            "claim_kinds": {
                "general": sum(1 for item in result.claim_decisions if item.claim_kind == "general"),
                "causal": sum(1 for item in result.claim_decisions if item.claim_kind == "causal"),
                "treatment": sum(1 for item in result.claim_decisions if item.claim_kind == "treatment"),
            },
        }
        final_answer = result.answer or (
            "No source-dependent claim remained admissible under the configured evidence policy."
        )
        return AgentAnswer(
            answer=final_answer,
            citations=list(result.citations),
            warnings=warnings,
            metadata=metadata,
        )

    agent_class.run = run
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if isinstance(prompt, str) and _PROMPT_LINE not in prompt:
        module.SYSTEM_PROMPT = prompt.rstrip() + "\n" + _PROMPT_LINE + "\n"
    module._evidence_admissibility_original_run = original_run
    module._evidence_admissibility_agent_gate_installed = True
    return module


__all__ = ["install_evidence_admissibility_agent_gate"]
