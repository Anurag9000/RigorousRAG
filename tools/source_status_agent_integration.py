"""Install fail-closed source-status gating on the live research agent.

The gate wraps the server citation registry rather than model text. Retracted,
withdrawn, superseded and corrected sources remain in historical storage but are not
published as current authoritative citations. The agent receives an explicit warning on
its final answer when evidence was rejected by source status.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Sequence

from tools.models import AgentAnswer, Citation
from tools.source_status_policy import effective_source_status

_PROMPT_LINE = (
    "- Source status is enforced by the server: evidence marked retracted, withdrawn, "
    "superseded or corrected may be withheld from authoritative citations. Do not "
    "reconstruct or cite withheld evidence from tool prose."
)


def _source_id(citation: Citation) -> str:
    return (citation.source_id or citation.url or "").strip()[:1000]


def install_source_status_agent_gate(module: ModuleType) -> ModuleType:
    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module")
    if getattr(module, "_source_status_agent_gate_installed", False):
        return module
    agent_class = getattr(module, "SearchAgent", None)
    original_register = getattr(agent_class, "_register_citations", None)
    original_run = getattr(agent_class, "run", None)
    if not isinstance(agent_class, type) or not callable(original_register) or not callable(original_run):
        raise RuntimeError("search agent citation boundary is unavailable")

    def register_citations(
        self: Any,
        incoming: Sequence[Citation],
        registry: list[Citation],
        seen: dict[tuple[str, str, str], str],
    ) -> list[Citation]:
        reader = getattr(self, "source_status_store", None)
        owner_id = getattr(self, "owner_id", "")
        if reader is None:
            return original_register(incoming, registry, seen)
        allowed: list[Citation] = []
        rejected: list[dict[str, str]] = []
        for citation in incoming:
            if not isinstance(citation, Citation):
                continue
            source_id = _source_id(citation)
            if not source_id:
                # A citation without stable source identity cannot be checked against a
                # status ledger; preserve legacy handling rather than fabricating status.
                allowed.append(citation)
                continue
            try:
                disposition = effective_source_status(reader, owner_id, source_id)
            except Exception:
                # Fail closed only for the status lookup of an explicitly identified
                # source: if the authoritative status service is unavailable, do not
                # publish the source as current evidence.
                rejected.append({"source_id": source_id, "status": "status_unavailable"})
                continue
            if disposition.allowed_as_current_evidence:
                allowed.append(citation)
                continue
            rejected.append(
                {
                    "source_id": source_id,
                    "status": disposition.status,
                    "replacement_source_id": disposition.replacement_source_id,
                    "event_sha256": disposition.event_sha256,
                }
            )
        if rejected:
            existing = getattr(self, "_source_status_rejections", None)
            if not isinstance(existing, list):
                existing = []
                setattr(self, "_source_status_rejections", existing)
            existing.extend(rejected[: max(0, 500 - len(existing))])
        return original_register(allowed, registry, seen)

    def run(self: Any, query: str) -> AgentAnswer:
        setattr(self, "_source_status_rejections", [])
        answer = original_run(self, query)
        if not isinstance(answer, AgentAnswer):
            return answer
        rejected = getattr(self, "_source_status_rejections", None)
        if not isinstance(rejected, list) or not rejected:
            return answer
        statuses = sorted({str(item.get("status") or "unavailable") for item in rejected if isinstance(item, dict)})
        warning = (
            f"The server withheld {len(rejected)} citation candidate(s) because their "
            f"current source status was {', '.join(statuses)}."
        )
        if warning not in answer.warnings:
            answer.warnings.append(warning)
        metadata = dict(answer.metadata or {})
        metadata["source_status_gate"] = {
            "withheld_count": len(rejected),
            "statuses": statuses,
            # Avoid exposing source IDs here; the source-status API is the explicit
            # owner-scoped inspection surface.
        }
        answer.metadata = metadata
        return answer

    agent_class._register_citations = register_citations
    agent_class.run = run
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if isinstance(prompt, str) and _PROMPT_LINE not in prompt:
        module.SYSTEM_PROMPT = prompt.rstrip() + "\n" + _PROMPT_LINE + "\n"
    module._source_status_original_register_citations = original_register
    module._source_status_original_run = original_run
    module._source_status_agent_gate_installed = True
    return module


__all__ = ["install_source_status_agent_gate"]
