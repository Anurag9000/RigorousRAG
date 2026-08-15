"""Apply trusted runtime composition/provider bindings to newly-created agents."""

from __future__ import annotations

from typing import Any

from tools.runtime_composition import RuntimeComposition
from tools.runtime_providers import RuntimeProviderRegistry, runtime_providers


_PROVIDER_BINDINGS = (
    ("claim_entailment", "nli.provider", "entailment_provider"),
    ("multimodal", "multimodal.backend", "multimodal_backend"),
    ("page_late_interaction", "page_late_interaction.backend", "page_late_interaction_backend"),
    ("policy", "adaptive_policy.provider", "adaptive_policy_provider"),
)


def configure_agent_runtime(
    agent: Any,
    composition: RuntimeComposition,
    *,
    providers: RuntimeProviderRegistry | None = None,
) -> Any:
    """Attach only trusted server-owned runtime objects to one request-scoped agent.

    The function is intentionally attribute-based so the compatibility ``SearchAgent``
    can consume capabilities incrementally without changing its constructor signature.
    Unknown provider bindings are ignored unless their capability is selected, in which
    case a missing provider is only fatal for optional capabilities that composition had
    explicitly marked selected from a healthy binding.
    """

    if agent is None:
        raise ValueError("agent must be supplied")
    if not isinstance(composition, RuntimeComposition):
        raise TypeError("composition must be RuntimeComposition")
    registry = providers or runtime_providers

    setattr(agent, "runtime_config", composition.config)
    setattr(agent, "capability_registry", composition.capabilities)
    setattr(agent, "domain_registry", composition.domains)
    setattr(agent, "selected_capabilities", dict(composition.selected_capabilities))

    for role, provider_id, attribute in _PROVIDER_BINDINGS:
        selected = composition.selected_capabilities.get(role)
        if not selected:
            continue
        provider = registry.get(provider_id)
        if provider is None:
            # Deterministic/local policy is implemented directly and does not require an
            # injected learned provider. Other selected optional roles imply composition
            # saw a healthy provider and therefore must fail closed if it disappeared.
            if role == "policy" and selected != "policy.learned_adaptive":
                continue
            if role == "policy" and selected == "retrieval.hybrid":
                continue
            raise RuntimeError(f"selected runtime provider disappeared: {provider_id}")
        if not registry.healthy(provider_id):
            raise RuntimeError(f"selected runtime provider is unhealthy: {provider_id}")
        setattr(agent, attribute, provider)

    return agent


__all__ = ["configure_agent_runtime"]
