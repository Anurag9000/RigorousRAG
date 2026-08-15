"""Trusted built-in scientific domain adapter registration."""
from __future__ import annotations
from tools.domain_adapter import DomainAdapterRegistry
from tools.hydrology_domain_adapter import HydrologyDomainAdapter

def build_default_domain_registry() -> DomainAdapterRegistry:
    registry=DomainAdapterRegistry()
    registry.register(HydrologyDomainAdapter(),activate=True)
    return registry

__all__=["build_default_domain_registry"]
