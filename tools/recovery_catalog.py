"""Dependency-aware disaster-recovery catalog and deterministic restore preflight."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryAsset:
    """One versioned recovery artifact and the logical assets it depends on."""

    name: str
    artifact: str
    sha256: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.artifact.strip():
            raise ValueError("recovery asset name and artifact must be non-empty")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("recovery asset sha256 must be a 64-character hexadecimal digest")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("recovery dependencies must be unique")
        if self.name in self.depends_on:
            raise ValueError("a recovery asset cannot depend on itself")
        if any(not item.strip() for item in self.depends_on):
            raise ValueError("recovery dependency names must be non-empty")


@dataclass(frozen=True)
class RecoveryCatalog:
    generation: str
    assets: tuple[RecoveryAsset, ...]

    def __post_init__(self) -> None:
        if not self.generation.strip():
            raise ValueError("recovery catalog generation must be non-empty")
        names = [asset.name for asset in self.assets]
        artifacts = [asset.artifact for asset in self.assets]
        if len(set(names)) != len(names):
            raise ValueError("recovery asset names must be unique")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("recovery artifact names must be unique")


@dataclass(frozen=True)
class RecoveryPlan:
    ready: bool
    restore_order: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    checksum_mismatches: tuple[str, ...]
    missing_dependencies: tuple[str, ...]
    cyclic_assets: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _topological_order(
    assets: Mapping[str, RecoveryAsset],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    missing_dependencies: set[str] = set()
    indegree = {name: 0 for name in assets}
    dependents: dict[str, list[str]] = {name: [] for name in assets}
    for asset in assets.values():
        for dependency in asset.depends_on:
            if dependency not in assets:
                missing_dependencies.add(f"{asset.name}:{dependency}")
                continue
            indegree[asset.name] += 1
            dependents[dependency].append(asset.name)

    ready = [name for name, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for dependent in sorted(dependents[name]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    cyclic = tuple(sorted(name for name, degree in indegree.items() if degree > 0))
    return tuple(order), tuple(sorted(missing_dependencies)), cyclic


def plan_restore(
    catalog: RecoveryCatalog,
    available_artifacts: Mapping[str, str],
) -> RecoveryPlan:
    """Validate artifact availability/integrity and compute a stable dependency order.

    ``available_artifacts`` maps artifact name to its independently computed SHA-256 digest.
    No restore is considered ready when an artifact is absent, corrupt, structurally
    disconnected, or part of a dependency cycle.
    """

    assets = {asset.name: asset for asset in catalog.assets}
    order, missing_dependencies, cyclic = _topological_order(assets)
    missing: list[str] = []
    mismatches: list[str] = []
    for asset in catalog.assets:
        observed = available_artifacts.get(asset.artifact)
        if observed is None:
            missing.append(asset.artifact)
        elif observed.lower() != asset.sha256.lower():
            mismatches.append(asset.artifact)

    reasons: list[str] = []
    if missing:
        reasons.append("restore_artifacts_missing")
    if mismatches:
        reasons.append("restore_checksum_mismatch")
    if missing_dependencies:
        reasons.append("restore_dependency_missing")
    if cyclic:
        reasons.append("restore_dependency_cycle")
    return RecoveryPlan(
        ready=not reasons,
        restore_order=order,
        missing_artifacts=tuple(sorted(missing)),
        checksum_mismatches=tuple(sorted(mismatches)),
        missing_dependencies=missing_dependencies,
        cyclic_assets=cyclic,
        reason_codes=tuple(reasons),
    )


__all__ = ["RecoveryAsset", "RecoveryCatalog", "RecoveryPlan", "plan_restore"]
