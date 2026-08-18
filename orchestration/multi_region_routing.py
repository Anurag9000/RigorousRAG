"""Provider-neutral, CAS-bound publication of multi-region traffic authority.

The region authority store decides *who may write*.  This module safely projects that
decision into a deployment traffic router.  A provider adapter must expose its current
route revision and echo the authority fencing token after compare-and-set publication.
No DNS/load-balancer/cloud API is imported or invoked at module import time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from orchestration.multi_region_authority import RegionAuthorityRecord, SQLiteRegionAuthorityStore


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _revision(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ProviderRegionRoute:
    owner_id: str
    service_id: str
    region_id: str
    provider_revision: int
    authority_fencing_token: int
    provider_evidence_sha256: str

    def __post_init__(self) -> None:
        for name in ("owner_id", "service_id", "region_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "provider_revision", _revision(self.provider_revision, "provider_revision"))
        token = self.authority_fencing_token
        if isinstance(token, bool) or not isinstance(token, int) or token < 1:
            raise ValueError("authority_fencing_token must be positive")
        object.__setattr__(self, "provider_evidence_sha256", _sha(self.provider_evidence_sha256, "provider_evidence_sha256"))

    @property
    def route_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-provider-region-route/v1", **asdict(self)})


class RegionRouteProvider(Protocol):
    def read_route(self, *, owner_id: str, service_id: str) -> ProviderRegionRoute | None: ...

    def compare_and_set_route(
        self,
        *,
        owner_id: str,
        service_id: str,
        expected_provider_revision: int | None,
        target_region_id: str,
        authority_fencing_token: int,
        idempotency_key: str,
    ) -> ProviderRegionRoute: ...


@dataclass(frozen=True)
class RegionRoutePublicationReceipt:
    owner_id: str
    service_id: str
    authority_region: str
    authority_revision: int
    authority_fencing_token: int
    authority_decision_sha256: str
    before_route_sha256: str | None
    after_route_sha256: str
    provider_revision: int
    publication_performed: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("owner_id", "service_id", "authority_region"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "authority_revision", _revision(self.authority_revision, "authority_revision"))
        if self.authority_revision < 1:
            raise ValueError("authority_revision must be positive")
        if isinstance(self.authority_fencing_token, bool) or not isinstance(self.authority_fencing_token, int) or self.authority_fencing_token < 1:
            raise ValueError("authority_fencing_token must be positive")
        object.__setattr__(self, "authority_decision_sha256", _sha(self.authority_decision_sha256, "authority_decision_sha256"))
        if self.before_route_sha256 is not None:
            object.__setattr__(self, "before_route_sha256", _sha(self.before_route_sha256, "before_route_sha256"))
        object.__setattr__(self, "after_route_sha256", _sha(self.after_route_sha256, "after_route_sha256"))
        object.__setattr__(self, "provider_revision", _revision(self.provider_revision, "provider_revision"))
        if not isinstance(self.publication_performed, bool):
            raise ValueError("publication_performed must be boolean")
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match region route publication")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-region-route-publication/v1",
            "owner_id": self.owner_id,
            "service_id": self.service_id,
            "authority_region": self.authority_region,
            "authority_revision": self.authority_revision,
            "authority_fencing_token": self.authority_fencing_token,
            "authority_decision_sha256": self.authority_decision_sha256,
            "before_route_sha256": self.before_route_sha256,
            "after_route_sha256": self.after_route_sha256,
            "provider_revision": self.provider_revision,
            "publication_performed": self.publication_performed,
        }


def _authority_idempotency_key(record: RegionAuthorityRecord) -> str:
    return _digest({
        "schema": "rigorousrag-region-route-publication-key/v1",
        "owner_id": record.owner_id,
        "service_id": record.service_id,
        "authority_region": record.authority_region,
        "authority_revision": record.revision,
        "authority_fencing_token": record.fencing_token,
        "decision_sha256": record.decision_sha256,
    })


def publish_authoritative_region_route(
    *,
    owner_id: str,
    service_id: str,
    authority_store: SQLiteRegionAuthorityStore,
    provider: RegionRouteProvider,
) -> RegionRoutePublicationReceipt:
    """CAS-publish the current authority region to an injected traffic provider."""

    owner = _text(owner_id, "owner_id")
    service = _text(service_id, "service_id")
    if not isinstance(authority_store, SQLiteRegionAuthorityStore):
        raise ValueError("authority_store must be SQLiteRegionAuthorityStore")
    authority = authority_store.get(owner_id=owner, service_id=service)
    if authority is None:
        raise RuntimeError("no durable region authority exists")
    # Re-assert through the store immediately before publication. This catches callers
    # carrying a stale record from a previous failover generation.
    authority_store.assert_write_authority(
        owner_id=owner,
        service_id=service,
        region_id=authority.authority_region,
        fencing_token=authority.fencing_token,
    )
    before = provider.read_route(owner_id=owner, service_id=service)
    if before is not None and (before.owner_id != owner or before.service_id != service):
        raise RuntimeError("provider returned a route outside the requested owner/service scope")
    if (
        before is not None
        and before.region_id == authority.authority_region
        and before.authority_fencing_token == authority.fencing_token
    ):
        after = before
        performed = False
    else:
        expected_revision = None if before is None else before.provider_revision
        after = provider.compare_and_set_route(
            owner_id=owner,
            service_id=service,
            expected_provider_revision=expected_revision,
            target_region_id=authority.authority_region,
            authority_fencing_token=authority.fencing_token,
            idempotency_key=_authority_idempotency_key(authority),
        )
        performed = True
    if after.owner_id != owner or after.service_id != service:
        raise RuntimeError("provider published route outside requested owner/service scope")
    if after.region_id != authority.authority_region:
        raise RuntimeError("provider did not publish the authoritative region")
    if after.authority_fencing_token != authority.fencing_token:
        raise RuntimeError("provider did not preserve the authority fencing token")
    if before is not None and performed and after.provider_revision <= before.provider_revision:
        raise RuntimeError("provider route revision did not advance after publication")
    payload = {
        "schema": "rigorousrag-region-route-publication/v1",
        "owner_id": owner,
        "service_id": service,
        "authority_region": authority.authority_region,
        "authority_revision": authority.revision,
        "authority_fencing_token": authority.fencing_token,
        "authority_decision_sha256": authority.decision_sha256,
        "before_route_sha256": None if before is None else before.route_sha256,
        "after_route_sha256": after.route_sha256,
        "provider_revision": after.provider_revision,
        "publication_performed": performed,
    }
    return RegionRoutePublicationReceipt(**payload, receipt_sha256=_digest(payload))


__all__ = [
    "ProviderRegionRoute",
    "RegionRouteProvider",
    "RegionRoutePublicationReceipt",
    "publish_authoritative_region_route",
]
