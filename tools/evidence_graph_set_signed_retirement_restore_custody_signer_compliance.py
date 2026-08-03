"""Read-only compliance audit joining signer records to admin-use reservations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_DIRECT_METHODS = frozenset({"process_environment", "descriptor_file"})
_CLASSIFICATIONS = frozenset(
    {
        "direct_compliant",
        "signed_committed_compliant",
        "signed_reserved_incomplete",
        "signed_missing_reservation",
        "signed_scope_mismatch",
        "not_applicable",
    }
)
_MAX_LIMIT = 10_000


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _register_action_digest(value: Any) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-signer-register-action-v1",
            "owner_id": value.owner_id,
            "key_id": value.key_id,
            "issuer": value.issuer,
            "algorithm": value.algorithm,
            "public_key_sha256": value.public_key_sha256,
        }
    )


def _retire_action_digest(value: Any) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-signer-retire-action-v1",
            "owner_id": value.owner_id,
            "key_id": value.key_id,
            "issuer": value.issuer,
            "algorithm": value.algorithm,
            "public_key_sha256": value.public_key_sha256,
            "registered_binding_digest": value.registered_binding_digest,
        }
    )


def _classify(
    *,
    owner_id: str,
    key_id: str,
    action: str,
    binding_method: str,
    binding_digest: str,
    action_digest: str,
    uses_by_binding: dict[str, Any],
) -> tuple[str, str | None]:
    method = _identifier(binding_method, "binding_method", 50)
    binding = _digest(binding_digest, "binding_digest")
    if method in _DIRECT_METHODS:
        return "direct_compliant", None
    use = uses_by_binding.get(binding)
    if use is None:
        return "signed_missing_reservation", None
    expected_scope = (
        use.owner_id == owner_id
        and use.key_id == key_id
        and use.action == action
        and use.action_digest == action_digest
        and use.binding_digest == binding
    )
    if not expected_scope:
        return "signed_scope_mismatch", use.use_id
    if use.state == "reserved":
        return "signed_reserved_incomplete", use.use_id
    if use.state == "committed":
        return "signed_committed_compliant", use.use_id
    raise RuntimeError("admin-use store returned an unsupported state.")


@dataclass(frozen=True)
class CustodySignerComplianceItem:
    key_id: str
    issuer: str
    public_key_sha256: str
    state: str
    registration_binding_method: str
    registration_binding_digest: str
    registration_classification: str
    registration_use_id: str | None
    retirement_binding_method: str | None
    retirement_binding_digest: str | None
    retirement_classification: str
    retirement_use_id: str | None
    eligible_for_new_signatures: bool
    governance_compliant_for_historical_verification: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _identifier(self.key_id, "key_id", 200))
        object.__setattr__(self, "issuer", _identifier(self.issuer, "issuer", 200))
        object.__setattr__(
            self,
            "public_key_sha256",
            _digest(self.public_key_sha256, "public_key_sha256"),
        )
        state = _identifier(self.state, "state", 30)
        if state not in {"active", "retired"}:
            raise ValueError("signer state is unsupported.")
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "registration_binding_method",
            _identifier(
                self.registration_binding_method,
                "registration_binding_method",
                50,
            ),
        )
        object.__setattr__(
            self,
            "registration_binding_digest",
            _digest(
                self.registration_binding_digest,
                "registration_binding_digest",
            ),
        )
        registration = _identifier(
            self.registration_classification,
            "registration_classification",
            80,
        )
        retirement = _identifier(
            self.retirement_classification,
            "retirement_classification",
            80,
        )
        if registration not in _CLASSIFICATIONS - {"not_applicable"}:
            raise ValueError("registration classification is unsupported.")
        if retirement not in _CLASSIFICATIONS:
            raise ValueError("retirement classification is unsupported.")
        object.__setattr__(self, "registration_classification", registration)
        object.__setattr__(self, "retirement_classification", retirement)
        for field in ("registration_use_id", "retirement_use_id"):
            value = getattr(self, field)
            object.__setattr__(
                self,
                field,
                None if value is None else _digest(value, field),
            )
        if state == "active":
            if (
                self.retirement_binding_method is not None
                or self.retirement_binding_digest is not None
                or retirement != "not_applicable"
                or self.retirement_use_id is not None
            ):
                raise ValueError("active signer retirement fields are invalid.")
        else:
            object.__setattr__(
                self,
                "retirement_binding_method",
                _identifier(
                    self.retirement_binding_method,
                    "retirement_binding_method",
                    50,
                ),
            )
            object.__setattr__(
                self,
                "retirement_binding_digest",
                _digest(
                    self.retirement_binding_digest,
                    "retirement_binding_digest",
                ),
            )
        for field in (
            "eligible_for_new_signatures",
            "governance_compliant_for_historical_verification",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean.")
        registration_ok = registration in {
            "direct_compliant",
            "signed_committed_compliant",
        }
        retirement_ok = retirement in {
            "not_applicable",
            "direct_compliant",
            "signed_committed_compliant",
        }
        if self.eligible_for_new_signatures != (state == "active" and registration_ok):
            raise ValueError("new-signature eligibility differs from governance state.")
        if self.governance_compliant_for_historical_verification != (
            registration_ok and retirement_ok
        ):
            raise ValueError("historical governance compliance differs from classifications.")


@dataclass(frozen=True)
class CustodySignerComplianceReport:
    owner_id: str
    generated_at: float
    signer_count: int
    admin_use_count: int
    compliant_active_count: int
    noncompliant_active_count: int
    classification_counts: dict[str, int]
    items: tuple[CustodySignerComplianceItem, ...]
    report_digest: str
    registry_mutation_performed: bool = False
    admin_use_mutation_performed: bool = False
    key_material_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        signer_count = _integer(self.signer_count, "signer_count", 0, _MAX_LIMIT)
        admin_count = _integer(self.admin_use_count, "admin_use_count", 0, _MAX_LIMIT)
        if signer_count != len(self.items):
            raise ValueError("signer compliance count differs from items.")
        compliant_active = sum(item.eligible_for_new_signatures for item in self.items)
        active = sum(item.state == "active" for item in self.items)
        if (
            compliant_active != self.compliant_active_count
            or active - compliant_active != self.noncompliant_active_count
        ):
            raise ValueError("active signer compliance counts differ from items.")
        counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        seen: set[str] = set()
        for item in self.items:
            if item.key_id in seen:
                raise ValueError("signer compliance report contains duplicate key IDs.")
            seen.add(item.key_id)
            counts[item.registration_classification] += 1
            counts[item.retirement_classification] += 1
        if dict(self.classification_counts) != counts:
            raise ValueError("signer compliance classification counts differ from items.")
        for field in (
            "registry_mutation_performed",
            "admin_use_mutation_performed",
            "key_material_mutation_performed",
            "source_text_returned",
            "raw_path_returned",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        stable = {
            "scope": "rigorousrag-custody-signer-compliance-report-v1",
            "owner_id": owner,
            "generated_at": generated,
            "signer_count": signer_count,
            "admin_use_count": admin_count,
            "compliant_active_count": compliant_active,
            "noncompliant_active_count": active - compliant_active,
            "classification_counts": counts,
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from signer compliance report.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "signer_count", signer_count)
        object.__setattr__(self, "admin_use_count", admin_count)
        object.__setattr__(self, "compliant_active_count", compliant_active)
        object.__setattr__(
            self,
            "noncompliant_active_count",
            active - compliant_active,
        )
        object.__setattr__(self, "classification_counts", counts)
        object.__setattr__(self, "report_digest", digest)


def audit_custody_signer_compliance(
    *,
    owner_id: str,
    registry: Any,
    admin_use_store: Any | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> CustodySignerComplianceReport:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    signers = tuple(registry.list(owner_id=owner, limit=count))
    if len(signers) >= count:
        raise RuntimeError("signer compliance audit reached the bounded signer limit.")
    uses = (
        ()
        if admin_use_store is None
        else tuple(admin_use_store.list(owner_id=owner, limit=count))
    )
    if len(uses) >= count:
        raise RuntimeError("signer compliance audit reached the bounded use limit.")
    uses_by_binding: dict[str, Any] = {}
    use_ids: set[str] = set()
    for use in uses:
        if use.use_id in use_ids or use.binding_digest in uses_by_binding:
            raise RuntimeError("admin-use store returned duplicate identities.")
        use_ids.add(use.use_id)
        uses_by_binding[use.binding_digest] = use
    items: list[CustodySignerComplianceItem] = []
    signer_ids: set[str] = set()
    for value in signers:
        if value.key_id in signer_ids:
            raise RuntimeError("signer registry returned duplicate key IDs.")
        signer_ids.add(value.key_id)
        registration, registration_use = _classify(
            owner_id=owner,
            key_id=value.key_id,
            action="register",
            binding_method=value.registered_binding_method,
            binding_digest=value.registered_binding_digest,
            action_digest=_register_action_digest(value),
            uses_by_binding=uses_by_binding,
        )
        if value.state == "active":
            retirement = "not_applicable"
            retirement_use = None
        else:
            retirement, retirement_use = _classify(
                owner_id=owner,
                key_id=value.key_id,
                action="retire",
                binding_method=value.retired_binding_method,
                binding_digest=value.retired_binding_digest,
                action_digest=_retire_action_digest(value),
                uses_by_binding=uses_by_binding,
            )
        registration_ok = registration in {
            "direct_compliant",
            "signed_committed_compliant",
        }
        retirement_ok = retirement in {
            "not_applicable",
            "direct_compliant",
            "signed_committed_compliant",
        }
        items.append(
            CustodySignerComplianceItem(
                key_id=value.key_id,
                issuer=value.issuer,
                public_key_sha256=value.public_key_sha256,
                state=value.state,
                registration_binding_method=value.registered_binding_method,
                registration_binding_digest=value.registered_binding_digest,
                registration_classification=registration,
                registration_use_id=registration_use,
                retirement_binding_method=value.retired_binding_method,
                retirement_binding_digest=value.retired_binding_digest,
                retirement_classification=retirement,
                retirement_use_id=retirement_use,
                eligible_for_new_signatures=(
                    value.state == "active" and registration_ok
                ),
                governance_compliant_for_historical_verification=(
                    registration_ok and retirement_ok
                ),
            )
        )
    rendered = tuple(sorted(items, key=lambda item: item.key_id))
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    for item in rendered:
        counts[item.registration_classification] += 1
        counts[item.retirement_classification] += 1
    compliant_active = sum(item.eligible_for_new_signatures for item in rendered)
    active = sum(item.state == "active" for item in rendered)
    stable = {
        "scope": "rigorousrag-custody-signer-compliance-report-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "signer_count": len(rendered),
        "admin_use_count": len(uses),
        "compliant_active_count": compliant_active,
        "noncompliant_active_count": active - compliant_active,
        "classification_counts": counts,
        "items": [asdict(item) for item in rendered],
    }
    return CustodySignerComplianceReport(
        owner_id=owner,
        generated_at=timestamp,
        signer_count=len(rendered),
        admin_use_count=len(uses),
        compliant_active_count=compliant_active,
        noncompliant_active_count=active - compliant_active,
        classification_counts=counts,
        items=rendered,
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "CustodySignerComplianceItem",
    "CustodySignerComplianceReport",
    "audit_custody_signer_compliance",
]
