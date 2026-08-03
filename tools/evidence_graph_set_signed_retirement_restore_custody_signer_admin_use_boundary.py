"""Credential-provenance boundary for signed signer-administration uses."""

from __future__ import annotations

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_admin_use as _base,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)

_DIRECT_OR_UNAUTHENTICATED = frozenset(
    {"process_environment", "descriptor_file", "command_line"}
)


def _assertion_fields(binding):
    if not isinstance(binding, _base.ReviewActorBinding):
        raise ValueError("binding must be ReviewActorBinding.")
    method = _identifier(binding.binding_method, "binding_method", 50)
    if method in _DIRECT_OR_UNAUTHENTICATED:
        raise PermissionError("signer admin-use requires a signed expiring assertion.")
    assertion_digest = getattr(binding, "assertion_digest", None)
    if assertion_digest is None:
        assertion_digest = binding.binding_digest
    issuer = getattr(binding, "assertion_issuer", None)
    if issuer is None:
        issuer = getattr(binding, "issuer", None)
    expires_at = getattr(binding, "assertion_expires_at", None)
    if expires_at is None:
        expires_at = getattr(binding, "expires_at", None)
    if issuer is None or expires_at is None:
        raise PermissionError(
            "signed signer administration requires issuer and expiry provenance."
        )
    _base._SIGNED_METHODS = frozenset(set(_base._SIGNED_METHODS) | {method})
    return (
        _digest(assertion_digest, "assertion_digest"),
        _identifier(issuer, "assertion_issuer", 200),
        _timestamp(expires_at, "assertion_expires_at"),
    )


_base._assertion_fields = _assertion_fields

CustodySignerAdminUse = _base.CustodySignerAdminUse
CustodySignerAdminUseStore = _base.CustodySignerAdminUseStore
deterministic_signer_admin_use_id = _base.deterministic_signer_admin_use_id


__all__ = [
    "CustodySignerAdminUse",
    "CustodySignerAdminUseStore",
    "deterministic_signer_admin_use_id",
]
