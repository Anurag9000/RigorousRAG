"""One-operation signed-assertion administration of custody signer keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_public,
    _public_fingerprint,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use import (
    CustodySignerAdminUse,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_runtime import (
    get_custody_signer_admin_use_store,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_runtime import (
    get_custody_signer_key_registry,
)
from tools.security import normalize_owner_id


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


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _record_scope(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "key_id": value.key_id,
        "issuer": value.issuer,
        "algorithm": value.algorithm,
        "public_key_sha256": value.public_key_sha256,
        "registered_binding_digest": value.registered_binding_digest,
    }


def _register_action_digest(
    *,
    owner_id: str,
    key_id: str,
    issuer: str,
    public_key_sha256: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-signer-register-action-v1",
            "owner_id": normalize_owner_id(owner_id),
            "key_id": _identifier(key_id, "key_id", 200),
            "issuer": _identifier(issuer, "issuer", 200),
            "algorithm": "ed25519",
            "public_key_sha256": _digest(
                public_key_sha256,
                "public_key_sha256",
            ),
        }
    )


def _retire_action_digest(value: Any) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-signer-retire-action-v1",
            **_record_scope(value),
        }
    )


def _use_summary(value: Any) -> dict[str, Any]:
    return {
        "use_id": value.use_id,
        "assertion_issuer": value.assertion_issuer,
        "assertion_expires_at": value.assertion_expires_at,
        "binding_method": value.binding_method,
        "owner_id": value.owner_id,
        "action": value.action,
        "key_id": value.key_id,
        "action_digest": value.action_digest,
        "state": value.state,
        "reserved_at": value.reserved_at,
        "committed_at": value.committed_at,
        "contains_assertion_body": False,
        "contains_assertion_signature": False,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def _record_summary(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "key_id": value.key_id,
        "issuer": value.issuer,
        "algorithm": value.algorithm,
        "public_key_sha256": value.public_key_sha256,
        "state": value.state,
        "registered_binding_method": value.registered_binding_method,
        "registered_binding_digest": value.registered_binding_digest,
        "registered_at": value.registered_at,
        "retired_binding_method": value.retired_binding_method,
        "retired_binding_digest": value.retired_binding_digest,
        "retired_at": value.retired_at,
        "record_digest": value.record_digest,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_cli"
        ),
        description=(
            "Use one short-lived signed actor assertion for exactly one custody "
            "signer registration or retirement."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register-signed")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--key-id", required=True)
    register.add_argument("--issuer", required=True)
    register.add_argument("--public-key-path", required=True)
    register.add_argument("--confirm-public-key-sha256", required=True)
    register.add_argument("--actor-id")
    register.add_argument("--registry-db-path")
    register.add_argument("--admin-use-db-path")

    retire = commands.add_parser("retire-signed")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--key-id", required=True)
    retire.add_argument("--confirm-key-id", required=True)
    retire.add_argument("--actor-id")
    retire.add_argument("--registry-db-path")
    retire.add_argument("--admin-use-db-path")

    status = commands.add_parser("status")
    status.add_argument("use_id")
    status.add_argument("--admin-use-db-path")
    return parser


def _reserve_or_require_prior(
    *,
    candidate: CustodySignerAdminUse,
    store: Any,
    action_already_exists: bool,
):
    if action_already_exists:
        try:
            stored = store.get(candidate.use_id)
        except KeyError as exc:
            raise PermissionError(
                "signed signer administration may not backfill an existing action."
            ) from exc
        if stored.use_id != candidate.use_id:
            raise RuntimeError("signer admin-use identity differs.")
        return stored
    return store.reserve(candidate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "status":
            value = get_custody_signer_admin_use_store(
                args.admin_use_db_path
            ).get(args.use_id)
            payload = _use_summary(value)
            payload.update(
                {
                    "admin_use_mutation_performed": False,
                    "registry_mutation_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retire-signed" and args.confirm_key_id != args.key_id:
            raise ValueError("signer retirement confirmation differs.")
        fingerprint: str | None = None
        if args.command == "register-signed":
            fingerprint = _public_fingerprint(_load_public(args.public_key_path))
            if fingerprint != _digest(
                args.confirm_public_key_sha256,
                "confirm_public_key_sha256",
            ):
                raise ValueError("public-key fingerprint confirmation differs.")
        binding = require_relation_review_actor(
            args.actor_id,
            binding=load_relation_review_actor(),
        )
        registry = get_custody_signer_key_registry(args.registry_db_path)
        store = get_custody_signer_admin_use_store(args.admin_use_db_path)
        now = time.time()
        owner = normalize_owner_id(args.owner_id)
        key_id = _identifier(args.key_id, "key_id", 200)
        if args.command == "register-signed":
            action_digest = _register_action_digest(
                owner_id=owner,
                key_id=key_id,
                issuer=args.issuer,
                public_key_sha256=fingerprint,
            )
            candidate = CustodySignerAdminUse.reserve(
                binding=binding,
                owner_id=owner,
                action="register",
                key_id=key_id,
                action_digest=action_digest,
                now=now,
            )
            try:
                existing = registry.get(owner_id=owner, key_id=key_id)
            except KeyError:
                existing = None
            use = _reserve_or_require_prior(
                candidate=candidate,
                store=store,
                action_already_exists=existing is not None,
            )
            record = registry.register(
                owner_id=owner,
                key_id=key_id,
                issuer=args.issuer,
                public_key_path=args.public_key_path,
                actor=binding,
                now=now,
            )
        else:
            existing = registry.get(owner_id=owner, key_id=key_id)
            action_digest = _retire_action_digest(existing)
            candidate = CustodySignerAdminUse.reserve(
                binding=binding,
                owner_id=owner,
                action="retire",
                key_id=key_id,
                action_digest=action_digest,
                now=now,
            )
            use = _reserve_or_require_prior(
                candidate=candidate,
                store=store,
                action_already_exists=existing.state == "retired",
            )
            record = registry.retire(
                owner_id=owner,
                key_id=key_id,
                confirm_key_id=key_id,
                actor=binding,
                now=now,
            )
        committed = store.commit(
            use.use_id,
            confirm_use_id=use.use_id,
            now=now,
        )
        _print(
            {
                "admin_use": _use_summary(committed),
                "signer_record": _record_summary(record),
                "admin_use_mutation_performed": True,
                "registry_mutation_performed": True,
                "key_material_mutation_performed": False,
                "key_deletion_performed": False,
                "source_text_returned": False,
            }
        )
        return 0
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
