"""Operator CLI for governed restore-intent deletion authorization and preflight."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_authorizations import (
    preflight_signed_retirement_restore_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_runtime import (
    get_signed_retirement_restore_deletion_authorization_store,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_runtime import (
    get_signed_retirement_restore_hold_store,
)
from tools.evidence_graph_set_signed_retirement_restore_runtime import (
    get_signed_retirement_restore_journal,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "authorization_id": value.authorization_id,
        "owner_id": value.owner_id,
        "restore_id": value.restore_id,
        "snapshot_digest": value.snapshot_digest,
        "target_path_digest": value.target_path_digest,
        "plan_digest": value.plan_digest,
        "policy_digest": value.policy_digest,
        "authorization_key": value.authorization_key,
        "minimum_age_seconds": value.minimum_age_seconds,
        "retain_latest_per_target": value.retain_latest_per_target,
        "include_completed": value.include_completed,
        "status": value.status,
        "authorized_actor_id": value.authorized_actor_id,
        "authorized_binding_method": value.authorized_binding_method,
        "authorized_binding_digest": value.authorized_binding_digest,
        "authorized_at": value.authorized_at,
        "expires_at": value.expires_at,
        "revoked_actor_id": value.revoked_actor_id,
        "revoked_binding_method": value.revoked_binding_method,
        "revoked_binding_digest": value.revoked_binding_digest,
        "revoked_at": value.revoked_at,
        "authorization_digest": value.authorization_digest,
        "contains_source_text": False,
        "raw_paths_returned": False,
        "deletion_performed": False,
        "restore_mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_deletion_cli"
        ),
        description=(
            "Authorize, revoke, inspect and preflight retention deletion of one "
            "terminal restore-intent record. No command deletes journal history."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    authorize = commands.add_parser("authorize")
    authorize.add_argument("restore_id")
    authorize.add_argument("--owner-id", required=True)
    authorize.add_argument("--confirm-restore-id", required=True)
    authorize.add_argument("--plan-digest", required=True)
    authorize.add_argument("--plan-generated-at", type=float, required=True)
    authorize.add_argument("--authorization-key", required=True)
    authorize.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=180 * 24 * 60 * 60,
    )
    authorize.add_argument("--retain-latest-per-target", type=int, default=1)
    authorize.add_argument("--include-completed", action="store_true")
    authorize.add_argument(
        "--expires-in-seconds", type=float, default=24 * 60 * 60
    )
    authorize.add_argument("--actor-id")
    authorize.add_argument("--limit", type=int, default=10_000)

    revoke = commands.add_parser("revoke")
    revoke.add_argument("authorization_id")
    revoke.add_argument("--owner-id", required=True)
    revoke.add_argument("--confirm-authorization-id", required=True)
    revoke.add_argument("--actor-id")

    status = commands.add_parser("status")
    status.add_argument("authorization_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--restore-id")
    listing.add_argument("--status")
    listing.add_argument("--limit", type=int, default=100)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("authorization_id")
    preflight.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if (
            args.command == "authorize"
            and args.restore_id != args.confirm_restore_id
        ):
            raise ValueError("restore confirmation differs.")
        if (
            args.command == "revoke"
            and args.authorization_id != args.confirm_authorization_id
        ):
            raise ValueError("authorization confirmation differs.")

        if args.command in {"status", "list"}:
            store = (
                get_signed_retirement_restore_deletion_authorization_store()
            )
            if args.command == "status":
                payload = _summary(store.get(args.authorization_id))
                payload["mutation_performed"] = False
                _print(payload)
                return 0
            values = store.list(
                owner_id=args.owner_id,
                restore_id=args.restore_id,
                status=args.status,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "authorizations": [
                        _summary(value) for value in values
                    ],
                    "mutation_performed": False,
                    "deletion_performed": False,
                    "contains_source_text": False,
                    "raw_paths_returned": False,
                }
            )
            return 0

        if args.command == "preflight":
            store = (
                get_signed_retirement_restore_deletion_authorization_store()
            )
            authorization = store.get(args.authorization_id)
            report = preflight_signed_retirement_restore_deletion(
                authorization=authorization,
                restore_journal=get_signed_retirement_restore_journal(),
                hold_store=get_signed_retirement_restore_hold_store(),
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "authorization_store_mutation_performed": False,
                    "deletion_performed": False,
                    "restore_mutation_performed": False,
                    "contains_source_text": False,
                    "raw_paths_returned": False,
                }
            )
            _print(payload)
            return 0

        binding = require_relation_review_actor(
            getattr(args, "actor_id", None),
            binding=load_relation_review_actor(),
        )
        store = get_signed_retirement_restore_deletion_authorization_store()
        if args.command == "authorize":
            value = store.authorize(
                owner_id=args.owner_id,
                restore_id=args.restore_id,
                plan_digest=args.plan_digest,
                plan_generated_at=args.plan_generated_at,
                authorization_key=args.authorization_key,
                actor=binding,
                restore_journal=get_signed_retirement_restore_journal(),
                hold_store=get_signed_retirement_restore_hold_store(),
                minimum_age_seconds=args.minimum_age_seconds,
                retain_latest_per_target=args.retain_latest_per_target,
                include_completed=args.include_completed,
                expires_in_seconds=args.expires_in_seconds,
                limit=args.limit,
            )
            payload = _summary(value)
            payload["authorization_mutation_performed"] = True
            _print(payload)
            return 0
        if args.command == "revoke":
            value = store.revoke(
                args.authorization_id,
                owner_id=args.owner_id,
                confirm_authorization_id=args.confirm_authorization_id,
                actor=binding,
            )
            payload = _summary(value)
            payload["authorization_mutation_performed"] = True
            _print(payload)
            return 0
        raise ValueError(
            "unsupported restore deletion authorization command."
        )
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
