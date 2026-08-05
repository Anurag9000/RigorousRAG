"""Privacy-conscious operator CLI for governed scientific claim extractors."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_claim_extractor_registry import (
    GovernedScientificClaimExtractorService,
)
from tools.evidence_graph_claim_extractor_runtime import (
    get_scientific_claim_extractor_policy,
    get_scientific_claim_extractor_registry,
)
from tools.evidence_graph_relation_actor import load_relation_review_actor


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "extractor_name": value.extractor_name,
        "extractor_version": value.extractor_version,
        "extractor_kind": value.extractor_kind,
        "implementation_sha256": value.implementation_sha256,
        "configuration_sha256": value.configuration_sha256,
        "output_schema_sha256": value.output_schema_sha256,
        "supported_claim_types": list(value.supported_claim_types),
        "supported_modalities": list(value.supported_modalities),
        "supported_languages": list(value.supported_languages),
        "state": value.state,
        "registered_actor_id": value.registered_actor_id,
        "registered_binding_method": value.registered_binding_method,
        "registered_binding_digest": value.registered_binding_digest,
        "registered_at": value.registered_at,
        "retired_actor_id": value.retired_actor_id,
        "retired_binding_method": value.retired_binding_method,
        "retired_binding_digest": value.retired_binding_digest,
        "retired_at": value.retired_at,
        "record_digest": value.record_digest,
        "contains_credentials": False,
        "contains_prompt_text": False,
        "contains_model_response": False,
        "contains_source_text": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_claim_extractor_cli",
        description=(
            "Inspect, register, and retire exact scientific claim extractor versions. "
            "The registry stores digests and capabilities, never credentials or prompts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--extractor-name", required=True)
    status.add_argument("--extractor-version", required=True)

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--extractor-name")
    listing.add_argument("--state", choices=["active", "retired"])
    listing.add_argument("--limit", type=int, default=100)

    register = commands.add_parser("register")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--extractor-name", required=True)
    register.add_argument("--extractor-version", required=True)
    register.add_argument("--extractor-kind", choices=["model", "rule"], required=True)
    register.add_argument("--implementation-sha256", required=True)
    register.add_argument("--configuration-sha256", required=True)
    register.add_argument("--claim-type", action="append", required=True)
    register.add_argument("--modality", action="append", required=True)
    register.add_argument("--language", action="append", required=True)

    retire = commands.add_parser("retire")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--extractor-name", required=True)
    retire.add_argument("--extractor-version", required=True)
    retire.add_argument("--confirm-record-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry = get_scientific_claim_extractor_registry()
        if args.command == "status":
            value = registry.get(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                extractor_version=args.extractor_version,
            )
            _print({**_summary(value), "mutation_performed": False})
            return 0
        if args.command == "list":
            values = registry.list(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "item_count": len(values),
                    "items": [_summary(value) for value in values],
                    "mutation_performed": False,
                    "contains_credentials": False,
                    "contains_prompt_text": False,
                    "contains_model_response": False,
                    "contains_source_text": False,
                }
            )
            return 0
        service = GovernedScientificClaimExtractorService(
            registry=registry,
            policy=get_scientific_claim_extractor_policy(),
        )
        actor = load_relation_review_actor()
        if args.command == "register":
            value = service.register(
                actor=actor,
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                extractor_version=args.extractor_version,
                extractor_kind=args.extractor_kind,
                implementation_sha256=args.implementation_sha256,
                configuration_sha256=args.configuration_sha256,
                supported_claim_types=tuple(args.claim_type),
                supported_modalities=tuple(args.modality),
                supported_languages=tuple(args.language),
            )
            _print(
                {
                    **_summary(value),
                    "mutation_performed": True,
                    "registration_performed": True,
                    "retirement_performed": False,
                }
            )
            return 0
        if args.command == "retire":
            current = registry.get(
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                extractor_version=args.extractor_version,
            )
            if current.record_digest != args.confirm_record_digest:
                raise ValueError("extractor record confirmation differs from current record.")
            value = service.retire(
                actor=actor,
                owner_id=args.owner_id,
                extractor_name=args.extractor_name,
                extractor_version=args.extractor_version,
            )
            _print(
                {
                    **_summary(value),
                    "mutation_performed": True,
                    "registration_performed": False,
                    "retirement_performed": True,
                }
            )
            return 0
        raise ValueError("unsupported claim extractor registry command.")
    except (KeyError, OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
