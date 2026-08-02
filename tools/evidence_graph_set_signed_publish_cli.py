"""Immediate publication CLI with signed actor-use provenance enforcement."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_relation_actor_use_runtime import get_signed_actor_use_store
from tools.evidence_graph_relation_authorization_runtime import (
    get_relation_review_authorization_store,
)
from tools.evidence_graph_relation_runtime import get_relation_review_ledger
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_publish import EvidenceGraphSetPublishError
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_signed_actor_provenance_boundary import (
    publish_signed_actor_governed_graph_set,
)
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_publish_cli",
        description=(
            "Publish approved reviewed relations while validating committed reviewer "
            "authorization receipts and any signed actor-use provenance."
        ),
    )
    publish = parser.add_subparsers(dest="command", required=True).add_parser(
        "publish-approved"
    )
    publish.add_argument("--owner-id", required=True)
    publish.add_argument("--graph-set-key", required=True)
    publish.add_argument("--proposal-id", action="append", required=True)
    expectation = publish.add_mutually_exclusive_group(required=True)
    expectation.add_argument("--expect-no-current", action="store_true")
    expectation.add_argument("--expected-current-set-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "publish-approved":
            raise ValueError("unsupported signed graph-set publication command.")
        expected = None if args.expect_no_current else args.expected_current_set_id
        result = publish_signed_actor_governed_graph_set(
            owner_id=args.owner_id,
            graph_set_key=args.graph_set_key,
            proposal_ids=args.proposal_id,
            expected_current_set_id=expected,
            ledger=get_relation_review_ledger(),
            authorization_store=get_relation_review_authorization_store(),
            actor_use_store=get_signed_actor_use_store(),
            set_store=get_evidence_graph_set_store(),
            generations=get_generation_store(),
            graphs=get_evidence_graph_store(),
        )
        payload = asdict(result)
        payload.update(
            {
                "semantic_inference_performed": False,
                "automatic_approval_performed": False,
                "reviewed_proposals_required": True,
                "committed_review_authorizations_required": True,
                "signed_actor_use_provenance_validated": True,
                "source_text_returned": False,
            }
        )
        _print(payload)
        return 0
    except EvidenceGraphSetPublishError as exc:
        _print(
            {
                "error": "publication_failed",
                "compensation_complete": not bool(exc.compensation_errors),
                "compensation_errors": list(exc.compensation_errors),
                "signed_actor_use_provenance_validated": False,
            },
            stream=sys.stderr,
        )
        return 1
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
