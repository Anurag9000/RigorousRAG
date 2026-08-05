"""Strict local-fixture CLI for privacy-safe scientific claim evaluation."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from tools.evidence_graph_claim_contracts import (
    ClaimEvidenceLocator,
    ScientificClaimProposal,
)
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)
from tools.evidence_graph_claim_evaluation_verification import (
    verify_scientific_claim_evaluation_report,
)

_MAX_FIXTURE_BYTES = 20_000_000
_MAX_PATH = 4096
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("claim evaluation fixture contains a duplicate JSON key.")
        result[key] = value
    return result


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("fixture path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("fixture path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or bool(
            int(getattr(info, "st_file_attributes", 0)) & _REPARSE
        ):
            raise ValueError("fixture path may not contain redirects.")
    return absolute


def _read_fixture(value: str | os.PathLike[str]) -> dict[str, Any]:
    selected = _path(value)
    descriptor = os.open(
        selected,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= _MAX_FIXTURE_BYTES:
            raise ValueError("claim evaluation fixture is invalid or too large.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("claim evaluation fixture changed while reading.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("claim evaluation fixture grew while reading.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_size) != int(before.st_size)
        ):
            raise RuntimeError("claim evaluation fixture identity changed while reading.")
    finally:
        os.close(descriptor)
    try:
        raw = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("claim evaluation fixture JSON is invalid.") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "minimum_span_iou",
        "minimum_claim_token_f1",
        "gold",
        "proposals",
    }:
        raise ValueError("claim evaluation fixture schema is invalid.")
    if raw["schema_version"] != 1:
        raise ValueError("claim evaluation fixture schema is unsupported.")
    if not isinstance(raw["gold"], list) or not isinstance(raw["proposals"], list):
        raise ValueError("claim evaluation fixture arrays are invalid.")
    return raw


def _gold(raw: Any) -> ScientificClaimGold:
    if not isinstance(raw, dict) or set(raw) != {
        "gold_id",
        "owner_id",
        "doc_id",
        "generation",
        "content_sha256",
        "profile_fingerprint",
        "claim_text",
        "claim_type",
        "modality",
        "locator",
        "schema_version",
    }:
        raise ValueError("claim gold fixture schema is invalid.")
    locator = raw["locator"]
    if not isinstance(locator, dict):
        raise ValueError("claim gold locator is invalid.")
    value = dict(raw)
    value["locator"] = ClaimEvidenceLocator(**locator)
    return ScientificClaimGold(**value)


def _proposal(raw: Any) -> ScientificClaimProposal:
    if not isinstance(raw, dict):
        raise ValueError("claim proposal fixture must be an object.")
    value = dict(raw)
    locator = value.get("locator")
    if not isinstance(locator, dict):
        raise ValueError("claim proposal locator is invalid.")
    value["locator"] = ClaimEvidenceLocator(**locator)
    return ScientificClaimProposal(**value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_claim_evaluation_cli",
        description=(
            "Evaluate a strict local scientific-claim fixture. Output is text-free and "
            "does not evaluate semantic entailment."
        ),
    )
    parser.add_argument("fixture_path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        raw = _read_fixture(args.fixture_path)
        gold = tuple(_gold(value) for value in raw["gold"])
        proposals = tuple(_proposal(value) for value in raw["proposals"])
        report = evaluate_scientific_claim_extraction(
            gold=gold,
            proposals=proposals,
            minimum_span_iou=raw["minimum_span_iou"],
            minimum_claim_token_f1=raw["minimum_claim_token_f1"],
        )
        verify_scientific_claim_evaluation_report(
            report,
            minimum_span_iou=raw["minimum_span_iou"],
            minimum_claim_token_f1=raw["minimum_claim_token_f1"],
        )
        payload = asdict(report)
        payload["mutation_performed"] = False
        payload["source_text_returned"] = False
        _print(payload)
        return 0
    except (KeyError, OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
