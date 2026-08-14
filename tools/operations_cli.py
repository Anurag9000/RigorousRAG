"""Operator CLI for deterministic DR, canary, and release inventory workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from tools.disaster_recovery import (
    BackupEntry,
    BackupManifest,
    CanaryObservation,
    CanaryPolicy,
    create_backup,
    evaluate_canary,
    restore_backup,
    verify_backup,
)
from tools.release_inventory import (
    build_cyclonedx_sbom,
    build_spdx_sbom,
    load_pip_list,
    reproducible_timestamp,
    write_canonical_json,
    write_provenance,
)
from tools.release_supply_chain import ReleaseProvenance, sha256_file


def _manifest(path: str | Path) -> BackupManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("backup manifest must be a JSON object.")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ValueError("backup manifest entries must be an array of objects.")
    return BackupManifest(
        schema=str(raw.get("schema", "")),
        generation=str(raw.get("generation", "")),
        entries=tuple(
            BackupEntry(
                name=str(item["name"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in entries
        ),
        encryption_key_id=(
            None if raw.get("encryption_key_id") is None else str(raw["encryption_key_id"])
        ),
    )


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False))


def _backup_create(args: argparse.Namespace) -> int:
    manifest = create_backup(
        sources=args.source,
        destination=args.destination,
        generation=args.generation,
        encryption_key_id=args.encryption_key_id,
    )
    _print(asdict(manifest))
    return 0


def _backup_verify(args: argparse.Namespace) -> int:
    manifest = _manifest(args.manifest)
    valid = verify_backup(source=args.source, manifest=manifest)
    _print({"valid": valid})
    return 0 if valid else 2


def _backup_restore(args: argparse.Namespace) -> int:
    manifest = _manifest(args.manifest)
    report = restore_backup(source=args.source, destination=args.destination, manifest=manifest)
    _print(asdict(report))
    return 0


def _canary(args: argparse.Namespace) -> int:
    observation = CanaryObservation(
        requests=args.requests,
        errors=args.errors,
        baseline_p95_latency_ms=args.baseline_p95_latency_ms,
        canary_p95_latency_ms=args.canary_p95_latency_ms,
        quality_delta=args.quality_delta,
    )
    policy = CanaryPolicy(
        max_error_rate=args.max_error_rate,
        max_p95_latency_ratio=args.max_p95_latency_ratio,
        min_quality_delta=args.min_quality_delta,
    )
    decision = evaluate_canary(observation, policy)
    _print(asdict(decision))
    return 0 if decision.promote else 3


def _inventory(args: argparse.Namespace) -> int:
    records = load_pip_list(args.pip_list)
    if args.format == "cyclonedx":
        document = build_cyclonedx_sbom(records)
    else:
        if args.source_date_epoch is None:
            raise ValueError("SPDX inventory requires --source-date-epoch or SOURCE_DATE_EPOCH.")
        document = build_spdx_sbom(
            records,
            namespace=args.namespace,
            created=reproducible_timestamp(args.source_date_epoch),
        )
    digest = write_canonical_json(args.output, document)
    _print({"output": str(args.output), "sha256": digest, "component_count": len(records)})
    return 0


def _provenance(args: argparse.Namespace) -> int:
    provenance = ReleaseProvenance(
        commit_sha=args.commit_sha,
        dependency_lock_sha256=sha256_file(args.dependency_lock),
        sbom_sha256=sha256_file(args.sbom),
        artifact_sha256=args.artifact_sha256,
        image_digest=args.image_digest,
        workflow=args.workflow,
        run_id=args.run_id,
    )
    digest = write_provenance(args.output, provenance)
    _print({"output": str(args.output), "sha256": digest})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rigorousrag-operations")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("backup-create")
    create.add_argument("--source", action="append", required=True)
    create.add_argument("--destination", required=True)
    create.add_argument("--generation", required=True)
    create.add_argument("--encryption-key-id")
    create.set_defaults(handler=_backup_create)

    verify = commands.add_parser("backup-verify")
    verify.add_argument("--source", required=True)
    verify.add_argument("--manifest", required=True)
    verify.set_defaults(handler=_backup_verify)

    restore = commands.add_parser("backup-restore")
    restore.add_argument("--source", required=True)
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--destination", required=True)
    restore.set_defaults(handler=_backup_restore)

    canary = commands.add_parser("canary-evaluate")
    canary.add_argument("--requests", type=int, required=True)
    canary.add_argument("--errors", type=int, required=True)
    canary.add_argument("--baseline-p95-latency-ms", type=float, required=True)
    canary.add_argument("--canary-p95-latency-ms", type=float, required=True)
    canary.add_argument("--quality-delta", type=float, required=True)
    canary.add_argument("--max-error-rate", type=float, default=0.01)
    canary.add_argument("--max-p95-latency-ratio", type=float, default=1.20)
    canary.add_argument("--min-quality-delta", type=float, default=-0.005)
    canary.set_defaults(handler=_canary)

    inventory = commands.add_parser("inventory-build")
    inventory.add_argument("--pip-list", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--format", choices=("cyclonedx", "spdx"), default="cyclonedx")
    inventory.add_argument(
        "--namespace",
        default="https://github.com/Anurag9000/RigorousRAG/releases/sbom",
    )
    inventory.add_argument("--source-date-epoch", default=os.getenv("SOURCE_DATE_EPOCH"))
    inventory.set_defaults(handler=_inventory)

    provenance = commands.add_parser("provenance-build")
    provenance.add_argument("--commit-sha", required=True)
    provenance.add_argument("--dependency-lock", required=True)
    provenance.add_argument("--sbom", required=True)
    provenance.add_argument("--artifact-sha256", required=True)
    provenance.add_argument("--image-digest")
    provenance.add_argument("--workflow", required=True)
    provenance.add_argument("--run-id", required=True)
    provenance.add_argument("--output", required=True)
    provenance.set_defaults(handler=_provenance)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
