"""Descriptor-safe verification boundary for signed retirement snapshots."""

from __future__ import annotations

import json
import os
import stat
from typing import Any

from tools.evidence_graph_set_signed_retirement_snapshot import (
    SignedRetirementSnapshot,
    _MAX_SNAPSHOT_BYTES,
    _attempt,
    _pairs,
    _path,
)


def _read_regular_snapshot(path: str | os.PathLike[str]) -> bytes:
    selected = _path(path, label="snapshot_path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("snapshot must be a regular file.")
        if before.st_size <= 0 or before.st_size > _MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot size is invalid.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("snapshot changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("snapshot grew while being read.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_size) != int(before.st_size)
        ):
            raise RuntimeError("snapshot identity changed while being read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_signed_retirement_snapshot(
    path: str | os.PathLike[str],
) -> SignedRetirementSnapshot:
    payload = _read_regular_snapshot(path)
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("snapshot JSON is invalid.") from exc
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "owner_id",
        "generated_at",
        "record_count",
        "records",
        "snapshot_digest",
        "contains_source_text",
        "contains_assertion_secrets",
        "journal_mutation_performed",
    }:
        raise ValueError("snapshot schema is invalid.")
    if (
        raw["contains_source_text"] is not False
        or raw["contains_assertion_secrets"] is not False
        or raw["journal_mutation_performed"] is not False
    ):
        raise ValueError("snapshot safety flags are invalid.")
    if not isinstance(raw["records"], list):
        raise ValueError("snapshot records must be a list.")
    records = tuple(_attempt(value) for value in raw["records"])
    return SignedRetirementSnapshot(
        owner_id=raw["owner_id"],
        generated_at=raw["generated_at"],
        record_count=raw["record_count"],
        records=records,
        snapshot_digest=raw["snapshot_digest"],
        schema_version=raw["schema_version"],
    )


__all__ = ["verify_signed_retirement_snapshot"]
