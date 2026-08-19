"""Authoritative path-safe governed grounded dataset publication.

This module is the final source authority for converting already-local reviewed JSON/JSONL
annotations into governed grounded-generation training splits.  It intentionally reuses the
existing declarative row-adaptation contracts from :mod:`training.governed_grounded_import`
but owns publication itself so logical split names never become filesystem components and
corpus-sized identifier sets are never retained in Python memory.

The publication is one-shot and staged: the requested output directory must not already exist;
all split bytes, manifests and receipts are produced in a sibling staging directory and the
complete closed artifact is renamed into place only after every digest and identity invariant
has been proved.  Importing this module performs no dataset download, model execution or
training.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.dataset_governance import DatasetManifest, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import parse_authoritative_grounded_example
from training.governed_grounded_import import (
    GovernedGroundedImportReceipt,
    GroundedDatasetGovernanceSpec,
    GroundedSplitImportReceipt,
    GroundedSplitImportSpec,
    _iter_rows,
)
from training.logical_filename import logical_filename

_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_SPLITS = 100


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_governance(governance: GroundedDatasetGovernanceSpec) -> None:
    if not isinstance(governance, GroundedDatasetGovernanceSpec):
        raise ValueError("governance must be GroundedDatasetGovernanceSpec")
    for name in (
        "dataset_id",
        "exact_version",
        "source_locator",
        "license_identifier",
        "license_evidence",
    ):
        value = getattr(governance, name)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"governance.{name} must be non-empty bounded text")
    if not governance.tasks or not governance.modalities:
        raise ValueError("governance tasks/modalities must be non-empty")
    if not isinstance(governance.metadata, Mapping):
        raise ValueError("governance.metadata must be a mapping")
    if not isinstance(governance.require_promotable, bool):
        raise ValueError("governance.require_promotable must be boolean")


def _selected_splits(splits: Sequence[GroundedSplitImportSpec]) -> tuple[GroundedSplitImportSpec, ...]:
    selected = tuple(splits)
    if not selected or len(selected) > _MAX_SPLITS:
        raise ValueError(f"splits must contain 1..{_MAX_SPLITS} entries")
    if any(not isinstance(item, GroundedSplitImportSpec) for item in selected):
        raise ValueError("splits must contain GroundedSplitImportSpec values")
    names = [item.name for item in selected]
    if len(set(names)) != len(names):
        raise ValueError("grounded split names must be unique")
    # Canonical ordering makes manifest/source identities independent of config list order.
    return tuple(sorted(selected, key=lambda item: item.name))


def _open_ledger(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE record_ids (id TEXT PRIMARY KEY, split_name TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE evidence_ids (split_name TEXT NOT NULL, id TEXT NOT NULL, "
        "PRIMARY KEY(split_name,id)) WITHOUT ROWID"
    )
    return connection


def _insert_record(connection: sqlite3.Connection, *, split_name: str, example_id: str) -> None:
    try:
        connection.execute(
            "INSERT INTO record_ids(id,split_name) VALUES (?,?)",
            (example_id, split_name),
        )
    except sqlite3.IntegrityError as exc:
        row = connection.execute(
            "SELECT split_name FROM record_ids WHERE id=?",
            (example_id,),
        ).fetchone()
        previous = None if row is None else str(row[0])
        if previous == split_name:
            raise ValueError(f"duplicate grounded example id {example_id!r} in split {split_name!r}") from exc
        raise ValueError(
            f"grounded example id {example_id!r} leaks across splits {previous!r}/{split_name!r}"
        ) from exc


def _insert_evidence(connection: sqlite3.Connection, *, split_name: str, evidence_id: str) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO evidence_ids(split_name,id) VALUES (?,?)",
        (split_name, evidence_id),
    )


def _sorted_id_digest(
    connection: sqlite3.Connection,
    *,
    table: str,
    split_name: str,
) -> str:
    if table not in {"record_ids", "evidence_ids"}:
        raise ValueError("unsupported identity table")
    digest = hashlib.sha256()
    query = (
        f"SELECT id FROM {table} WHERE split_name=? ORDER BY id COLLATE BINARY"
    )
    for (value,) in connection.execute(query, (split_name,)):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publish_split(
    spec: GroundedSplitImportSpec,
    *,
    dataset_id: str,
    stage: Path,
    ledger: sqlite3.Connection,
) -> GroundedSplitImportReceipt:
    filename = logical_filename(spec.name, ".grounded.jsonl")
    destination = stage / filename
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{filename}-",
        suffix=".tmp",
        dir=stage,
    )
    output_digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for ordinal, row in enumerate(_iter_rows(spec), start=1):
                if count >= _MAX_RECORDS:
                    raise ValueError("grounded split exceeds record safety bound")
                payload = spec.profile.adapt(
                    row,
                    dataset_id=dataset_id,
                    split_name=spec.name,
                    ordinal=ordinal,
                )
                # Reparse the exact canonical output so adaptation cannot widen the authority schema.
                parse_authoritative_grounded_example(payload)
                example_id = str(payload["example_id"])
                _insert_record(ledger, split_name=spec.name, example_id=example_id)
                evidence = payload.get("evidence")
                if not isinstance(evidence, list):
                    raise ValueError("adapted grounded evidence must be a list")
                for item in evidence:
                    if not isinstance(item, Mapping) or "evidence_id" not in item:
                        raise ValueError("adapted grounded evidence entry is malformed")
                    _insert_evidence(
                        ledger,
                        split_name=spec.name,
                        evidence_id=str(item["evidence_id"]),
                    )
                encoded = _canonical(payload) + b"\n"
                if len(encoded) > _MAX_LINE_BYTES:
                    raise ValueError("canonical grounded row exceeds line safety bound")
                handle.write(encoded)
                output_digest.update(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if spec.expected_record_count is not None and count != spec.expected_record_count:
            raise ValueError("grounded split record count differs from expected_record_count")
        if count <= 0:
            raise ValueError(f"grounded split {spec.name!r} is empty")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    output_sha = output_digest.hexdigest()
    if _stream_sha(destination) != output_sha:
        raise RuntimeError("grounded split changed during authoritative publication")
    ledger.commit()
    return GroundedSplitImportReceipt(
        spec.name,
        spec.source_sha256,
        str(destination),
        output_sha,
        count,
        _sorted_id_digest(ledger, table="record_ids", split_name=spec.name),
        _sorted_id_digest(ledger, table="evidence_ids", split_name=spec.name),
        spec.transformation_sha256,
    )


def _closed_stage(stage: Path, receipts: Iterable[GroundedSplitImportReceipt]) -> None:
    expected = {
        Path(item.output_path).name for item in receipts
    } | {"dataset_manifest.json", "import_receipt.json"}
    actual = {item.name for item in stage.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"authoritative grounded publication directory is not closed: "
            f"unexpected={sorted(actual-expected)} missing={sorted(expected-actual)}"
        )
    for item in stage.iterdir():
        if item.is_symlink() or not item.is_file():
            raise RuntimeError("authoritative grounded publication contains a non-regular child")


def import_authoritative_governed_grounded_dataset(
    governance: GroundedDatasetGovernanceSpec,
    splits: Sequence[GroundedSplitImportSpec],
    *,
    output_dir: str | Path,
) -> tuple[DatasetManifest, GovernedGroundedImportReceipt]:
    """Publish one immutable governed grounded dataset using the authoritative v2 path."""
    _validate_governance(governance)
    selected = _selected_splits(splits)

    root = safe_advanced_path(
        output_dir,
        label="authoritative grounded import output",
        must_exist=False,
    )
    if root.exists():
        raise ValueError("authoritative grounded import output must not already exist")
    parent = safe_advanced_path(
        root.parent,
        label="authoritative grounded import parent",
        must_exist=True,
        require_directory=True,
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name or 'grounded'}-stage-", dir=parent)
    )
    ledger_path = stage / ".identity-ledger.sqlite3"
    ledger: sqlite3.Connection | None = None
    try:
        ledger = _open_ledger(ledger_path)
        receipts = tuple(
            _publish_split(
                item,
                dataset_id=governance.dataset_id,
                stage=stage,
                ledger=ledger,
            )
            for item in selected
        )
        ledger.commit()
        ledger.close()
        ledger = None
        ledger_path.unlink()

        source_set = _digest(
            {
                "schema": "rigorousrag-grounded-source-set/v2",
                "dataset_id": governance.dataset_id,
                "exact_version": governance.exact_version,
                "splits": [
                    {"name": item.name, "sha256": item.source_sha256}
                    for item in receipts
                ],
            }
        )
        transformation = _digest(
            {
                "schema": "rigorousrag-governed-grounded-transformation/v2",
                "loader": "training.authoritative_governed_grounded_import",
                "version": "2",
                "split_order": "logical_name_binary_sort",
                "filename_policy": "sha256(logical_name)+fixed_extension",
                "record_identity_policy": "global_example_id_uniqueness_sqlite",
                "splits": [
                    {"name": item.name, "sha256": item.transformation_sha256}
                    for item in receipts
                ],
            }
        )
        manifest = DatasetManifest(
            dataset_id=governance.dataset_id,
            exact_version=governance.exact_version,
            source_locator=governance.source_locator,
            artifact_sha256=source_set,
            license_identifier=governance.license_identifier,
            license_status=governance.license_status,
            license_evidence=governance.license_evidence,
            loader_name="training.authoritative_governed_grounded_import",
            loader_version="2",
            transformation_sha256=transformation,
            splits=tuple(
                SplitManifest(
                    name=item.name,
                    content_sha256=item.output_sha256,
                    record_count=item.record_count,
                    record_id_sha256=item.record_id_sha256,
                    query_id_sha256=item.record_id_sha256,
                    document_id_sha256=item.evidence_id_sha256,
                )
                for item in receipts
            ),
            tasks=governance.tasks,
            modalities=governance.modalities,
            card=governance.card,
            metadata={
                **governance.metadata,
                "canonical_record_kind": "grounded_generation",
                "canonical_parser": "parse_authoritative_grounded_example",
                "publication_authority": "authoritative_governed_grounded_import/v2",
                "filename_policy": "sha256(logical_name)+fixed_extension",
            },
        )
        if governance.require_promotable:
            manifest.assert_promotable()

        manifest_path = stage / "dataset_manifest.json"
        _atomic(
            manifest_path,
            _canonical(
                {
                    "schema": "rigorousrag-dataset-manifest/v1",
                    "manifest": asdict(manifest),
                    "manifest_sha256": manifest.manifest_digest,
                }
            )
            + b"\n",
        )
        unsigned = {
            "schema": "rigorousrag-governed-grounded-import-receipt/v1",
            "dataset_manifest_sha256": manifest.manifest_digest,
            "source_set_sha256": source_set,
            "transformation_sha256": transformation,
            "manifest_path": str(root / "dataset_manifest.json"),
            "splits": [
                {
                    **asdict(item),
                    "output_path": str(root / Path(item.output_path).name),
                }
                for item in receipts
            ],
        }
        final_receipts = tuple(
            GroundedSplitImportReceipt(
                item.name,
                item.source_sha256,
                str(root / Path(item.output_path).name),
                item.output_sha256,
                item.record_count,
                item.record_id_sha256,
                item.evidence_id_sha256,
                item.transformation_sha256,
            )
            for item in receipts
        )
        receipt = GovernedGroundedImportReceipt(
            manifest.manifest_digest,
            source_set,
            transformation,
            str(root / "dataset_manifest.json"),
            final_receipts,
            _digest(unsigned),
        )
        _atomic(
            stage / "import_receipt.json",
            _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n",
        )
        # The stage receipt contains final paths by design; prove the staged bytes are otherwise closed.
        _closed_stage(stage, receipts)
        os.replace(stage, root)

        # Re-hash every published split after the final rename before returning authority.
        for item in final_receipts:
            path = safe_advanced_path(
                item.output_path,
                label=f"authoritative grounded split {item.name}",
                must_exist=True,
                require_file=True,
            )
            if _stream_sha(path) != item.output_sha256:
                raise RuntimeError(f"grounded split {item.name!r} changed during final publication")
        return manifest, receipt
    except Exception:
        if ledger is not None:
            ledger.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["import_authoritative_governed_grounded_dataset"]
