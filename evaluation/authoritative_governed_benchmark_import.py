"""Authoritative publication for already-local governed benchmark sources.

The large benchmark-import module owns semantic adapters and declarative field mappings.  This
module owns the final publication boundary: logical split names are never filesystem
components, example identifiers are globally unique across splits, identity digests are
computed with a disk-backed ledger, and the complete result is staged before one directory
rename.  No dataset download, model execution or training occurs on import.
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
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetManifest, SplitManifest
from evaluation.governed_benchmark_import import (
    BenchmarkGovernanceSpec,
    BenchmarkSplitImportSpec,
    GovernedBenchmarkImportReceipt,
    ImportedSplitReceipt,
    _adapt,
    _benchmark_payload,
    _iter_rows,
)
from training.advanced_path_authority import safe_advanced_path
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


def _selected_splits(
    splits: Sequence[BenchmarkSplitImportSpec],
) -> tuple[BenchmarkSplitImportSpec, ...]:
    selected = tuple(splits)
    if not selected or len(selected) > _MAX_SPLITS:
        raise ValueError(f"benchmark splits must contain 1..{_MAX_SPLITS} entries")
    if any(not isinstance(item, BenchmarkSplitImportSpec) for item in selected):
        raise ValueError("benchmark splits must contain BenchmarkSplitImportSpec values")
    if len({item.name for item in selected}) != len(selected):
        raise ValueError("benchmark split names must be unique")
    return tuple(sorted(selected, key=lambda item: item.name))


def _open_ledger(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE records (id TEXT PRIMARY KEY, split_name TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE documents (split_name TEXT NOT NULL, id TEXT NOT NULL, "
        "PRIMARY KEY(split_name,id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE source_groups (split_name TEXT NOT NULL, id TEXT NOT NULL, "
        "PRIMARY KEY(split_name,id)) WITHOUT ROWID"
    )
    return connection


def _insert_record(
    connection: sqlite3.Connection,
    *,
    split_name: str,
    example_id: str,
) -> None:
    try:
        connection.execute(
            "INSERT INTO records(id,split_name) VALUES (?,?)",
            (example_id, split_name),
        )
    except sqlite3.IntegrityError as exc:
        row = connection.execute(
            "SELECT split_name FROM records WHERE id=?",
            (example_id,),
        ).fetchone()
        previous = None if row is None else str(row[0])
        if previous == split_name:
            raise ValueError(
                f"benchmark split {split_name!r} contains duplicate example id {example_id!r}"
            ) from exc
        raise ValueError(
            f"benchmark example id {example_id!r} leaks across splits {previous!r}/{split_name!r}"
        ) from exc


def _insert_optional(
    connection: sqlite3.Connection,
    *,
    table: str,
    split_name: str,
    value: str,
) -> None:
    if table not in {"documents", "source_groups"}:
        raise ValueError("unsupported benchmark identity table")
    connection.execute(
        f"INSERT OR IGNORE INTO {table}(split_name,id) VALUES (?,?)",
        (split_name, value),
    )


def _sorted_digest(
    connection: sqlite3.Connection,
    *,
    table: str,
    split_name: str,
) -> str | None:
    if table not in {"records", "documents", "source_groups"}:
        raise ValueError("unsupported benchmark identity table")
    count = int(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE split_name=?",
            (split_name,),
        ).fetchone()[0]
    )
    if count == 0 and table != "records":
        return None
    digest = hashlib.sha256()
    for (value,) in connection.execute(
        f"SELECT id FROM {table} WHERE split_name=? ORDER BY id COLLATE BINARY",
        (split_name,),
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _publish_split(
    spec: BenchmarkSplitImportSpec,
    *,
    dataset_id: str,
    stage: Path,
    ledger: sqlite3.Connection,
) -> ImportedSplitReceipt:
    filename = logical_filename(spec.name, ".benchmark.jsonl")
    destination = stage / filename
    output_digest = hashlib.sha256()
    count = 0
    with destination.open("xb") as handle:
        for ordinal, row in enumerate(_iter_rows(spec), start=1):
            if count >= _MAX_RECORDS:
                raise ValueError(f"split {spec.name} exceeds record safety bound")
            example = _adapt(spec, row, ordinal=ordinal, dataset_id=dataset_id)
            _insert_record(
                ledger,
                split_name=spec.name,
                example_id=example.example_id,
            )
            for document_id in example.relevant_ids:
                _insert_optional(
                    ledger,
                    table="documents",
                    split_name=spec.name,
                    value=document_id,
                )
            source_group = (
                example.metadata.get("source_group_id")
                if isinstance(example.metadata, Mapping)
                else None
            )
            if isinstance(source_group, str) and source_group.strip():
                _insert_optional(
                    ledger,
                    table="source_groups",
                    split_name=spec.name,
                    value=source_group.strip(),
                )
            payload = _canonical(_benchmark_payload(example)) + b"\n"
            if len(payload) > _MAX_LINE_BYTES:
                raise ValueError(
                    f"canonical split {spec.name} record exceeds byte safety bound"
                )
            handle.write(payload)
            output_digest.update(payload)
            count += 1
            if count % 10_000 == 0:
                ledger.commit()
        handle.flush()
        os.fsync(handle.fileno())
    if spec.expected_record_count is not None and count != spec.expected_record_count:
        raise ValueError(
            f"split {spec.name} record count differs from configured expected_record_count"
        )
    if count <= 0:
        raise ValueError(f"benchmark split {spec.name!r} is empty")
    ledger.commit()
    output_sha = output_digest.hexdigest()
    if _stream_sha(destination) != output_sha:
        raise RuntimeError("canonical benchmark split changed during staged publication")
    record_digest = _sorted_digest(ledger, table="records", split_name=spec.name)
    assert record_digest is not None
    return ImportedSplitReceipt(
        name=spec.name,
        source_sha256=spec.source_sha256,
        output_path=str(destination),
        output_sha256=output_sha,
        record_count=count,
        record_id_sha256=record_digest,
        query_id_sha256=record_digest,
        document_id_sha256=_sorted_digest(
            ledger,
            table="documents",
            split_name=spec.name,
        ),
        source_group_sha256=_sorted_digest(
            ledger,
            table="source_groups",
            split_name=spec.name,
        ),
        transformation_component_sha256=spec.transformation_component_sha256,
    )


def _closed_stage(stage: Path, receipts: Sequence[ImportedSplitReceipt]) -> None:
    expected = {Path(item.output_path).name for item in receipts} | {
        "dataset_manifest.json",
        "import_receipt.json",
    }
    actual = {item.name for item in stage.iterdir()}
    if actual != expected:
        raise RuntimeError(
            "authoritative benchmark publication directory is not closed: "
            f"unexpected={sorted(actual-expected)} missing={sorted(expected-actual)}"
        )
    for child in stage.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError("authoritative benchmark publication contains a non-regular child")


def import_authoritative_governed_benchmark(
    governance: BenchmarkGovernanceSpec,
    splits: Sequence[BenchmarkSplitImportSpec],
    *,
    output_dir: str | Path,
) -> tuple[DatasetManifest, GovernedBenchmarkImportReceipt]:
    """Publish exact local benchmark sources through the authoritative v2 boundary."""
    if not isinstance(governance, BenchmarkGovernanceSpec):
        raise ValueError("governance must be BenchmarkGovernanceSpec")
    selected = _selected_splits(splits)
    root = safe_advanced_path(
        output_dir,
        label="authoritative benchmark import output",
        must_exist=False,
    )
    if root.exists():
        raise ValueError("authoritative benchmark import output must not already exist")
    parent = safe_advanced_path(
        root.parent,
        label="authoritative benchmark import parent",
        must_exist=True,
        require_directory=True,
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name or 'benchmark'}-stage-", dir=parent)
    )
    ledger_path = stage / ".identity-ledger.sqlite3"
    ledger: sqlite3.Connection | None = None
    try:
        ledger = _open_ledger(ledger_path)
        receipts = tuple(
            _publish_split(
                spec,
                dataset_id=governance.dataset_id,
                stage=stage,
                ledger=ledger,
            )
            for spec in selected
        )
        ledger.commit()
        ledger.close()
        ledger = None
        ledger_path.unlink()

        source_set_sha = _digest(
            {
                "schema": "rigorousrag-benchmark-source-set/v2",
                "dataset_id": governance.dataset_id,
                "exact_version": governance.exact_version,
                "splits": [
                    {"name": item.name, "source_sha256": item.source_sha256}
                    for item in receipts
                ],
            }
        )
        transformation_sha = _digest(
            {
                "schema": "rigorousrag-governed-benchmark-import-transformation/v2",
                "loader_name": "evaluation.authoritative_governed_benchmark_import",
                "loader_version": "2",
                "split_order": "logical_name_binary_sort",
                "filename_policy": "sha256(logical_name)+fixed_extension",
                "record_identity_policy": "global_example_id_uniqueness_sqlite",
                "publication_policy": "staged_closed_directory_rename",
                "split_transformations": [
                    {
                        "name": item.name,
                        "sha256": item.transformation_component_sha256,
                    }
                    for item in receipts
                ],
            }
        )
        manifest = DatasetManifest(
            dataset_id=governance.dataset_id,
            exact_version=governance.exact_version,
            source_locator=governance.source_locator,
            artifact_sha256=source_set_sha,
            license_identifier=governance.license_identifier,
            license_status=governance.license_status,
            license_evidence=governance.license_evidence,
            loader_name="evaluation.authoritative_governed_benchmark_import",
            loader_version="2",
            transformation_sha256=transformation_sha,
            splits=tuple(
                SplitManifest(
                    name=item.name,
                    content_sha256=item.output_sha256,
                    record_count=item.record_count,
                    record_id_sha256=item.record_id_sha256,
                    source_group_sha256=item.source_group_sha256,
                    query_id_sha256=item.query_id_sha256,
                    document_id_sha256=item.document_id_sha256,
                )
                for item in receipts
            ),
            tasks=governance.tasks,
            modalities=governance.modalities,
            card=governance.card,
            metadata={
                **governance.metadata,
                "canonical_record_schema": "rigorousrag-benchmark-example/v1",
                "publication_authority": "authoritative_governed_benchmark_import/v2",
                "filename_policy": "sha256(logical_name)+fixed_extension",
            },
        )
        if governance.require_promotable:
            manifest.assert_promotable()

        final_receipts = tuple(
            ImportedSplitReceipt(
                name=item.name,
                source_sha256=item.source_sha256,
                output_path=str(root / Path(item.output_path).name),
                output_sha256=item.output_sha256,
                record_count=item.record_count,
                record_id_sha256=item.record_id_sha256,
                query_id_sha256=item.query_id_sha256,
                document_id_sha256=item.document_id_sha256,
                source_group_sha256=item.source_group_sha256,
                transformation_component_sha256=item.transformation_component_sha256,
            )
            for item in receipts
        )
        manifest_path = root / "dataset_manifest.json"
        _atomic(
            stage / "dataset_manifest.json",
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
            "schema": "rigorousrag-governed-benchmark-import-receipt/v1",
            "dataset_manifest_sha256": manifest.manifest_digest,
            "dataset_artifact_sha256": source_set_sha,
            "transformation_sha256": transformation_sha,
            "manifest_path": str(manifest_path),
            "split_receipts": [asdict(item) for item in final_receipts],
        }
        receipt = GovernedBenchmarkImportReceipt(
            dataset_manifest_sha256=manifest.manifest_digest,
            dataset_artifact_sha256=source_set_sha,
            transformation_sha256=transformation_sha,
            manifest_path=str(manifest_path),
            split_receipts=final_receipts,
            receipt_sha256=_digest(unsigned),
        )
        _atomic(
            stage / "import_receipt.json",
            _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n",
        )
        _closed_stage(stage, receipts)
        os.replace(stage, root)
        for item in final_receipts:
            path = safe_advanced_path(
                item.output_path,
                label=f"published benchmark split {item.name}",
                must_exist=True,
                require_file=True,
            )
            if _stream_sha(path) != item.output_sha256:
                raise RuntimeError(
                    f"benchmark split {item.name!r} changed during final publication"
                )
        return manifest, receipt
    except Exception:
        if ledger is not None:
            ledger.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["import_authoritative_governed_benchmark"]
