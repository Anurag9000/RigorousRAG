"""Strict v2 read authority for governed benchmark publications.

Historical v1 benchmark imports remain readable through ``evaluation.governed_benchmark_io``
for research reproducibility.  Promotion-capable code must use this module: it accepts only the
staged/path-safe authoritative v2 publisher, verifies the closed publication directory, hashes
all split bytes, and recomputes corpus identities with a disk-backed ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.dataset_governance import DatasetManifest
from evaluation.governed_benchmark_import import (
    GovernedBenchmarkImportReceipt,
    ImportedSplitReceipt,
)
from evaluation.governed_benchmark_io import _benchmark_example, _manifest
from tools.benchmark_adapters import BenchmarkExample
from training.advanced_path_authority import safe_advanced_path
from training.logical_filename import logical_filename

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_HEX = frozenset("0123456789abcdef")


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


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds JSON byte safety bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _open_ledger() -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="rigorousrag-benchmark-verify-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
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
    return connection, path


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
                f"authoritative benchmark split {split_name!r} contains duplicate example id {example_id!r}"
            ) from exc
        raise ValueError(
            f"authoritative benchmark example id {example_id!r} leaks across splits {previous!r}/{split_name!r}"
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


def _iter_split(
    path: Path,
    *,
    expected_sha256: str,
) -> Iterator[BenchmarkExample]:
    if _stream_sha(path) != expected_sha256:
        raise ValueError("authoritative benchmark split digest differs from receipt")
    count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(
                    f"canonical benchmark line {line_number} exceeds byte safety bound"
                )
            if count >= _MAX_RECORDS:
                raise ValueError("canonical benchmark split exceeds record safety bound")
            try:
                value = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                )
            except Exception as exc:
                raise ValueError(
                    f"canonical benchmark line {line_number} is not strict JSON"
                ) from exc
            count += 1
            yield _benchmark_example(value, line_number=line_number)


@dataclass(frozen=True)
class VerifiedAuthoritativeGovernedBenchmark:
    manifest: DatasetManifest
    receipt: GovernedBenchmarkImportReceipt
    root: str

    def split(self, name: str) -> Iterator[BenchmarkExample]:
        matches = [item for item in self.receipt.split_receipts if item.name == name]
        if len(matches) != 1:
            raise ValueError(f"unknown authoritative benchmark split {name!r}")
        item = matches[0]
        path = safe_advanced_path(
            item.output_path,
            label=f"authoritative benchmark split {item.name}",
            must_exist=True,
            require_file=True,
        )
        return _iter_split(path, expected_sha256=item.output_sha256)


def verify_authoritative_governed_benchmark_import(
    receipt_path: str | Path,
    *,
    require_promotable: bool = False,
) -> VerifiedAuthoritativeGovernedBenchmark:
    receipt_file = safe_advanced_path(
        receipt_path,
        label="authoritative benchmark import receipt",
        must_exist=True,
        require_file=True,
    )
    root = receipt_file.parent
    if receipt_file != root / "import_receipt.json":
        raise ValueError("authoritative benchmark receipt must use the canonical filename")
    raw = _read_json(receipt_file, "authoritative benchmark import receipt")
    expected_fields = {
        "schema",
        "dataset_manifest_sha256",
        "dataset_artifact_sha256",
        "transformation_sha256",
        "manifest_path",
        "split_receipts",
        "receipt_sha256",
    }
    if (
        set(raw) != expected_fields
        or raw.get("schema") != "rigorousrag-governed-benchmark-import-receipt/v1"
    ):
        raise ValueError("unsupported authoritative benchmark receipt schema")
    split_raw = raw.get("split_receipts")
    if not isinstance(split_raw, list) or not split_raw:
        raise ValueError("authoritative benchmark receipt requires split receipts")
    split_fields = {
        "name",
        "source_sha256",
        "output_path",
        "output_sha256",
        "record_count",
        "record_id_sha256",
        "query_id_sha256",
        "document_id_sha256",
        "source_group_sha256",
        "transformation_component_sha256",
    }
    receipts = []
    for item in split_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields:
            raise ValueError("authoritative benchmark split receipt fields are invalid")
        receipts.append(ImportedSplitReceipt(**dict(item)))
    receipt = GovernedBenchmarkImportReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"],
        dataset_artifact_sha256=raw["dataset_artifact_sha256"],
        transformation_sha256=raw["transformation_sha256"],
        manifest_path=raw["manifest_path"],
        split_receipts=tuple(receipts),
        receipt_sha256=raw["receipt_sha256"],
    )

    manifest_path = safe_advanced_path(
        receipt.manifest_path,
        label="authoritative benchmark dataset manifest",
        must_exist=True,
        require_file=True,
    )
    if manifest_path != root / "dataset_manifest.json":
        raise ValueError("authoritative benchmark manifest must be the canonical root child")
    envelope = _read_json(manifest_path, "authoritative benchmark dataset manifest")
    if (
        set(envelope) != {"schema", "manifest", "manifest_sha256"}
        or envelope.get("schema") != "rigorousrag-dataset-manifest/v1"
    ):
        raise ValueError("unsupported authoritative benchmark manifest envelope")
    manifest = _manifest(envelope["manifest"])
    if manifest.loader_name != "evaluation.authoritative_governed_benchmark_import":
        raise ValueError("benchmark was not produced by the authoritative v2 importer")
    if manifest.loader_version != "2":
        raise ValueError("authoritative benchmark loader_version must be 2")
    if manifest.metadata.get("publication_authority") != "authoritative_governed_benchmark_import/v2":
        raise ValueError("authoritative benchmark publication marker is missing")
    if (
        manifest.manifest_digest
        != _sha(envelope["manifest_sha256"], "manifest_sha256")
        or manifest.manifest_digest != receipt.dataset_manifest_sha256
    ):
        raise ValueError("authoritative benchmark manifest digest differs from receipt")
    if (
        manifest.artifact_sha256 != receipt.dataset_artifact_sha256
        or manifest.transformation_sha256 != receipt.transformation_sha256
    ):
        raise ValueError("authoritative benchmark source/transformation differs from receipt")
    if require_promotable:
        manifest.assert_promotable()

    manifest_by_name = {item.name: item for item in manifest.splits}
    if len(manifest_by_name) != len(manifest.splits):
        raise ValueError("authoritative benchmark manifest has duplicate split names")
    if set(manifest_by_name) != {item.name for item in receipt.split_receipts}:
        raise ValueError("authoritative benchmark manifest splits differ from receipt")

    expected_children = {"dataset_manifest.json", "import_receipt.json"}
    for item in receipt.split_receipts:
        filename = logical_filename(item.name, ".benchmark.jsonl")
        path = safe_advanced_path(
            item.output_path,
            label=f"authoritative benchmark split {item.name}",
            must_exist=True,
            require_file=True,
        )
        if path != root / filename:
            raise ValueError(
                f"authoritative benchmark split {item.name!r} has a non-canonical path"
            )
        expected_children.add(filename)
    actual_children = {item.name for item in root.iterdir()}
    if actual_children != expected_children:
        raise ValueError(
            "authoritative benchmark publication directory is not closed: "
            f"unexpected={sorted(actual_children-expected_children)} "
            f"missing={sorted(expected_children-actual_children)}"
        )
    for child in root.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("authoritative benchmark publication contains a non-regular child")

    connection, ledger_path = _open_ledger()
    try:
        total = 0
        for item in receipt.split_receipts:
            split_manifest = manifest_by_name[item.name]
            if (
                split_manifest.content_sha256 != item.output_sha256
                or split_manifest.record_count != item.record_count
                or split_manifest.record_id_sha256 != item.record_id_sha256
                or split_manifest.query_id_sha256 != item.query_id_sha256
                or split_manifest.document_id_sha256 != item.document_id_sha256
                or split_manifest.source_group_sha256 != item.source_group_sha256
            ):
                raise ValueError(
                    f"authoritative benchmark split {item.name} differs between manifest and receipt"
                )
            path = root / logical_filename(item.name, ".benchmark.jsonl")
            count = 0
            for example in _iter_split(path, expected_sha256=item.output_sha256):
                _insert_record(
                    connection,
                    split_name=item.name,
                    example_id=example.example_id,
                )
                for document_id in example.relevant_ids:
                    _insert_optional(
                        connection,
                        table="documents",
                        split_name=item.name,
                        value=document_id,
                    )
                source_group = (
                    example.metadata.get("source_group_id")
                    if isinstance(example.metadata, Mapping)
                    else None
                )
                if isinstance(source_group, str) and source_group.strip():
                    _insert_optional(
                        connection,
                        table="source_groups",
                        split_name=item.name,
                        value=source_group.strip(),
                    )
                count += 1
                total += 1
                if total % 10_000 == 0:
                    connection.commit()
            connection.commit()
            if count != item.record_count:
                raise ValueError(
                    f"authoritative benchmark split {item.name} record count differs from receipt"
                )
            record_digest = _sorted_digest(
                connection,
                table="records",
                split_name=item.name,
            )
            if record_digest != item.record_id_sha256 or record_digest != item.query_id_sha256:
                raise ValueError(
                    f"authoritative benchmark split {item.name} record/query digest differs from receipt"
                )
            if _sorted_digest(
                connection,
                table="documents",
                split_name=item.name,
            ) != item.document_id_sha256:
                raise ValueError(
                    f"authoritative benchmark split {item.name} document digest differs from receipt"
                )
            if _sorted_digest(
                connection,
                table="source_groups",
                split_name=item.name,
            ) != item.source_group_sha256:
                raise ValueError(
                    f"authoritative benchmark split {item.name} source-group digest differs from receipt"
                )
        if total <= 0:
            raise ValueError("authoritative benchmark contains no records")
        return VerifiedAuthoritativeGovernedBenchmark(
            manifest=manifest,
            receipt=receipt,
            root=str(root),
        )
    finally:
        connection.close()
        try:
            ledger_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "VerifiedAuthoritativeGovernedBenchmark",
    "verify_authoritative_governed_benchmark_import",
]
