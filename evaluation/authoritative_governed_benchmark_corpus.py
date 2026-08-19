"""Authoritative v2 corpus publication and verification for retrieval benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.governed_benchmark_corpus import (
    BenchmarkCorpusDocument,
    BenchmarkCorpusImportSpec,
    _rows,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_RECORDS = 200_000_000
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def _strict(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _open_ledger() -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(prefix="rigorousrag-corpus-ids-", suffix=".sqlite3")
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE documents (id TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE source_groups (id TEXT PRIMARY KEY) WITHOUT ROWID")
    return connection, path


def _insert_document(connection: sqlite3.Connection, document_id: str) -> None:
    try:
        connection.execute("INSERT INTO documents(id) VALUES (?)", (document_id,))
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate corpus document id {document_id!r}") from exc


def _digest_table(connection: sqlite3.Connection, table: str) -> str | None:
    if table not in {"documents", "source_groups"}:
        raise ValueError("unsupported corpus identity table")
    count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    if count == 0 and table == "source_groups":
        return None
    digest = hashlib.sha256()
    for (value,) in connection.execute(f"SELECT id FROM {table} ORDER BY id COLLATE BINARY"):
        digest.update(str(value).encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class AuthoritativeBenchmarkCorpusReceipt:
    source_path: str
    source_sha256: str
    profile_sha256: str
    transformation_sha256: str
    output_path: str
    output_sha256: str
    record_count: int
    document_id_sha256: str
    source_group_sha256: str | None
    publication_contract_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", str(safe_advanced_path(self.source_path, label="authoritative corpus source", must_exist=True, require_file=True)))
        object.__setattr__(self, "output_path", str(safe_advanced_path(self.output_path, label="authoritative corpus output", must_exist=True, require_file=True)))
        for name in ("source_sha256", "profile_sha256", "transformation_sha256", "output_sha256", "document_id_sha256", "publication_contract_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.source_group_sha256 is not None:
            object.__setattr__(self, "source_group_sha256", _sha(self.source_group_sha256, "source_group_sha256"))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative corpus receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-benchmark-corpus-receipt/v2",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "profile_sha256": self.profile_sha256,
            "transformation_sha256": self.transformation_sha256,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "record_count": self.record_count,
            "document_id_sha256": self.document_id_sha256,
            "source_group_sha256": self.source_group_sha256,
            "publication_contract_sha256": self.publication_contract_sha256,
        }


def _document_payload(document: BenchmarkCorpusDocument, *, source_sha256: str) -> Mapping[str, Any]:
    metadata = dict(document.metadata); metadata["source_sha256"] = source_sha256
    return {"schema": "rigorousrag-benchmark-corpus-document/v1", "document_id": document.document_id, "title": document.title, "text": document.text, "source_group_id": document.source_group_id, "metadata": metadata}


def _iter_documents(path: Path, *, expected_sha256: str) -> Iterator[BenchmarkCorpusDocument]:
    if _stream_sha(path) != expected_sha256:
        raise ValueError("authoritative corpus bytes differ from receipt")
    count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(f"corpus line {line_number} exceeds safety bound")
            if count >= _MAX_RECORDS:
                raise ValueError("corpus exceeds record safety bound")
            value = _strict(raw, f"corpus line {line_number}")
            required = {"schema", "document_id", "title", "text", "source_group_id", "metadata"}
            if not isinstance(value, Mapping) or set(value) != required or value.get("schema") != "rigorousrag-benchmark-corpus-document/v1":
                raise ValueError(f"corpus line {line_number} has unsupported schema")
            count += 1
            yield BenchmarkCorpusDocument(value["document_id"], value["text"], value["title"], value["source_group_id"], value["metadata"])


def publish_authoritative_benchmark_corpus(spec: BenchmarkCorpusImportSpec, *, output_dir: str | Path) -> AuthoritativeBenchmarkCorpusReceipt:
    if not isinstance(spec, BenchmarkCorpusImportSpec):
        raise ValueError("spec must be BenchmarkCorpusImportSpec")
    root = safe_advanced_path(output_dir, label="authoritative benchmark corpus output", must_exist=False)
    if root.exists():
        raise ValueError("authoritative benchmark corpus output must not already exist")
    parent = safe_advanced_path(root.parent, label="authoritative benchmark corpus parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'corpus'}-stage-", dir=parent))
    ledger, ledger_path = _open_ledger()
    try:
        corpus_path = stage / "corpus.jsonl"
        digest = hashlib.sha256(); count = 0
        with corpus_path.open("xb") as handle:
            for row in _rows(spec):
                if count >= _MAX_RECORDS:
                    raise ValueError("corpus exceeds record safety bound")
                document = spec.profile.adapt(row)
                _insert_document(ledger, document.document_id)
                if document.source_group_id is not None:
                    ledger.execute("INSERT OR IGNORE INTO source_groups(id) VALUES (?)", (document.source_group_id,))
                encoded = _canonical(_document_payload(document, source_sha256=spec.source_sha256)) + b"\n"
                if len(encoded) > _MAX_LINE_BYTES:
                    raise ValueError("canonical corpus row exceeds line safety bound")
                handle.write(encoded); digest.update(encoded); count += 1
                if count % 10_000 == 0:
                    ledger.commit()
            handle.flush(); os.fsync(handle.fileno())
        ledger.commit()
        if count <= 0:
            raise ValueError("corpus may not be empty")
        if spec.expected_record_count is not None and count != spec.expected_record_count:
            raise ValueError("corpus count differs from expected_record_count")
        output_sha = digest.hexdigest()
        if _stream_sha(corpus_path) != output_sha:
            raise RuntimeError("authoritative corpus changed during staged publication")
        document_sha = _digest_table(ledger, "documents"); assert document_sha is not None
        group_sha = _digest_table(ledger, "source_groups")
        publication_contract = _digest({
            "schema": "rigorousrag-authoritative-benchmark-corpus-publication/v2",
            "source_sha256": spec.source_sha256,
            "source_transform_sha256": spec.transformation_sha256,
            "profile_sha256": spec.profile.profile_sha256,
            "output_schema": "rigorousrag-benchmark-corpus-document/v1",
            "publication_policy": "staged_closed_directory_rename",
            "identity_policy": "sqlite_document_primary_key",
        })
        transformation = _digest({
            "schema": "rigorousrag-authoritative-benchmark-corpus-transform/v2",
            "source_transform_sha256": spec.transformation_sha256,
            "publication_contract_sha256": publication_contract,
        })
        final_output = root / "corpus.jsonl"
        unsigned = {
            "schema": "rigorousrag-authoritative-benchmark-corpus-receipt/v2",
            "source_path": spec.source_path,
            "source_sha256": spec.source_sha256,
            "profile_sha256": spec.profile.profile_sha256,
            "transformation_sha256": transformation,
            "output_path": str(final_output),
            "output_sha256": output_sha,
            "record_count": count,
            "document_id_sha256": document_sha,
            "source_group_sha256": group_sha,
            "publication_contract_sha256": publication_contract,
        }
        receipt_sha = _digest(unsigned)
        receipt_payload = {**unsigned, "receipt_sha256": receipt_sha}
        receipt_path = stage / "corpus_receipt.json"
        with receipt_path.open("xb") as handle:
            payload = _canonical(receipt_payload) + b"\n"; handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"corpus.jsonl", "corpus_receipt.json"}:
            raise RuntimeError("authoritative corpus staging directory is not closed")
        os.replace(stage, root)
        receipt = AuthoritativeBenchmarkCorpusReceipt(
            source_path=spec.source_path,
            source_sha256=spec.source_sha256,
            profile_sha256=spec.profile.profile_sha256,
            transformation_sha256=transformation,
            output_path=str(final_output),
            output_sha256=output_sha,
            record_count=count,
            document_id_sha256=document_sha,
            source_group_sha256=group_sha,
            publication_contract_sha256=publication_contract,
            receipt_sha256=receipt_sha,
        )
        return receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        ledger.close()
        try: ledger_path.unlink()
        except FileNotFoundError: pass


def verify_authoritative_benchmark_corpus_receipt(path: str | Path) -> AuthoritativeBenchmarkCorpusReceipt:
    receipt_path = safe_advanced_path(path, label="authoritative corpus receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / "corpus_receipt.json":
        raise ValueError("authoritative corpus receipt must use canonical root filename")
    if receipt_path.stat().st_size <= 0 or receipt_path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("authoritative corpus receipt exceeds safety bound")
    raw = _strict(receipt_path.read_bytes(), "authoritative corpus receipt")
    required = {"schema", "source_path", "source_sha256", "profile_sha256", "transformation_sha256", "output_path", "output_sha256", "record_count", "document_id_sha256", "source_group_sha256", "publication_contract_sha256", "receipt_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-benchmark-corpus-receipt/v2":
        raise ValueError("unsupported authoritative corpus receipt schema")
    receipt = AuthoritativeBenchmarkCorpusReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    output = safe_advanced_path(receipt.output_path, label="authoritative corpus output", must_exist=True, require_file=True)
    if output != root / "corpus.jsonl":
        raise ValueError("authoritative corpus output must be canonical root child")
    if {item.name for item in root.iterdir()} != {"corpus.jsonl", "corpus_receipt.json"}:
        raise ValueError("authoritative corpus directory is not closed")
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ValueError("authoritative corpus directory contains non-regular child")
    if _stream_sha(Path(receipt.source_path)) != receipt.source_sha256:
        raise ValueError("authoritative corpus source bytes changed after publication")
    ledger, ledger_path = _open_ledger()
    try:
        count = 0
        for document in _iter_documents(output, expected_sha256=receipt.output_sha256):
            _insert_document(ledger, document.document_id)
            if document.source_group_id is not None:
                ledger.execute("INSERT OR IGNORE INTO source_groups(id) VALUES (?)", (document.source_group_id,))
            count += 1
            if count % 10_000 == 0: ledger.commit()
        ledger.commit()
        document_sha = _digest_table(ledger, "documents"); assert document_sha is not None
        if count != receipt.record_count or document_sha != receipt.document_id_sha256 or _digest_table(ledger, "source_groups") != receipt.source_group_sha256:
            raise ValueError("authoritative corpus count/identity digests differ from receipt")
        return receipt
    finally:
        ledger.close()
        try: ledger_path.unlink()
        except FileNotFoundError: pass


def iter_authoritative_benchmark_corpus(receipt: AuthoritativeBenchmarkCorpusReceipt) -> Iterator[BenchmarkCorpusDocument]:
    if not isinstance(receipt, AuthoritativeBenchmarkCorpusReceipt):
        raise ValueError("receipt must be AuthoritativeBenchmarkCorpusReceipt")
    return _iter_documents(Path(receipt.output_path), expected_sha256=receipt.output_sha256)


__all__ = ["AuthoritativeBenchmarkCorpusReceipt", "iter_authoritative_benchmark_corpus", "publish_authoritative_benchmark_corpus", "verify_authoritative_benchmark_corpus_receipt"]
