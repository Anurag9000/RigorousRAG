"""Restart-verifiable snapshot-keyed dynamic feature observations.

This authority is the generic production path when uncertainty/semantic features were computed by
an external admitted generator, verifier, annotator or seq2seq stack.  It indexes exact local
JSONL by ``DynamicRuntimeSnapshot.snapshot_sha256`` and implements both
``NextTokenUncertaintyProvider`` and ``DynamicSemanticSignalProvider`` for
``ReferenceDynamicFeatureProvider``.  The sidecar does not calculate or invent semantic signals;
it preserves already-produced values with exact source, row-set and SQLite identities.

JSONL row schema (all fields required):
``snapshot_sha256``, ``token_entropy``, ``top1_margin``, ``evidence_sufficiency``,
``semantic_support``, ``contradiction_risk``, ``citation_coverage``, ``context_novelty``,
``unresolved_entity_ratio``, ``temporal_uncertainty``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from orchestration.dynamic_feature_authority import (
    DynamicSemanticSignals,
    NextTokenUncertainty,
)
from orchestration.dynamic_rag_runtime import DynamicRuntimeSnapshot
from training.advanced_path_authority import safe_advanced_path

_MAX_LINE_BYTES = 4 * 1024 * 1024
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_HEX = frozenset("0123456789abcdef")
_FIELDS = {
    "snapshot_sha256",
    "token_entropy",
    "top1_margin",
    "evidence_sufficiency",
    "semantic_support",
    "contradiction_risk",
    "citation_coverage",
    "context_novelty",
    "unresolved_entity_ratio",
    "temporal_uncertainty",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _unit(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return selected


def _strict(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _normalize(raw: Mapping[str, Any], line_number: int) -> tuple[str, Mapping[str, Any]]:
    if set(raw) != _FIELDS:
        raise ValueError(f"dynamic feature observation line {line_number} fields are invalid")
    snapshot_sha = _sha(raw["snapshot_sha256"], f"line {line_number} snapshot_sha256")
    entropy = _finite(raw["token_entropy"], f"line {line_number} token_entropy")
    if entropy < 0.0:
        raise ValueError("token_entropy must be non-negative")
    payload = {
        "token_entropy": entropy,
        "top1_margin": _unit(raw["top1_margin"], f"line {line_number} top1_margin"),
        "evidence_sufficiency": _unit(raw["evidence_sufficiency"], f"line {line_number} evidence_sufficiency"),
        "semantic_support": _unit(raw["semantic_support"], f"line {line_number} semantic_support"),
        "contradiction_risk": _unit(raw["contradiction_risk"], f"line {line_number} contradiction_risk"),
        "citation_coverage": _unit(raw["citation_coverage"], f"line {line_number} citation_coverage"),
        "context_novelty": _unit(raw["context_novelty"], f"line {line_number} context_novelty"),
        "unresolved_entity_ratio": _unit(raw["unresolved_entity_ratio"], f"line {line_number} unresolved_entity_ratio"),
        "temporal_uncertainty": _unit(raw["temporal_uncertainty"], f"line {line_number} temporal_uncertainty"),
    }
    return snapshot_sha, payload


def _row_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute("SELECT snapshot_sha256,payload_json FROM entries ORDER BY snapshot_sha256")
    while True:
        rows = cursor.fetchmany(8192)
        if not rows:
            break
        for snapshot_sha, payload_json in rows:
            digest.update(_canonical({"snapshot_sha256": snapshot_sha, "payload": json.loads(str(payload_json))}))
            digest.update(b"\n")
    return digest.hexdigest()


def _provider_contract(*, source_sha256: str, semantic_contract_sha256: str, row_digest_sha256: str, record_count: int) -> str:
    return _digest({
        "schema": "rigorousrag-dynamic-feature-observation-provider/v1",
        "source_sha256": source_sha256,
        "semantic_contract_sha256": semantic_contract_sha256,
        "row_digest_sha256": row_digest_sha256,
        "record_count": record_count,
        "keying": "exact_dynamic_runtime_snapshot_sha256",
        "semantics": "precomputed_non_budget_dynamic_retrieval_features",
    })


@dataclass(frozen=True)
class DynamicFeatureObservationReceipt:
    source_path: str
    source_sha256: str
    semantic_contract_sha256: str
    record_count: int
    row_digest_sha256: str
    index_sha256: str
    provider_contract_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        source = safe_advanced_path(self.source_path, label="dynamic feature observation source", must_exist=True, require_file=True)
        object.__setattr__(self, "source_path", str(source))
        for name in (
            "source_sha256", "semantic_contract_sha256", "row_digest_sha256", "index_sha256",
            "provider_contract_sha256", "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("dynamic feature observation receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-feature-observation-receipt/v1",
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "record_count": self.record_count,
            "row_digest_sha256": self.row_digest_sha256,
            "index_sha256": self.index_sha256,
            "provider_contract_sha256": self.provider_contract_sha256,
        }


def publish_dynamic_feature_observations(
    source_path: str | Path,
    *,
    expected_sha256: str,
    semantic_contract_sha256: str,
    output_dir: str | Path,
) -> DynamicFeatureObservationReceipt:
    source = safe_advanced_path(source_path, label="dynamic feature observation JSONL", must_exist=True, require_file=True)
    source_sha = _stream_sha(source)
    if source_sha != _sha(expected_sha256, "expected feature observation SHA-256"):
        raise ValueError("dynamic feature observation source digest mismatch")
    semantic_sha = _sha(semantic_contract_sha256, "semantic_contract_sha256")
    root = safe_advanced_path(output_dir, label="dynamic feature observation authority output", must_exist=False)
    if root.exists():
        raise ValueError("dynamic feature observation authority output must not already exist")
    parent = safe_advanced_path(root.parent, label="dynamic feature observation authority parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'dynamic-feature'}-stage-", dir=parent))
    database = stage / "authority.sqlite"
    connection = sqlite3.connect(str(database), timeout=30.0)
    published = False
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TABLE entries(snapshot_sha256 TEXT PRIMARY KEY,payload_json TEXT NOT NULL) WITHOUT ROWID")
        count = 0
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                if len(raw) > _MAX_LINE_BYTES:
                    raise ValueError(f"dynamic feature observation line {line_number} exceeds byte safety bound")
                if count >= _MAX_RECORDS:
                    raise ValueError("dynamic feature observation sidecar exceeds record safety bound")
                snapshot_sha, payload = _normalize(_strict(raw, f"dynamic feature observation line {line_number}"), line_number)
                try:
                    connection.execute(
                        "INSERT INTO entries(snapshot_sha256,payload_json) VALUES(?,?)",
                        (snapshot_sha, _canonical(payload).decode("utf-8")),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate dynamic feature observation snapshot {snapshot_sha}") from exc
                count += 1
                if count % 10_000 == 0:
                    connection.commit()
        if count <= 0:
            raise ValueError("dynamic feature observation sidecar may not be empty")
        connection.commit()
        row_sha = _row_digest(connection)
        connection.close()
        index_sha = _stream_sha(database)
        provider_sha = _provider_contract(
            source_sha256=source_sha,
            semantic_contract_sha256=semantic_sha,
            row_digest_sha256=row_sha,
            record_count=count,
        )
        unsigned = {
            "schema": "rigorousrag-dynamic-feature-observation-receipt/v1",
            "source_path": str(source),
            "source_sha256": source_sha,
            "semantic_contract_sha256": semantic_sha,
            "record_count": count,
            "row_digest_sha256": row_sha,
            "index_sha256": index_sha,
            "provider_contract_sha256": provider_sha,
        }
        receipt = DynamicFeatureObservationReceipt(
            source_path=str(source),
            source_sha256=source_sha,
            semantic_contract_sha256=semantic_sha,
            record_count=count,
            row_digest_sha256=row_sha,
            index_sha256=index_sha,
            provider_contract_sha256=provider_sha,
            receipt_sha256=_digest(unsigned),
        )
        receipt_path = stage / "feature_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"authority.sqlite", "feature_receipt.json"}:
            raise RuntimeError("dynamic feature observation authority directory is not closed")
        os.replace(stage, root)
        published = True
        verified = verify_dynamic_feature_observations(root / "feature_receipt.json")
        if verified.receipt_sha256 != receipt.receipt_sha256:
            raise RuntimeError("dynamic feature observation identity changed after publication")
        return verified
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_dynamic_feature_observations(path: str | Path) -> DynamicFeatureObservationReceipt:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("dynamic feature observation receipt may not be a symlink")
    receipt_path = safe_advanced_path(raw_path, label="dynamic feature observation receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / "feature_receipt.json":
        raise ValueError("dynamic feature observation receipt must use canonical filename")
    if {item.name for item in root.iterdir()} != {"authority.sqlite", "feature_receipt.json"}:
        raise ValueError("dynamic feature observation authority directory is not closed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("dynamic feature observation authority contains a non-regular child")
    if receipt_path.stat().st_size <= 0 or receipt_path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("dynamic feature observation receipt exceeds byte safety bound")
    raw = _strict(receipt_path.read_bytes(), "dynamic feature observation receipt")
    expected = {
        "schema", "source_path", "source_sha256", "semantic_contract_sha256", "record_count",
        "row_digest_sha256", "index_sha256", "provider_contract_sha256", "receipt_sha256",
    }
    if set(raw) != expected or raw.get("schema") != "rigorousrag-dynamic-feature-observation-receipt/v1":
        raise ValueError("unsupported dynamic feature observation receipt schema")
    receipt = DynamicFeatureObservationReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    if _stream_sha(Path(receipt.source_path)) != receipt.source_sha256:
        raise ValueError("dynamic feature observation source bytes changed")
    database = root / "authority.sqlite"
    if _stream_sha(database) != receipt.index_sha256:
        raise ValueError("dynamic feature observation index bytes differ")
    with sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True, timeout=30.0) as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        row_sha = _row_digest(connection)
    if count != receipt.record_count or row_sha != receipt.row_digest_sha256:
        raise ValueError("dynamic feature observation row authority differs")
    if _provider_contract(
        source_sha256=receipt.source_sha256,
        semantic_contract_sha256=receipt.semantic_contract_sha256,
        row_digest_sha256=receipt.row_digest_sha256,
        record_count=receipt.record_count,
    ) != receipt.provider_contract_sha256:
        raise ValueError("dynamic feature observation provider contract differs")
    return receipt


class SqliteDynamicFeatureObservationProvider:
    """Worker-safe provider implementing uncertainty and semantic feature protocols."""

    def __init__(self, receipt_path: str | Path) -> None:
        receipt = verify_dynamic_feature_observations(receipt_path)
        selected = safe_advanced_path(receipt_path, label="dynamic feature observation receipt", must_exist=True, require_file=True)
        self.receipt = receipt
        self.database_path = str(selected.parent / "authority.sqlite")
        self._local = threading.local()

    @property
    def contract_sha256(self) -> str:
        return self.receipt.provider_contract_sha256

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_local", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            database = Path(self.database_path)
            if database.is_symlink() or not database.is_file():
                raise ValueError("dynamic feature observation index is missing or unsafe")
            if _stream_sha(database) != self.receipt.index_sha256:
                raise ValueError("dynamic feature observation index changed after verification")
            connection = sqlite3.connect(
                f"file:{database}?mode=ro&immutable=1",
                uri=True,
                timeout=30.0,
            )
            self._local.connection = connection
        return connection

    def _payload(self, snapshot: DynamicRuntimeSnapshot) -> Mapping[str, Any]:
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        row = self._connection().execute(
            "SELECT payload_json FROM entries WHERE snapshot_sha256=?",
            (snapshot.snapshot_sha256,),
        ).fetchone()
        if row is None:
            raise ValueError(f"dynamic feature observation sidecar lacks snapshot {snapshot.snapshot_sha256}")
        value = json.loads(str(row[0]), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        if not isinstance(value, Mapping):
            raise ValueError("dynamic feature observation indexed payload is malformed")
        return value

    def uncertainty(self, snapshot: DynamicRuntimeSnapshot) -> NextTokenUncertainty:
        value = self._payload(snapshot)
        return NextTokenUncertainty(
            token_entropy=value["token_entropy"],
            top1_margin=value["top1_margin"],
        )

    def signals(self, snapshot: DynamicRuntimeSnapshot) -> DynamicSemanticSignals:
        value = self._payload(snapshot)
        return DynamicSemanticSignals(
            evidence_sufficiency=value["evidence_sufficiency"],
            semantic_support=value["semantic_support"],
            contradiction_risk=value["contradiction_risk"],
            citation_coverage=value["citation_coverage"],
            context_novelty=value["context_novelty"],
            unresolved_entity_ratio=value["unresolved_entity_ratio"],
            temporal_uncertainty=value["temporal_uncertainty"],
        )

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None


__all__ = [
    "DynamicFeatureObservationReceipt",
    "SqliteDynamicFeatureObservationProvider",
    "publish_dynamic_feature_observations",
    "verify_dynamic_feature_observations",
]
