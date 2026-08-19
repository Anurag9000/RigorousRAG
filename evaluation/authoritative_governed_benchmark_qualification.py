"""Promotion-grade leakage qualification for authoritative v2 governed benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.authoritative_governed_benchmark_io import (
    VerifiedAuthoritativeGovernedBenchmark,
    verify_authoritative_governed_benchmark_import,
)
from evaluation.benchmark_run_evidence import (
    BenchmarkResultArtifactReceipt,
    materialize_benchmark_run_evidence,
)
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.dataset_governance import LeakageSeverity
from evaluation.governed_benchmark_qualification import (
    GovernedBenchmarkLeakageReceipt,
    LeakageFindingEvidence,
)
from training.advanced_path_authority import safe_advanced_path

_HEX = frozenset("0123456789abcdef")
_MAX_SPLITS = 100
_MAX_KEY_KINDS = 100
_MAX_KEYS = 400_000_000


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


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid")
    return selected


def _open_ledger() -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="rigorousrag-leakage-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE keys ("
        "split_name TEXT NOT NULL, key_kind TEXT NOT NULL, value TEXT NOT NULL, "
        "PRIMARY KEY(split_name,key_kind,value)) WITHOUT ROWID"
    )
    return connection, path


def _insert(
    connection: sqlite3.Connection,
    *,
    split_name: str,
    key_kind: str,
    value: str,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO keys(split_name,key_kind,value) VALUES (?,?,?)",
        (split_name, key_kind, value),
    )


def _key_digest(
    connection: sqlite3.Connection,
    *,
    split_name: str,
    key_kind: str,
) -> str:
    digest = hashlib.sha256()
    for (value,) in connection.execute(
        "SELECT value FROM keys WHERE split_name=? AND key_kind=? "
        "ORDER BY value COLLATE BINARY",
        (split_name, key_kind),
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _overlap(
    connection: sqlite3.Connection,
    *,
    left: str,
    right: str,
    kind: str,
    sample_limit: int = 100,
) -> tuple[int, str, tuple[str, ...]] | None:
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM keys a JOIN keys b ON a.key_kind=b.key_kind AND a.value=b.value "
            "WHERE a.split_name=? AND b.split_name=? AND a.key_kind=?",
            (left, right, kind),
        ).fetchone()[0]
    )
    if count <= 0:
        return None
    digest = hashlib.sha256()
    sample: list[str] = []
    cursor = connection.execute(
        "SELECT a.value FROM keys a JOIN keys b ON a.key_kind=b.key_kind AND a.value=b.value "
        "WHERE a.split_name=? AND b.split_name=? AND a.key_kind=? "
        "ORDER BY a.value COLLATE BINARY",
        (left, right, kind),
    )
    for (value,) in cursor:
        selected = str(value)
        digest.update(selected.encode("utf-8"))
        digest.update(b"\n")
        if len(sample) < sample_limit:
            sample.append(selected)
    return count, digest.hexdigest(), tuple(sample)


def qualify_authoritative_governed_benchmark_leakage(
    benchmark: VerifiedAuthoritativeGovernedBenchmark,
    *,
    blocking_key_kinds: Sequence[str] = (
        "record_id",
        "query_id",
        "source_group_id",
    ),
    require_pass: bool = True,
) -> GovernedBenchmarkLeakageReceipt:
    if not isinstance(benchmark, VerifiedAuthoritativeGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedAuthoritativeGovernedBenchmark")
    if not isinstance(require_pass, bool):
        raise ValueError("require_pass must be boolean")
    blocking = tuple(
        _identifier(item, "blocking key kind", 200) for item in blocking_key_kinds
    )
    if not blocking or len(set(blocking)) != len(blocking):
        raise ValueError("blocking_key_kinds must be unique and non-empty")

    connection, ledger_path = _open_ledger()
    try:
        total_keys = 0
        split_kinds: dict[str, set[str]] = {}
        for split_manifest in benchmark.manifest.splits:
            split = split_manifest.name
            kinds = {"record_id", "query_id", "document_id"}
            for example in benchmark.split(split):
                _insert(
                    connection,
                    split_name=split,
                    key_kind="record_id",
                    value=example.example_id,
                )
                _insert(
                    connection,
                    split_name=split,
                    key_kind="query_id",
                    value=example.example_id,
                )
                total_keys += 2
                for document_id in example.relevant_ids:
                    _insert(
                        connection,
                        split_name=split,
                        key_kind="document_id",
                        value=document_id,
                    )
                    total_keys += 1
                source_group = (
                    example.metadata.get("source_group_id")
                    if isinstance(example.metadata, Mapping)
                    else None
                )
                if isinstance(source_group, str) and source_group.strip():
                    _insert(
                        connection,
                        split_name=split,
                        key_kind="source_group_id",
                        value=source_group.strip(),
                    )
                    kinds.add("source_group_id")
                    total_keys += 1
                if total_keys > _MAX_KEYS:
                    raise ValueError("benchmark leakage key material exceeds safety bound")
                if total_keys % 20_000 == 0:
                    connection.commit()
            split_kinds[split] = kinds
            connection.commit()

        if not 2 <= len(split_kinds) <= _MAX_SPLITS:
            raise ValueError("authoritative leakage qualification requires 2..100 splits")
        if any(len(kinds) > _MAX_KEY_KINDS for kinds in split_kinds.values()):
            raise ValueError("benchmark leakage key-kind count exceeds safety bound")

        key_digests: dict[str, dict[str, str]] = {}
        for split in sorted(split_kinds):
            key_digests[split] = {
                kind: _key_digest(
                    connection,
                    split_name=split,
                    key_kind=kind,
                )
                for kind in sorted(split_kinds[split])
            }

        findings: list[LeakageFindingEvidence] = []
        names = sorted(split_kinds)
        blocking_set = set(blocking)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                common = sorted(split_kinds[left] & split_kinds[right])
                for kind in common:
                    overlap = _overlap(
                        connection,
                        left=left,
                        right=right,
                        kind=kind,
                    )
                    if overlap is None:
                        continue
                    count, overlap_sha, sample = overlap
                    findings.append(
                        LeakageFindingEvidence(
                            left_split=left,
                            right_split=right,
                            key_kind=kind,
                            severity=(
                                LeakageSeverity.BLOCKING.value
                                if kind in blocking_set
                                else LeakageSeverity.WARNING.value
                            ),
                            overlap_count=count,
                            overlap_sha256=overlap_sha,
                            overlap_sample=sample,
                        )
                    )

        passed = not any(
            item.severity == LeakageSeverity.BLOCKING.value for item in findings
        )
        if require_pass and not passed:
            summary = ", ".join(
                f"{item.left_split}/{item.right_split}:{item.key_kind}"
                for item in findings
                if item.severity == LeakageSeverity.BLOCKING.value
            )
            raise ValueError(f"blocking split leakage detected: {summary[:4000]}")
        unsigned = {
            "schema": "rigorousrag-governed-benchmark-leakage-receipt/v1",
            "dataset_manifest_sha256": benchmark.manifest.manifest_digest,
            "import_receipt_sha256": benchmark.receipt.receipt_sha256,
            "split_key_sha256": {
                split: dict(groups)
                for split, groups in sorted(key_digests.items())
            },
            "blocking_key_kinds": list(blocking),
            "findings": [asdict(item) for item in findings],
            "passed": passed,
        }
        return GovernedBenchmarkLeakageReceipt(
            dataset_manifest_sha256=benchmark.manifest.manifest_digest,
            import_receipt_sha256=benchmark.receipt.receipt_sha256,
            split_key_sha256=key_digests,
            blocking_key_kinds=blocking,
            findings=tuple(findings),
            passed=passed,
            receipt_sha256=_digest(unsigned),
        )
    finally:
        connection.close()
        try:
            ledger_path.unlink()
        except FileNotFoundError:
            pass


def qualified_authoritative_evaluator_contract_sha256(
    base_evaluator_contract_sha256: str,
    leakage_receipt: GovernedBenchmarkLeakageReceipt,
) -> str:
    base = _sha(
        base_evaluator_contract_sha256,
        "base_evaluator_contract_sha256",
    )
    if (
        not isinstance(leakage_receipt, GovernedBenchmarkLeakageReceipt)
        or not leakage_receipt.passed
    ):
        raise ValueError("qualified evaluation requires a passed leakage receipt")
    return _digest(
        {
            "schema": "rigorousrag-qualified-authoritative-benchmark-evaluator-contract/v2",
            "base_evaluator_contract_sha256": base,
            "dataset_manifest_sha256": leakage_receipt.dataset_manifest_sha256,
            "import_receipt_sha256": leakage_receipt.import_receipt_sha256,
            "leakage_receipt_sha256": leakage_receipt.receipt_sha256,
            "benchmark_authority": "authoritative_governed_benchmark_import/v2",
        }
    )


def materialize_qualified_authoritative_benchmark_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark: VerifiedAuthoritativeGovernedBenchmark,
    leakage_receipt: GovernedBenchmarkLeakageReceipt,
    base_evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_path: str | Path,
) -> tuple[AdvancedEvaluationRun, BenchmarkResultArtifactReceipt]:
    if not isinstance(benchmark, VerifiedAuthoritativeGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedAuthoritativeGovernedBenchmark")
    if (
        leakage_receipt.dataset_manifest_sha256 != benchmark.manifest.manifest_digest
        or leakage_receipt.import_receipt_sha256 != benchmark.receipt.receipt_sha256
        or not leakage_receipt.passed
    ):
        raise ValueError("leakage receipt differs from authoritative governed benchmark")
    evaluator_sha = qualified_authoritative_evaluator_contract_sha256(
        base_evaluator_contract_sha256,
        leakage_receipt,
    )
    return materialize_benchmark_run_evidence(
        result,
        benchmark_manifest=benchmark.manifest,
        evaluator_contract_sha256=evaluator_sha,
        seed=seed,
        repeat_index=repeat_index,
        output_path=output_path,
    )


def qualify_authoritative_governed_benchmark_from_import(
    import_receipt_path: str | Path,
    *,
    require_promotable: bool = True,
    blocking_key_kinds: Sequence[str] = (
        "record_id",
        "query_id",
        "source_group_id",
    ),
) -> tuple[
    VerifiedAuthoritativeGovernedBenchmark,
    GovernedBenchmarkLeakageReceipt,
]:
    if not isinstance(require_promotable, bool):
        raise ValueError("require_promotable must be boolean")
    benchmark = verify_authoritative_governed_benchmark_import(
        import_receipt_path,
        require_promotable=require_promotable,
    )
    leakage = qualify_authoritative_governed_benchmark_leakage(
        benchmark,
        blocking_key_kinds=blocking_key_kinds,
        require_pass=True,
    )
    return benchmark, leakage


def write_authoritative_governed_benchmark_leakage_receipt(
    path: str | Path,
    receipt: GovernedBenchmarkLeakageReceipt,
) -> str:
    if not isinstance(receipt, GovernedBenchmarkLeakageReceipt):
        raise ValueError("receipt must be GovernedBenchmarkLeakageReceipt")
    destination = safe_advanced_path(
        path,
        label="authoritative benchmark leakage receipt",
        must_exist=False,
    )
    if destination.exists() and destination.is_dir():
        raise ValueError("benchmark leakage receipt destination must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(
        {**receipt._unsigned(), "receipt_sha256": receipt.receipt_sha256}
    ) + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}-",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt.receipt_sha256


__all__ = [
    "materialize_qualified_authoritative_benchmark_run_evidence",
    "qualified_authoritative_evaluator_contract_sha256",
    "qualify_authoritative_governed_benchmark_from_import",
    "qualify_authoritative_governed_benchmark_leakage",
    "write_authoritative_governed_benchmark_leakage_receipt",
]
