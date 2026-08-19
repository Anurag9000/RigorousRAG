"""Authoritative benchmark/sample-universe contracts for promotion-grade evaluation.

A result artifact is not promotion-grade merely because it contains valid rows and an arbitrary
``evaluator_contract_sha256``.  This module binds evaluation to a verified benchmark authority,
an exact sample-ID universe, and a base evaluator contract.  It supports generic governed
benchmarks (explicit selected splits) and the stricter retrieval benchmark v3 authority.

The persisted cohort can be independently reconstructed after restart, and result receipts can
be proved to cover exactly the cohort sample universe with a disk-backed identity ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.authoritative_benchmark_leakage_io import (
    verify_authoritative_benchmark_leakage_receipt,
)
from evaluation.authoritative_governed_benchmark_io import (
    VerifiedAuthoritativeGovernedBenchmark,
    verify_authoritative_governed_benchmark_import,
)
from evaluation.authoritative_governed_retrieval_benchmark import (
    AuthoritativeGovernedRetrievalBenchmark,
    authoritative_retrieval_evaluator_contract_sha256,
)
from evaluation.authoritative_governed_retrieval_io import (
    close_reconstructed_authoritative_retrieval_benchmark,
    load_authoritative_retrieval_benchmark_receipt,
    reconstruct_authoritative_retrieval_benchmark,
)
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt
from evaluation.strict_authoritative_benchmark_result_verification import (
    verify_strict_authoritative_benchmark_result_receipt,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 32 * 1024 * 1024
_MAX_SPLITS = 100
_MAX_SAMPLE = 100
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


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("evaluation cohort destination must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class EvaluationAuthorityReceiptBinding:
    role: str
    path: str
    file_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {"benchmark_import", "leakage", "retrieval_benchmark"}:
            raise ValueError("unsupported evaluation authority receipt role")
        selected = safe_advanced_path(self.path, label=f"{self.role} authority receipt", must_exist=True, require_file=True)
        object.__setattr__(self, "path", str(selected))
        object.__setattr__(self, "file_sha256", _sha(self.file_sha256, "file_sha256"))
        if _file_sha(selected) != self.file_sha256:
            raise ValueError(f"{self.role} authority receipt bytes differ from binding")


@dataclass(frozen=True)
class AuthoritativeEvaluationCohortContract:
    authority_kind: str
    benchmark_id: str
    benchmark_manifest_sha256: str
    benchmark_contract_sha256: str
    selected_splits: tuple[str, ...]
    sample_count: int
    sample_universe_sha256: str
    base_evaluator_contract_sha256: str
    evaluator_contract_sha256: str
    authority_receipts: tuple[EvaluationAuthorityReceiptBinding, ...]
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.authority_kind not in {"governed_benchmark_v2", "retrieval_benchmark_v3"}:
            raise ValueError("unsupported evaluation authority kind")
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        for name in (
            "benchmark_manifest_sha256",
            "benchmark_contract_sha256",
            "sample_universe_sha256",
            "base_evaluator_contract_sha256",
            "evaluator_contract_sha256",
            "contract_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        splits = tuple(self.selected_splits)
        if not splits or len(splits) > _MAX_SPLITS or len(set(splits)) != len(splits):
            raise ValueError("selected_splits must be unique and non-empty")
        if any(not isinstance(item, str) or not item.strip() for item in splits):
            raise ValueError("selected_splits must contain non-empty strings")
        object.__setattr__(self, "selected_splits", tuple(sorted(splits)))
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        receipts = tuple(self.authority_receipts)
        if not receipts or any(not isinstance(item, EvaluationAuthorityReceiptBinding) for item in receipts):
            raise ValueError("authority_receipts must be non-empty receipt bindings")
        if len({item.role for item in receipts}) != len(receipts):
            raise ValueError("evaluation authority receipt roles must be unique")
        expected_roles = {"benchmark_import", "leakage"} if self.authority_kind == "governed_benchmark_v2" else {"retrieval_benchmark"}
        if {item.role for item in receipts} != expected_roles:
            raise ValueError("evaluation authority receipts differ from authority-kind requirements")
        object.__setattr__(self, "authority_receipts", tuple(sorted(receipts, key=lambda item: item.role)))
        if _digest(self.unsigned()) != self.contract_sha256:
            raise ValueError("authoritative evaluation cohort digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-evaluation-cohort/v1",
            "authority_kind": self.authority_kind,
            "benchmark_id": self.benchmark_id,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "selected_splits": list(self.selected_splits),
            "sample_count": self.sample_count,
            "sample_universe_sha256": self.sample_universe_sha256,
            "base_evaluator_contract_sha256": self.base_evaluator_contract_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "authority_receipts": [asdict(item) for item in self.authority_receipts],
        }


def _sample_universe(examples: Iterable[Any]) -> tuple[int, str]:
    descriptor, raw_database = tempfile.mkstemp(prefix="rigorousrag-cohort-ids-", suffix=".sqlite3")
    os.close(descriptor)
    try:
        connection = sqlite3.connect(raw_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE ids(id TEXT PRIMARY KEY) WITHOUT ROWID")
            count = 0
            for example in examples:
                example_id = str(getattr(example, "example_id", "")).strip()
                if not example_id:
                    raise ValueError("benchmark sample lacks example_id")
                try:
                    connection.execute("INSERT INTO ids(id) VALUES (?)", (example_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"benchmark sample id {example_id!r} is duplicated") from exc
                count += 1
                if count % 20_000 == 0: connection.commit()
            connection.commit()
            if count <= 0:
                raise ValueError("evaluation cohort contains no samples")
            digest = hashlib.sha256()
            for (value,) in connection.execute("SELECT id FROM ids ORDER BY id COLLATE BINARY"):
                digest.update(str(value).encode("utf-8")); digest.update(b"\n")
            return count, digest.hexdigest()
        finally:
            connection.close()
    finally:
        try: os.unlink(raw_database)
        except FileNotFoundError: pass


def _selected_examples(benchmark: VerifiedAuthoritativeGovernedBenchmark, selected_splits: Sequence[str]) -> Iterable[Any]:
    known = {item.name for item in benchmark.manifest.splits}
    selected = tuple(sorted(selected_splits))
    if not selected or len(set(selected)) != len(selected) or any(name not in known for name in selected):
        raise ValueError("selected_splits must be unique known benchmark splits")
    for split in selected:
        yield from benchmark.split(split)


def build_governed_benchmark_evaluation_cohort(
    benchmark: VerifiedAuthoritativeGovernedBenchmark,
    *,
    leakage_receipt_path: str | Path,
    selected_splits: Sequence[str],
    base_evaluator_contract_sha256: str,
) -> AuthoritativeEvaluationCohortContract:
    if not isinstance(benchmark, VerifiedAuthoritativeGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedAuthoritativeGovernedBenchmark")
    import_receipt_path = safe_advanced_path(Path(benchmark.root) / "import_receipt.json", label="benchmark import receipt", must_exist=True, require_file=True)
    leakage_path = safe_advanced_path(leakage_receipt_path, label="benchmark leakage receipt", must_exist=True, require_file=True)
    leakage = verify_authoritative_benchmark_leakage_receipt(leakage_path, benchmark=benchmark, require_pass=True)
    selected = tuple(sorted(selected_splits))
    sample_count, sample_sha = _sample_universe(_selected_examples(benchmark, selected))
    benchmark_contract = _digest({
        "schema": "rigorousrag-governed-benchmark-evaluation-universe/v1",
        "benchmark_manifest_sha256": benchmark.manifest.manifest_digest,
        "import_receipt_sha256": benchmark.receipt.receipt_sha256,
        "leakage_receipt_sha256": leakage.receipt_sha256,
        "selected_splits": list(selected),
        "sample_count": sample_count,
        "sample_universe_sha256": sample_sha,
    })
    base = _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256")
    evaluator = _digest({
        "schema": "rigorousrag-governed-benchmark-evaluator-contract/v1",
        "base_evaluator_contract_sha256": base,
        "benchmark_contract_sha256": benchmark_contract,
        "sample_universe_sha256": sample_sha,
    })
    receipts = (
        EvaluationAuthorityReceiptBinding("benchmark_import", str(import_receipt_path), _file_sha(import_receipt_path)),
        EvaluationAuthorityReceiptBinding("leakage", str(leakage_path), _file_sha(leakage_path)),
    )
    unsigned = {
        "schema": "rigorousrag-authoritative-evaluation-cohort/v1",
        "authority_kind": "governed_benchmark_v2",
        "benchmark_id": benchmark.manifest.dataset_id,
        "benchmark_manifest_sha256": benchmark.manifest.manifest_digest,
        "benchmark_contract_sha256": benchmark_contract,
        "selected_splits": list(selected),
        "sample_count": sample_count,
        "sample_universe_sha256": sample_sha,
        "base_evaluator_contract_sha256": base,
        "evaluator_contract_sha256": evaluator,
        "authority_receipts": [asdict(item) for item in sorted(receipts, key=lambda item: item.role)],
    }
    return AuthoritativeEvaluationCohortContract(
        authority_kind="governed_benchmark_v2", benchmark_id=benchmark.manifest.dataset_id,
        benchmark_manifest_sha256=benchmark.manifest.manifest_digest, benchmark_contract_sha256=benchmark_contract,
        selected_splits=selected, sample_count=sample_count, sample_universe_sha256=sample_sha,
        base_evaluator_contract_sha256=base, evaluator_contract_sha256=evaluator,
        authority_receipts=receipts, contract_sha256=_digest(unsigned),
    )


def build_retrieval_evaluation_cohort(
    benchmark: AuthoritativeGovernedRetrievalBenchmark,
    *,
    retrieval_benchmark_receipt_path: str | Path,
    base_evaluator_contract_sha256: str,
) -> AuthoritativeEvaluationCohortContract:
    if not isinstance(benchmark, AuthoritativeGovernedRetrievalBenchmark):
        raise ValueError("benchmark must be AuthoritativeGovernedRetrievalBenchmark")
    receipt_path = safe_advanced_path(retrieval_benchmark_receipt_path, label="authoritative retrieval benchmark receipt", must_exist=True, require_file=True)
    persisted = load_authoritative_retrieval_benchmark_receipt(receipt_path)
    if persisted.retrieval_contract_sha256 != benchmark.contract_sha256 or persisted.query_manifest_sha256 != benchmark.queries.manifest.manifest_digest:
        raise ValueError("retrieval benchmark receipt differs from in-memory authoritative benchmark")
    base = _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256")
    evaluator = authoritative_retrieval_evaluator_contract_sha256(base, benchmark)
    selected = tuple(sorted(item.name for item in benchmark.queries.manifest.splits))
    sample_count = benchmark.qrels.receipt.query_count
    receipt_binding = EvaluationAuthorityReceiptBinding("retrieval_benchmark", str(receipt_path), _file_sha(receipt_path))
    unsigned = {
        "schema": "rigorousrag-authoritative-evaluation-cohort/v1",
        "authority_kind": "retrieval_benchmark_v3",
        "benchmark_id": benchmark.queries.manifest.dataset_id,
        "benchmark_manifest_sha256": benchmark.queries.manifest.manifest_digest,
        "benchmark_contract_sha256": benchmark.contract_sha256,
        "selected_splits": list(selected),
        "sample_count": sample_count,
        "sample_universe_sha256": benchmark.query_universe_sha256,
        "base_evaluator_contract_sha256": base,
        "evaluator_contract_sha256": evaluator,
        "authority_receipts": [asdict(receipt_binding)],
    }
    return AuthoritativeEvaluationCohortContract(
        authority_kind="retrieval_benchmark_v3", benchmark_id=benchmark.queries.manifest.dataset_id,
        benchmark_manifest_sha256=benchmark.queries.manifest.manifest_digest, benchmark_contract_sha256=benchmark.contract_sha256,
        selected_splits=selected, sample_count=sample_count, sample_universe_sha256=benchmark.query_universe_sha256,
        base_evaluator_contract_sha256=base, evaluator_contract_sha256=evaluator,
        authority_receipts=(receipt_binding,), contract_sha256=_digest(unsigned),
    )


def write_authoritative_evaluation_cohort(path: str | Path, cohort: AuthoritativeEvaluationCohortContract) -> None:
    if not isinstance(cohort, AuthoritativeEvaluationCohortContract):
        raise ValueError("cohort must be AuthoritativeEvaluationCohortContract")
    destination = safe_advanced_path(path, label="authoritative evaluation cohort", must_exist=False)
    if destination.exists():
        raise ValueError("authoritative evaluation cohort destination must not already exist")
    _atomic(destination, _canonical({**cohort.unsigned(), "contract_sha256": cohort.contract_sha256}) + b"\n")


def read_authoritative_evaluation_cohort(path: str | Path) -> AuthoritativeEvaluationCohortContract:
    source = safe_advanced_path(path, label="authoritative evaluation cohort", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("authoritative evaluation cohort exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("authoritative evaluation cohort is not strict JSON") from exc
    required = {"schema", "authority_kind", "benchmark_id", "benchmark_manifest_sha256", "benchmark_contract_sha256", "selected_splits", "sample_count", "sample_universe_sha256", "base_evaluator_contract_sha256", "evaluator_contract_sha256", "authority_receipts", "contract_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-evaluation-cohort/v1" or not isinstance(raw.get("selected_splits"), list) or not isinstance(raw.get("authority_receipts"), list):
        raise ValueError("unsupported authoritative evaluation cohort schema")
    bindings = []
    for item in raw["authority_receipts"]:
        if not isinstance(item, Mapping) or set(item) != {"role", "path", "file_sha256"}:
            raise ValueError("evaluation authority receipt binding is malformed")
        bindings.append(EvaluationAuthorityReceiptBinding(**dict(item)))
    return AuthoritativeEvaluationCohortContract(
        authority_kind=raw["authority_kind"], benchmark_id=raw["benchmark_id"], benchmark_manifest_sha256=raw["benchmark_manifest_sha256"],
        benchmark_contract_sha256=raw["benchmark_contract_sha256"], selected_splits=tuple(raw["selected_splits"]), sample_count=raw["sample_count"],
        sample_universe_sha256=raw["sample_universe_sha256"], base_evaluator_contract_sha256=raw["base_evaluator_contract_sha256"], evaluator_contract_sha256=raw["evaluator_contract_sha256"],
        authority_receipts=tuple(bindings), contract_sha256=raw["contract_sha256"],
    )


def _binding(cohort: AuthoritativeEvaluationCohortContract, role: str) -> EvaluationAuthorityReceiptBinding:
    matches = [item for item in cohort.authority_receipts if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"evaluation cohort lacks unique {role} authority binding")
    return matches[0]


def verify_authoritative_evaluation_cohort(path: str | Path) -> AuthoritativeEvaluationCohortContract:
    persisted = read_authoritative_evaluation_cohort(path)
    if persisted.authority_kind == "governed_benchmark_v2":
        import_binding = _binding(persisted, "benchmark_import")
        leakage_binding = _binding(persisted, "leakage")
        benchmark = verify_authoritative_governed_benchmark_import(import_binding.path, require_promotable=True)
        rebuilt = build_governed_benchmark_evaluation_cohort(
            benchmark,
            leakage_receipt_path=leakage_binding.path,
            selected_splits=persisted.selected_splits,
            base_evaluator_contract_sha256=persisted.base_evaluator_contract_sha256,
        )
    else:
        retrieval_binding = _binding(persisted, "retrieval_benchmark")
        benchmark, _ = reconstruct_authoritative_retrieval_benchmark(retrieval_binding.path)
        try:
            rebuilt = build_retrieval_evaluation_cohort(
                benchmark,
                retrieval_benchmark_receipt_path=retrieval_binding.path,
                base_evaluator_contract_sha256=persisted.base_evaluator_contract_sha256,
            )
        finally:
            close_reconstructed_authoritative_retrieval_benchmark(benchmark)
    if rebuilt.contract_sha256 != persisted.contract_sha256:
        raise ValueError("persisted evaluation cohort differs from independent reconstruction")
    return persisted


def _expected_ids(cohort: AuthoritativeEvaluationCohortContract) -> Iterable[str]:
    if cohort.authority_kind == "governed_benchmark_v2":
        benchmark = verify_authoritative_governed_benchmark_import(_binding(cohort, "benchmark_import").path, require_promotable=True)
        for split in cohort.selected_splits:
            for example in benchmark.split(split):
                yield example.example_id
        return
    benchmark, _ = reconstruct_authoritative_retrieval_benchmark(_binding(cohort, "retrieval_benchmark").path)
    try:
        for split in cohort.selected_splits:
            for example in benchmark.queries.split(split):
                yield example.example_id
    finally:
        close_reconstructed_authoritative_retrieval_benchmark(benchmark)


def assert_result_receipt_matches_cohort(
    result_receipt_path: str | Path,
    *,
    cohort: AuthoritativeEvaluationCohortContract,
) -> Any:
    if not isinstance(cohort, AuthoritativeEvaluationCohortContract):
        raise ValueError("cohort must be AuthoritativeEvaluationCohortContract")
    run, receipt = verify_strict_authoritative_benchmark_result_receipt(result_receipt_path)
    checks = {
        "benchmark_id": run.benchmark_id == cohort.benchmark_id,
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256 == cohort.benchmark_manifest_sha256,
        "evaluator_contract_sha256": run.evaluator_contract_sha256 == cohort.evaluator_contract_sha256,
        "sample_count": run.sample_count == cohort.sample_count,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError("result receipt differs from authoritative evaluation cohort: " + ",".join(failures))

    descriptor, raw_database = tempfile.mkstemp(prefix="rigorousrag-result-cohort-", suffix=".sqlite3")
    os.close(descriptor)
    try:
        connection = sqlite3.connect(raw_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF"); connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE expected(id TEXT PRIMARY KEY) WITHOUT ROWID")
            connection.execute("CREATE TABLE actual(id TEXT PRIMARY KEY) WITHOUT ROWID")
            expected_count = 0
            for example_id in _expected_ids(cohort):
                try: connection.execute("INSERT INTO expected(id) VALUES (?)", (example_id,))
                except sqlite3.IntegrityError as exc: raise ValueError(f"cohort sample id {example_id!r} is duplicated") from exc
                expected_count += 1
                if expected_count % 20_000 == 0: connection.commit()
            connection.commit()
            artifact = safe_advanced_path(receipt.result_artifact_path, label="authoritative result artifact", must_exist=True, require_file=True)
            actual_count = 0
            with artifact.open("rb") as handle:
                _ = handle.readline()
                for line_number, raw in enumerate(handle, start=2):
                    if not raw.strip(): continue
                    try: value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                    except Exception as exc: raise ValueError(f"result line {line_number} is not strict JSON") from exc
                    if not isinstance(value, Mapping): raise ValueError(f"result line {line_number} must be an object")
                    if value.get("record_type") == "footer": break
                    if value.get("record_type") != "row" or not isinstance(value.get("example_id"), str): raise ValueError(f"result line {line_number} lacks canonical row identity")
                    example_id = value["example_id"]
                    try: connection.execute("INSERT INTO actual(id) VALUES (?)", (example_id,))
                    except sqlite3.IntegrityError as exc: raise ValueError(f"result example id {example_id!r} is duplicated") from exc
                    actual_count += 1
                    if actual_count % 20_000 == 0: connection.commit()
            connection.commit()
            missing = [str(row[0]) for row in connection.execute("SELECT e.id FROM expected e LEFT JOIN actual a ON e.id=a.id WHERE a.id IS NULL ORDER BY e.id COLLATE BINARY LIMIT ?", (_MAX_SAMPLE,))]
            extra = [str(row[0]) for row in connection.execute("SELECT a.id FROM actual a LEFT JOIN expected e ON a.id=e.id WHERE e.id IS NULL ORDER BY a.id COLLATE BINARY LIMIT ?", (_MAX_SAMPLE,))]
            if expected_count != cohort.sample_count or actual_count != cohort.sample_count or missing or extra:
                raise ValueError(
                    "result/cohort sample universes differ; "
                    f"expected_count={expected_count} actual_count={actual_count} "
                    f"missing_sample={missing} extra_sample={extra}"
                )
            digest = hashlib.sha256()
            for (value,) in connection.execute("SELECT id FROM actual ORDER BY id COLLATE BINARY"):
                digest.update(str(value).encode("utf-8")); digest.update(b"\n")
            if digest.hexdigest() != cohort.sample_universe_sha256:
                raise ValueError("result sample-universe digest differs from authoritative cohort")
            return run
        finally:
            connection.close()
    finally:
        try: os.unlink(raw_database)
        except FileNotFoundError: pass


__all__ = [
    "AuthoritativeEvaluationCohortContract",
    "EvaluationAuthorityReceiptBinding",
    "assert_result_receipt_matches_cohort",
    "build_governed_benchmark_evaluation_cohort",
    "build_retrieval_evaluation_cohort",
    "read_authoritative_evaluation_cohort",
    "verify_authoritative_evaluation_cohort",
    "write_authoritative_evaluation_cohort",
]
