"""Leakage qualification and evaluator-contract binding for governed benchmarks.

``DatasetManifest`` deliberately records exact bytes/licensing/transforms while split leakage
is a separate analysis.  This module makes the two inseparable for the qualified benchmark
path without changing the mature ``AdvancedEvaluationRun`` schema: a self-verifying leakage
receipt is built from re-verified canonical splits, and the evaluator contract SHA is derived
from the base evaluator contract plus the exact import and leakage receipts.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.benchmark_run_evidence import BenchmarkResultArtifactReceipt, materialize_benchmark_run_evidence
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.dataset_governance import LeakageFinding, LeakageSeverity, assert_no_blocking_leakage, check_split_leakage
from evaluation.governed_benchmark_io import VerifiedGovernedBenchmark, verify_governed_benchmark_import
from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from training.advanced_path_authority import safe_advanced_path

_HEX = frozenset("0123456789abcdef")
_MAX_KEYS_PER_SPLIT = 100_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _key_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LeakageFindingEvidence:
    left_split: str
    right_split: str
    key_kind: str
    severity: str
    overlap_count: int
    overlap_sha256: str
    overlap_sample: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("left_split", "right_split", "key_kind"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.severity not in {item.value for item in LeakageSeverity}:
            raise ValueError("invalid leakage severity")
        if isinstance(self.overlap_count, bool) or not isinstance(self.overlap_count, int) or self.overlap_count <= 0:
            raise ValueError("overlap_count must be positive")
        object.__setattr__(self, "overlap_sha256", _sha(self.overlap_sha256, "overlap_sha256"))
        sample = tuple(_identifier(item, "overlap sample", 2_000) for item in self.overlap_sample)
        if len(sample) > 1000:
            raise ValueError("overlap_sample exceeds safety bound")
        object.__setattr__(self, "overlap_sample", sample)


@dataclass(frozen=True)
class GovernedBenchmarkLeakageReceipt:
    dataset_manifest_sha256: str
    import_receipt_sha256: str
    split_key_sha256: Mapping[str, Mapping[str, str]]
    blocking_key_kinds: tuple[str, ...]
    findings: tuple[LeakageFindingEvidence, ...]
    passed: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "import_receipt_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        normalized: dict[str, dict[str, str]] = {}
        for split_name, groups in self.split_key_sha256.items():
            selected_split = _identifier(split_name, "split name", 200)
            if not isinstance(groups, Mapping):
                raise ValueError("split_key_sha256 entries must be mappings")
            normalized[selected_split] = {_identifier(kind, "key kind", 200): _sha(value, "split key digest") for kind, value in groups.items()}
        object.__setattr__(self, "split_key_sha256", normalized)
        blocking = tuple(_identifier(item, "blocking key kind", 200) for item in self.blocking_key_kinds)
        if not blocking or len(set(blocking)) != len(blocking):
            raise ValueError("blocking_key_kinds must be unique and non-empty")
        object.__setattr__(self, "blocking_key_kinds", blocking)
        findings = tuple(self.findings)
        if any(not isinstance(item, LeakageFindingEvidence) for item in findings):
            raise ValueError("findings must contain LeakageFindingEvidence")
        object.__setattr__(self, "findings", findings)
        expected_passed = not any(item.severity == LeakageSeverity.BLOCKING.value for item in findings)
        if bool(self.passed) != expected_passed:
            raise ValueError("leakage receipt passed flag differs from findings")
        if _digest(self._unsigned()) != self.receipt_sha256:
            raise ValueError("governed benchmark leakage receipt digest mismatch")

    def _unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-governed-benchmark-leakage-receipt/v1",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "import_receipt_sha256": self.import_receipt_sha256,
            "split_key_sha256": {split: dict(groups) for split, groups in sorted(self.split_key_sha256.items())},
            "blocking_key_kinds": list(self.blocking_key_kinds),
            "findings": [asdict(item) for item in self.findings],
            "passed": bool(self.passed),
        }


def _split_keys(benchmark: VerifiedGovernedBenchmark) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for split_manifest in benchmark.manifest.splits:
        record_ids: list[str] = []
        source_groups: list[str] = []
        document_ids: list[str] = []
        for example in benchmark.split(split_manifest.name):
            record_ids.append(example.example_id)
            document_ids.extend(example.relevant_ids)
            source_group = example.metadata.get("source_group_id") if isinstance(example.metadata, Mapping) else None
            if isinstance(source_group, str) and source_group.strip():
                source_groups.append(source_group.strip())
            if len(record_ids) + len(document_ids) + len(source_groups) > _MAX_KEYS_PER_SPLIT:
                raise ValueError("benchmark leakage key material exceeds safety bound")
        groups: dict[str, tuple[str, ...]] = {
            "record_id": tuple(record_ids),
            "query_id": tuple(record_ids),
            "document_id": tuple(document_ids),
        }
        if source_groups:
            groups["source_group_id"] = tuple(source_groups)
        result[split_manifest.name] = groups
    return result


def _finding_evidence(split_keys: Mapping[str, Mapping[str, Sequence[str]]], finding: LeakageFinding) -> LeakageFindingEvidence:
    left = set(split_keys[finding.left_split][finding.key_kind])
    right = set(split_keys[finding.right_split][finding.key_kind])
    overlap = sorted(left & right)
    if not overlap:
        raise RuntimeError("leakage finding has no reproducible overlap")
    return LeakageFindingEvidence(
        left_split=finding.left_split,
        right_split=finding.right_split,
        key_kind=finding.key_kind,
        severity=finding.severity.value,
        overlap_count=len(overlap),
        overlap_sha256=_key_digest(overlap),
        overlap_sample=tuple(overlap[:100]),
    )


def qualify_governed_benchmark_leakage(
    benchmark: VerifiedGovernedBenchmark,
    *,
    blocking_key_kinds: Sequence[str] = ("record_id", "query_id", "source_group_id"),
    require_pass: bool = True,
) -> GovernedBenchmarkLeakageReceipt:
    """Analyze exact canonical splits and return a content-addressed leakage decision."""
    if not isinstance(benchmark, VerifiedGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedGovernedBenchmark")
    blocking = tuple(_identifier(item, "blocking key kind", 200) for item in blocking_key_kinds)
    split_keys = _split_keys(benchmark)
    findings = check_split_leakage(split_keys, blocking_key_kinds=blocking, sample_limit=100)
    if require_pass:
        assert_no_blocking_leakage(findings)
    evidence = tuple(_finding_evidence(split_keys, item) for item in findings)
    key_digests = {split: {kind: _key_digest(tuple(values)) for kind, values in groups.items()} for split, groups in split_keys.items()}
    unsigned = {
        "schema": "rigorousrag-governed-benchmark-leakage-receipt/v1",
        "dataset_manifest_sha256": benchmark.manifest.manifest_digest,
        "import_receipt_sha256": benchmark.receipt.receipt_sha256,
        "split_key_sha256": {split: dict(groups) for split, groups in sorted(key_digests.items())},
        "blocking_key_kinds": list(blocking),
        "findings": [asdict(item) for item in evidence],
        "passed": not any(item.severity == LeakageSeverity.BLOCKING.value for item in evidence),
    }
    return GovernedBenchmarkLeakageReceipt(
        dataset_manifest_sha256=benchmark.manifest.manifest_digest,
        import_receipt_sha256=benchmark.receipt.receipt_sha256,
        split_key_sha256=key_digests,
        blocking_key_kinds=blocking,
        findings=evidence,
        passed=unsigned["passed"],
        receipt_sha256=_digest(unsigned),
    )


def write_governed_benchmark_leakage_receipt(path: str | Path, receipt: GovernedBenchmarkLeakageReceipt) -> str:
    if not isinstance(receipt, GovernedBenchmarkLeakageReceipt):
        raise ValueError("receipt must be GovernedBenchmarkLeakageReceipt")
    destination = safe_advanced_path(path, label="benchmark leakage receipt", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("benchmark leakage receipt destination must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical({**receipt._unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt.receipt_sha256


def qualified_evaluator_contract_sha256(base_evaluator_contract_sha256: str, leakage_receipt: GovernedBenchmarkLeakageReceipt) -> str:
    """Bind evaluator logic to the exact imported benchmark and leakage decision."""
    base = _sha(base_evaluator_contract_sha256, "base_evaluator_contract_sha256")
    if not isinstance(leakage_receipt, GovernedBenchmarkLeakageReceipt) or not leakage_receipt.passed:
        raise ValueError("qualified evaluation requires a passed leakage receipt")
    return _digest({
        "schema": "rigorousrag-qualified-benchmark-evaluator-contract/v1",
        "base_evaluator_contract_sha256": base,
        "dataset_manifest_sha256": leakage_receipt.dataset_manifest_sha256,
        "import_receipt_sha256": leakage_receipt.import_receipt_sha256,
        "leakage_receipt_sha256": leakage_receipt.receipt_sha256,
    })


def materialize_qualified_benchmark_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark: VerifiedGovernedBenchmark,
    leakage_receipt: GovernedBenchmarkLeakageReceipt,
    base_evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_path: str | Path,
) -> tuple[AdvancedEvaluationRun, BenchmarkResultArtifactReceipt]:
    """Bridge a passed governed benchmark directly into promotion-ready run evidence."""
    if leakage_receipt.dataset_manifest_sha256 != benchmark.manifest.manifest_digest or leakage_receipt.import_receipt_sha256 != benchmark.receipt.receipt_sha256:
        raise ValueError("leakage receipt differs from verified governed benchmark")
    evaluator_sha = qualified_evaluator_contract_sha256(base_evaluator_contract_sha256, leakage_receipt)
    return materialize_benchmark_run_evidence(
        result,
        benchmark_manifest=benchmark.manifest,
        evaluator_contract_sha256=evaluator_sha,
        seed=seed,
        repeat_index=repeat_index,
        output_path=output_path,
    )


def qualify_governed_benchmark_from_import(
    import_receipt_path: str | Path,
    *,
    require_promotable: bool = True,
    blocking_key_kinds: Sequence[str] = ("record_id", "query_id", "source_group_id"),
) -> tuple[VerifiedGovernedBenchmark, GovernedBenchmarkLeakageReceipt]:
    """One authoritative read→manifest/license verification→leakage qualification entry point."""
    benchmark = verify_governed_benchmark_import(import_receipt_path, require_promotable=require_promotable)
    leakage = qualify_governed_benchmark_leakage(benchmark, blocking_key_kinds=blocking_key_kinds, require_pass=True)
    return benchmark, leakage


__all__ = [
    "GovernedBenchmarkLeakageReceipt",
    "LeakageFindingEvidence",
    "materialize_qualified_benchmark_run_evidence",
    "qualified_evaluator_contract_sha256",
    "qualify_governed_benchmark_from_import",
    "qualify_governed_benchmark_leakage",
    "write_governed_benchmark_leakage_receipt",
]
