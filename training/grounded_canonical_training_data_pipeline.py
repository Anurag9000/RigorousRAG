"""Canonical final-manifest-bound grounded-RAG supervision pipeline.

Raw governed annotations are first republished with deterministic teacher/retriever cache keys
and proven split-level example-id isolation. Only then are optional teacher/reference/document-
utility tensors materialized into strict caches whose identities bind that final manifest.
Completed caches are sealed read-only before their content contracts enter the canonical
receipt. Existing low-level supervision providers/materializers are reused; importing this
module runs no model, download, retrieval or training.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from evaluation.dataset_governance import DatasetManifest, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import (
    ManifestBoundAuthoritativeJsonlDataset,
    StancedGroundedClaimAnnotation,
    parse_authoritative_grounded_example,
)
from training.advanced_rag_cache_materialization import (
    materialize_document_utility_cache,
    materialize_reference_policy_cache,
    materialize_teacher_logit_cache,
)
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.advanced_rag_supervision import GroundedSupervisionMaterializer, SupervisionCacheIdentity
from training.governed_grounded_io import VerifiedGovernedGroundedDataset

_HEX = frozenset("0123456789abcdef")
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_MATERIALIZATION_BATCH_SIZE = 4096


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _commit(value: Any) -> str:
    selected = str(value).strip().lower()
    if len(selected) not in {40, 64} or any(ch not in _HEX for ch in selected):
        raise ValueError("source_commit must be a full Git object id")
    return selected


def _batch_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_MATERIALIZATION_BATCH_SIZE:
        raise ValueError(f"materialization_batch_size must lie in [1,{_MAX_MATERIALIZATION_BATCH_SIZE}]")
    return value


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _empty_root(path: str | Path, label: str) -> Path:
    root = safe_advanced_path(path, label=label, must_exist=False)
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"{label} must be a directory")
        if any(root.iterdir()):
            raise ValueError(f"{label} must be empty before canonical materialization")
    return root


def _span(span: Any) -> Mapping[str, int]:
    return {"start": int(span.start), "end": int(span.end)}


def _payload(example: Any, *, source_manifest_sha256: str, source_receipt_sha256: str) -> Mapping[str, Any]:
    claims = []
    for claim in example.claims:
        item: dict[str, Any] = {
            "span": _span(claim.span),
            "evidence_ids": list(claim.evidence_ids),
            "supported": bool(claim.supported),
            "contradicted": bool(claim.contradicted),
        }
        if isinstance(claim, StancedGroundedClaimAnnotation):
            item["supporting_evidence_ids"] = list(claim.supporting_evidence_ids)
            item["contradicting_evidence_ids"] = list(claim.contradicting_evidence_ids)
        claims.append(item)
    metadata = dict(example.metadata)
    metadata["grounded_source_manifest_sha256"] = source_manifest_sha256
    metadata["grounded_source_import_receipt_sha256"] = source_receipt_sha256
    return {
        "example_id": example.example_id,
        "prompt": example.prompt,
        "answer": example.answer,
        "evidence": [{"evidence_id": item.evidence_id, "text": item.text, "source_id": item.source_id} for item in example.evidence],
        "claims": claims,
        "abstain": example.abstain,
        "reflection_action": example.reflection_action.value,
        "unsupported_spans": [_span(item) for item in example.unsupported_spans],
        "chosen_answer": example.chosen_answer,
        "rejected_answer": example.rejected_answer,
        "reference_chosen_log_prob": example.reference_chosen_log_prob,
        "reference_rejected_log_prob": example.reference_rejected_log_prob,
        "teacher_cache_key": f"teacher:{example.example_id}",
        "retriever_cache_key": f"utility:{example.example_id}",
        "metadata": metadata,
    }


@dataclass(frozen=True)
class CanonicalGroundedSplit:
    name: str
    path: str
    sha256: str
    record_count: int
    record_id_sha256: str
    evidence_id_sha256: str


@dataclass(frozen=True)
class GroundedCacheBinding:
    kind: str
    root: str
    identity_sha256: str
    contract_sha256: str
    producer_sha256: str

    def __post_init__(self) -> None:
        for name in ("identity_sha256", "contract_sha256", "producer_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))


@dataclass(frozen=True)
class CanonicalGroundedTrainingDataReceipt:
    source_manifest_sha256: str
    source_import_receipt_sha256: str
    dataset_manifest_sha256: str
    transformation_sha256: str
    splits: tuple[CanonicalGroundedSplit, ...]
    caches: tuple[GroundedCacheBinding, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("source_manifest_sha256", "source_import_receipt_sha256", "dataset_manifest_sha256", "transformation_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not self.splits:
            raise ValueError("canonical grounded receipt requires splits")
        if len({item.name for item in self.caches}) != len(self.caches):
            raise ValueError("canonical grounded cache kinds must be unique")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("canonical grounded training-data receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-canonical-grounded-training-data-receipt/v1",
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_import_receipt_sha256": self.source_import_receipt_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "transformation_sha256": self.transformation_sha256,
            "splits": [asdict(item) for item in self.splits],
            "caches": [asdict(item) for item in self.caches],
        }


@dataclass(frozen=True)
class CanonicalGroundedTrainingDataResult:
    manifest: DatasetManifest
    splits: tuple[CanonicalGroundedSplit, ...]
    teacher_cache: AuthoritativeSafetensorSupervisionCache | None
    reference_cache: AuthoritativeSafetensorSupervisionCache | None
    retriever_utility_cache: AuthoritativeSafetensorSupervisionCache | None
    receipt: CanonicalGroundedTrainingDataReceipt

    def split(self, name: str) -> ManifestBoundAuthoritativeJsonlDataset:
        matches = [item for item in self.splits if item.name == name]
        if len(matches) != 1:
            raise ValueError(f"unknown canonical grounded split {name!r}")
        item = matches[0]
        return ManifestBoundAuthoritativeJsonlDataset(item.path, expected_sha256=item.sha256, dataset_manifest_sha256=self.manifest.manifest_digest, split_name=item.name, record_kind="grounded_generation", expected_record_count=item.record_count)


def _provider(provider: Any, *, tokenizer_sha256: str, label: str) -> tuple[str, str]:
    contract = _sha(getattr(provider, "contract_sha256", None), f"{label} contract_sha256")
    model_sha = _sha(getattr(provider, "model_sha256", None), f"{label} model_sha256")
    provider_tokenizer = _sha(getattr(provider, "tokenizer_sha256", None), f"{label} tokenizer_sha256")
    if provider_tokenizer != tokenizer_sha256:
        raise ValueError(f"{label} tokenizer identity differs from grounded training tokenizer")
    return contract, model_sha


def _cache(kind: str, root: str | Path, *, producer_sha256: str, tokenizer_sha256: str, dataset_manifest_sha256: str, source_commit: str, provider_contract_sha256: str) -> AuthoritativeSafetensorSupervisionCache:
    selected = _empty_root(root, f"{kind} cache root")
    config_sha = _digest({"schema": "rigorousrag-canonical-grounded-cache-config/v1", "kind": kind, "provider_contract_sha256": provider_contract_sha256, "dataset_manifest_sha256": dataset_manifest_sha256})
    return AuthoritativeSafetensorSupervisionCache(selected, SupervisionCacheIdentity(cache_kind=kind, producer_sha256=producer_sha256, tokenizer_sha256=tokenizer_sha256, dataset_manifest_sha256=dataset_manifest_sha256, source_commit=source_commit, config_sha256=config_sha))


def _example_batches(published: Sequence[CanonicalGroundedSplit], *, dataset_manifest_sha256: str, batch_size: int) -> Iterator[tuple[Any, ...]]:
    pending: list[Any] = []
    for item in published:
        dataset = ManifestBoundAuthoritativeJsonlDataset(item.path, expected_sha256=item.sha256, dataset_manifest_sha256=dataset_manifest_sha256, split_name=item.name, record_kind="grounded_generation", expected_record_count=item.record_count)
        for index in range(len(dataset)):
            pending.append(dataset[index])
            if len(pending) == batch_size:
                yield tuple(pending)
                pending.clear()
    if pending:
        yield tuple(pending)


def build_canonical_grounded_training_data(
    source: VerifiedGovernedGroundedDataset,
    *,
    tokenizer_sha256: str,
    source_commit: str,
    output_dir: str | Path,
    teacher_provider: Any | None = None,
    teacher_cache_root: str | Path | None = None,
    reference_provider: Any | None = None,
    reference_cache_root: str | Path | None = None,
    document_utility_provider: Any | None = None,
    retriever_utility_cache_root: str | Path | None = None,
    materialization_batch_size: int = 8,
) -> CanonicalGroundedTrainingDataResult:
    if not isinstance(source, VerifiedGovernedGroundedDataset):
        raise ValueError("source must be VerifiedGovernedGroundedDataset")
    tokenizer_sha = _sha(tokenizer_sha256, "tokenizer_sha256")
    commit = _commit(source_commit)
    batch_size = _batch_size(materialization_batch_size)
    for provider, cache_root, label in ((teacher_provider, teacher_cache_root, "teacher"), (reference_provider, reference_cache_root, "reference"), (document_utility_provider, retriever_utility_cache_root, "document utility")):
        if (provider is None) != (cache_root is None):
            raise ValueError(f"{label} provider/cache root must be configured together")

    root = _empty_root(output_dir, "canonical grounded output")
    root.mkdir(parents=True, exist_ok=True)
    published: list[CanonicalGroundedSplit] = []
    example_sets: dict[str, set[str]] = {}
    total_records = 0
    for split_manifest in source.manifest.splits:
        dataset = source.split(split_manifest.name)
        destination = root / f"{split_manifest.name}.grounded.jsonl"
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=root)
        digest = hashlib.sha256()
        record_ids: list[str] = []
        evidence_ids: list[str] = []
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for index in range(len(dataset)):
                    example = dataset[index]
                    payload = _payload(example, source_manifest_sha256=source.manifest.manifest_digest, source_receipt_sha256=source.receipt.receipt_sha256)
                    parse_authoritative_grounded_example(payload)
                    encoded = _canonical(payload) + b"\n"
                    if len(encoded) > _MAX_LINE_BYTES:
                        raise ValueError("canonical grounded record exceeds line safety bound")
                    handle.write(encoded); digest.update(encoded)
                    record_ids.append(example.example_id)
                    evidence_ids.extend(item.evidence_id for item in example.evidence)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        split_sha = digest.hexdigest()
        if _stream_sha(destination) != split_sha:
            raise RuntimeError("canonical grounded split changed during publication")
        published.append(CanonicalGroundedSplit(split_manifest.name, str(destination), split_sha, len(dataset), _id_digest(record_ids), _id_digest(evidence_ids)))
        example_sets[split_manifest.name] = set(record_ids)
        total_records += len(dataset)

    if total_records <= 0:
        raise ValueError("canonical grounded training data requires at least one record")
    names = sorted(example_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = example_sets[left] & example_sets[right]
            if overlap:
                raise ValueError(f"grounded example-id leakage across {left}/{right}: {sorted(overlap)[:20]}")

    transformation = _digest({"schema": "rigorousrag-canonical-grounded-training-data/v1", "source_manifest_sha256": source.manifest.manifest_digest, "source_import_receipt_sha256": source.receipt.receipt_sha256, "cache_keys": "teacher:<example_id>|utility:<example_id>"})
    manifest = DatasetManifest(
        dataset_id=source.manifest.dataset_id, exact_version=source.manifest.exact_version, source_locator=source.manifest.source_locator,
        artifact_sha256=source.manifest.artifact_sha256, license_identifier=source.manifest.license_identifier, license_status=source.manifest.license_status,
        license_evidence=source.manifest.license_evidence, loader_name="training.grounded_canonical_training_data_pipeline", loader_version="1",
        transformation_sha256=transformation,
        splits=tuple(SplitManifest(name=item.name, content_sha256=item.sha256, record_count=item.record_count, record_id_sha256=item.record_id_sha256, query_id_sha256=item.record_id_sha256, document_id_sha256=item.evidence_id_sha256) for item in published),
        tasks=source.manifest.tasks, modalities=source.manifest.modalities, card=source.manifest.card,
        metadata={**source.manifest.metadata, "canonical_record_kind": "grounded_generation", "source_manifest_sha256": source.manifest.manifest_digest},
    )

    materializer = GroundedSupervisionMaterializer(reference_provider=reference_provider, teacher_provider=teacher_provider, document_utility_provider=document_utility_provider)
    cache_bindings: list[GroundedCacheBinding] = []
    teacher_cache = reference_cache = utility_cache = None

    if teacher_provider is not None:
        contract, producer = _provider(teacher_provider, tokenizer_sha256=tokenizer_sha, label="teacher provider")
        teacher_cache = _cache("teacher_logits", teacher_cache_root, producer_sha256=producer, tokenizer_sha256=tokenizer_sha, dataset_manifest_sha256=manifest.manifest_digest, source_commit=commit, provider_contract_sha256=contract)
    if reference_provider is not None:
        contract, producer = _provider(reference_provider, tokenizer_sha256=tokenizer_sha, label="reference provider")
        reference_cache = _cache("reference_policy_log_probs", reference_cache_root, producer_sha256=producer, tokenizer_sha256=tokenizer_sha, dataset_manifest_sha256=manifest.manifest_digest, source_commit=commit, provider_contract_sha256=contract)
    if document_utility_provider is not None:
        contract, producer = _provider(document_utility_provider, tokenizer_sha256=tokenizer_sha, label="document utility provider")
        utility_cache = _cache("document_lm_utility", retriever_utility_cache_root, producer_sha256=producer, tokenizer_sha256=tokenizer_sha, dataset_manifest_sha256=manifest.manifest_digest, source_commit=commit, provider_contract_sha256=contract)

    if teacher_cache is not None or reference_cache is not None or utility_cache is not None:
        for batch in _example_batches(published, dataset_manifest_sha256=manifest.manifest_digest, batch_size=batch_size):
            if teacher_cache is not None:
                materialize_teacher_logit_cache(batch, materializer=materializer, cache=teacher_cache)
            if reference_cache is not None:
                materialize_reference_policy_cache(batch, materializer=materializer, cache=reference_cache)
            if utility_cache is not None:
                materialize_document_utility_cache(batch, materializer=materializer, cache=utility_cache)

    teacher_contract = teacher_cache.seal() if teacher_cache is not None else None
    reference_contract = reference_cache.seal() if reference_cache is not None else None
    utility_contract = utility_cache.seal() if utility_cache is not None else None

    if teacher_cache is not None and teacher_contract is not None:
        cache_bindings.append(GroundedCacheBinding("teacher_logits", str(teacher_cache.root), teacher_cache.identity.digest, teacher_contract, teacher_cache.identity.producer_sha256))
    if reference_cache is not None and reference_contract is not None:
        cache_bindings.append(GroundedCacheBinding("reference_policy_log_probs", str(reference_cache.root), reference_cache.identity.digest, reference_contract, reference_cache.identity.producer_sha256))
    if utility_cache is not None and utility_contract is not None:
        cache_bindings.append(GroundedCacheBinding("document_lm_utility", str(utility_cache.root), utility_cache.identity.digest, utility_contract, utility_cache.identity.producer_sha256))

    unsigned = {
        "schema": "rigorousrag-canonical-grounded-training-data-receipt/v1",
        "source_manifest_sha256": source.manifest.manifest_digest,
        "source_import_receipt_sha256": source.receipt.receipt_sha256,
        "dataset_manifest_sha256": manifest.manifest_digest,
        "transformation_sha256": transformation,
        "splits": [asdict(item) for item in published],
        "caches": [asdict(item) for item in cache_bindings],
    }
    receipt = CanonicalGroundedTrainingDataReceipt(source.manifest.manifest_digest, source.receipt.receipt_sha256, manifest.manifest_digest, transformation, tuple(published), tuple(cache_bindings), _digest(unsigned))
    _atomic(root / "dataset_manifest.json", _canonical({"schema": "rigorousrag-dataset-manifest/v1", "manifest": asdict(manifest), "manifest_sha256": manifest.manifest_digest}) + b"\n")
    _atomic(root / "canonical_receipt.json", _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n")
    return CanonicalGroundedTrainingDataResult(manifest, tuple(published), teacher_cache, reference_cache, utility_cache, receipt)


__all__ = ["CanonicalGroundedTrainingDataReceipt", "CanonicalGroundedTrainingDataResult", "CanonicalGroundedSplit", "GroundedCacheBinding", "build_canonical_grounded_training_data"]
