"""Atomic, corpus-scale canonical grounded-RAG training-data authority v2.

This module preserves the v1 grounded supervision semantics while replacing its authority
mechanics. Logical split names never become filesystem paths; example/evidence identities and
cross-split uniqueness are disk-backed; optional teacher/reference/document-utility caches use a
disk-backed sealed key authority; and the whole dataset + caches + manifest + receipt is built in
a sibling staging directory and atomically renamed only after all materialization succeeds.
Restart verification re-parses every split, recomputes every identity digest, reconstructs every
cache identity/content contract, and rejects unexpected or symlinked publication children.
Importing this module performs no model execution, download, retrieval or training.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from evaluation.dataset_governance import (
    DatasetCard,
    DatasetManifest,
    DatasetModality,
    DatasetTask,
    LicenseStatus,
    SplitManifest,
)
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
from training.advanced_rag_supervision import (
    GroundedSupervisionMaterializer,
    SupervisionCacheIdentity,
)
from training.disk_backed_supervision_cache import (
    DiskBackedAuthoritativeSafetensorCache,
)
from training.governed_grounded_io import VerifiedGovernedGroundedDataset
from training.sqlite_identity_ledger import SqliteIdentityLedger

_HEX = frozenset("0123456789abcdef")
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_BATCH = 4096
_MAX_SPLITS = 100_000
_CACHE_KINDS = (
    "teacher_logits",
    "reference_policy_log_probs",
    "document_lm_utility",
)


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


def _text(value: Any, label: str, maximum: int = 4000) -> str:
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


def _batch_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_BATCH:
        raise ValueError(f"materialization_batch_size must lie in [1,{_MAX_BATCH}]")
    return value


def _split_filename(name: str) -> str:
    selected = _text(name, "split name")
    return f"split-{hashlib.sha256(selected.encode('utf-8')).hexdigest()}.grounded.jsonl"


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _span(span: Any) -> Mapping[str, int]:
    return {"start": int(span.start), "end": int(span.end)}


def _payload(
    example: Any,
    *,
    source_manifest_sha256: str,
    source_receipt_sha256: str,
) -> Mapping[str, Any]:
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
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "text": item.text,
                "source_id": item.source_id,
            }
            for item in example.evidence
        ],
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


def _manifest(value: Any) -> DatasetManifest:
    if not isinstance(value, Mapping):
        raise ValueError("grounded canonical dataset manifest must be an object")
    expected = {
        "dataset_id",
        "exact_version",
        "source_locator",
        "artifact_sha256",
        "license_identifier",
        "license_status",
        "license_evidence",
        "loader_name",
        "loader_version",
        "transformation_sha256",
        "splits",
        "tasks",
        "modalities",
        "card",
        "metadata",
    }
    if set(value) != expected:
        raise ValueError("grounded canonical dataset manifest fields differ from DatasetManifest")
    card_raw = value["card"]
    if not isinstance(card_raw, Mapping):
        raise ValueError("grounded canonical dataset card must be an object")
    card = DatasetCard(
        summary=card_raw["summary"],
        intended_uses=tuple(card_raw["intended_uses"]),
        forbidden_uses=tuple(card_raw["forbidden_uses"]),
        populations_or_domains=tuple(card_raw["populations_or_domains"]),
        languages=tuple(card_raw["languages"]),
        pii_notes=card_raw["pii_notes"],
        safety_notes=card_raw["safety_notes"],
        source_citation=card_raw["source_citation"],
        known_limitations=tuple(card_raw["known_limitations"]),
    )
    splits_raw = value["splits"]
    if not isinstance(splits_raw, list):
        raise ValueError("grounded canonical manifest splits must be an array")
    splits = tuple(
        SplitManifest(**dict(item)) for item in splits_raw if isinstance(item, Mapping)
    )
    if len(splits) != len(splits_raw):
        raise ValueError("grounded canonical split entries must be objects")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("grounded canonical metadata must be an object")
    return DatasetManifest(
        dataset_id=value["dataset_id"],
        exact_version=value["exact_version"],
        source_locator=value["source_locator"],
        artifact_sha256=value["artifact_sha256"],
        license_identifier=value["license_identifier"],
        license_status=LicenseStatus(value["license_status"]),
        license_evidence=value["license_evidence"],
        loader_name=value["loader_name"],
        loader_version=value["loader_version"],
        transformation_sha256=value["transformation_sha256"],
        splits=splits,
        tasks=tuple(DatasetTask(item) for item in value["tasks"]),
        modalities=tuple(DatasetModality(item) for item in value["modalities"]),
        card=card,
        metadata={str(key): str(item) for key, item in metadata.items()},
    )


def _provider(
    provider: Any,
    *,
    tokenizer_sha256: str,
    label: str,
) -> tuple[str, str]:
    contract = _sha(getattr(provider, "contract_sha256", None), f"{label} contract_sha256")
    producer = _sha(getattr(provider, "model_sha256", None), f"{label} model_sha256")
    provider_tokenizer = _sha(
        getattr(provider, "tokenizer_sha256", None),
        f"{label} tokenizer_sha256",
    )
    if provider_tokenizer != tokenizer_sha256:
        raise ValueError(f"{label} tokenizer identity differs from canonical tokenizer")
    return contract, producer


def _evidence_payload_sha(evidence: Any) -> str:
    return _digest(
        {
            "evidence_id": evidence.evidence_id,
            "text": evidence.text,
            "source_id": evidence.source_id,
        }
    )


def _key_digest(values: Iterator[str], prefix: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update((prefix + value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _cache_key_digest(cache_root: Path) -> tuple[int, str]:
    authority_db = cache_root / "authority.sqlite"
    if authority_db.is_symlink() or not authority_db.is_file():
        raise ValueError("disk-backed cache authority index is missing or unsafe")
    digest = hashlib.sha256()
    count = 0
    with sqlite3.connect(f"file:{authority_db}?mode=ro", uri=True, timeout=30.0) as connection:
        rows = connection.execute("SELECT key FROM entries ORDER BY key")
        for row in rows:
            digest.update(str(row[0]).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


@dataclass(frozen=True)
class AuthoritativeGroundedCanonicalSplit:
    name: str
    filename: str
    sha256: str
    record_count: int
    record_id_sha256: str
    evidence_id_sha256: str

    def __post_init__(self) -> None:
        name = _text(self.name, "split name")
        object.__setattr__(self, "name", name)
        expected_filename = _split_filename(name)
        if self.filename != expected_filename or Path(self.filename).name != self.filename:
            raise ValueError("grounded canonical physical split filename is invalid")
        for field in ("sha256", "record_id_sha256", "evidence_id_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count < 0
        ):
            raise ValueError("record_count must be a non-negative integer")


@dataclass(frozen=True)
class AuthoritativeGroundedCacheBinding:
    kind: str
    relative_root: str
    producer_sha256: str
    tokenizer_sha256: str
    dataset_manifest_sha256: str
    source_commit: str
    config_sha256: str
    identity_sha256: str
    contract_sha256: str
    authority_json_sha256: str
    authority_db_sha256: str
    entry_count: int
    expected_key_sha256: str

    def __post_init__(self) -> None:
        kind = _text(self.kind, "cache kind", 200)
        if kind not in _CACHE_KINDS:
            raise ValueError("unsupported grounded canonical cache kind")
        object.__setattr__(self, "kind", kind)
        expected_root = f"caches/{kind}"
        if self.relative_root != expected_root or Path(self.relative_root).as_posix() != expected_root:
            raise ValueError("grounded canonical cache relative_root is invalid")
        for field in (
            "producer_sha256",
            "tokenizer_sha256",
            "dataset_manifest_sha256",
            "config_sha256",
            "identity_sha256",
            "contract_sha256",
            "authority_json_sha256",
            "authority_db_sha256",
            "expected_key_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        object.__setattr__(self, "source_commit", _commit(self.source_commit))
        if isinstance(self.entry_count, bool) or not isinstance(self.entry_count, int) or self.entry_count < 0:
            raise ValueError("cache entry_count must be a non-negative integer")
        identity = self.identity()
        if identity.digest != self.identity_sha256:
            raise ValueError("grounded canonical cache identity digest mismatch")

    def identity(self) -> SupervisionCacheIdentity:
        return SupervisionCacheIdentity(
            cache_kind=self.kind,
            producer_sha256=self.producer_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            source_commit=self.source_commit,
            config_sha256=self.config_sha256,
        )


@dataclass(frozen=True)
class AuthoritativeGroundedCanonicalReceipt:
    source_manifest_sha256: str
    source_import_receipt_sha256: str
    dataset_manifest_sha256: str
    transformation_sha256: str
    tokenizer_sha256: str
    source_commit: str
    splits: tuple[AuthoritativeGroundedCanonicalSplit, ...]
    caches: tuple[AuthoritativeGroundedCacheBinding, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "source_manifest_sha256",
            "source_import_receipt_sha256",
            "dataset_manifest_sha256",
            "transformation_sha256",
            "tokenizer_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        object.__setattr__(self, "source_commit", _commit(self.source_commit))
        splits = tuple(self.splits)
        caches = tuple(self.caches)
        if not splits or len(splits) > _MAX_SPLITS:
            raise ValueError("grounded canonical receipt requires bounded non-empty splits")
        if len({item.name for item in splits}) != len(splits):
            raise ValueError("grounded canonical split names must be unique")
        if len({item.filename for item in splits}) != len(splits):
            raise ValueError("grounded canonical physical split filenames must be unique")
        if len({item.kind for item in caches}) != len(caches):
            raise ValueError("grounded canonical cache kinds must be unique")
        object.__setattr__(self, "splits", tuple(sorted(splits, key=lambda item: item.name)))
        object.__setattr__(self, "caches", tuple(sorted(caches, key=lambda item: item.kind)))
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("grounded canonical v2 receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-grounded-canonical-receipt/v2",
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_import_receipt_sha256": self.source_import_receipt_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "transformation_sha256": self.transformation_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "source_commit": self.source_commit,
            "splits": [asdict(item) for item in self.splits],
            "caches": [asdict(item) for item in self.caches],
        }


@dataclass(frozen=True)
class VerifiedAuthoritativeGroundedCanonicalData:
    root: str
    manifest: DatasetManifest
    receipt: AuthoritativeGroundedCanonicalReceipt
    caches: tuple[tuple[str, DiskBackedAuthoritativeSafetensorCache], ...]

    def __post_init__(self) -> None:
        root = safe_advanced_path(
            self.root,
            label="authoritative grounded canonical root",
            must_exist=True,
            require_directory=True,
        )
        object.__setattr__(self, "root", str(root))

    def split(self, name: str) -> ManifestBoundAuthoritativeJsonlDataset:
        selected = [item for item in self.receipt.splits if item.name == name]
        if len(selected) != 1:
            raise ValueError(f"unknown authoritative grounded split {name!r}")
        item = selected[0]
        return ManifestBoundAuthoritativeJsonlDataset(
            Path(self.root) / item.filename,
            expected_sha256=item.sha256,
            dataset_manifest_sha256=self.manifest.manifest_digest,
            split_name=item.name,
            record_kind="grounded_generation",
            expected_record_count=item.record_count,
        )

    def cache(self, kind: str) -> DiskBackedAuthoritativeSafetensorCache:
        selected = [cache for cache_kind, cache in self.caches if cache_kind == kind]
        if len(selected) != 1:
            raise ValueError(f"unknown authoritative grounded cache {kind!r}")
        return selected[0]


def _cache_identity(
    *,
    kind: str,
    provider_contract_sha256: str,
    producer_sha256: str,
    tokenizer_sha256: str,
    dataset_manifest_sha256: str,
    source_commit: str,
) -> SupervisionCacheIdentity:
    config_sha = _digest(
        {
            "schema": "rigorousrag-authoritative-grounded-cache-config/v2",
            "kind": kind,
            "provider_contract_sha256": provider_contract_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
    )
    return SupervisionCacheIdentity(
        cache_kind=kind,
        producer_sha256=producer_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_commit=source_commit,
        config_sha256=config_sha,
    )


def _batches(
    root: Path,
    splits: Sequence[AuthoritativeGroundedCanonicalSplit],
    *,
    manifest_sha256: str,
    batch_size: int,
) -> Iterator[tuple[Any, ...]]:
    pending: list[Any] = []
    for item in splits:
        dataset = ManifestBoundAuthoritativeJsonlDataset(
            root / item.filename,
            expected_sha256=item.sha256,
            dataset_manifest_sha256=manifest_sha256,
            split_name=item.name,
            record_kind="grounded_generation",
            expected_record_count=item.record_count,
        )
        for index in range(len(dataset)):
            pending.append(dataset[index])
            if len(pending) == batch_size:
                yield tuple(pending)
                pending.clear()
    if pending:
        yield tuple(pending)


def _cache_binding(
    *,
    kind: str,
    cache: DiskBackedAuthoritativeSafetensorCache,
    expected_key_sha256: str,
) -> AuthoritativeGroundedCacheBinding:
    authority_json_sha, authority_db_sha = cache.authority_file_sha256s
    return AuthoritativeGroundedCacheBinding(
        kind=kind,
        relative_root=f"caches/{kind}",
        producer_sha256=cache.identity.producer_sha256,
        tokenizer_sha256=cache.identity.tokenizer_sha256,
        dataset_manifest_sha256=cache.identity.dataset_manifest_sha256,
        source_commit=cache.identity.source_commit,
        config_sha256=cache.identity.config_sha256,
        identity_sha256=cache.identity.digest,
        contract_sha256=cache.contract_sha256,
        authority_json_sha256=authority_json_sha,
        authority_db_sha256=authority_db_sha,
        entry_count=cache.entry_count,
        expected_key_sha256=expected_key_sha256,
    )


def build_authoritative_grounded_canonical_training_data(
    source: VerifiedGovernedGroundedDataset,
    *,
    tokenizer_sha256: str,
    source_commit: str,
    output_dir: str | Path,
    teacher_provider: Any | None = None,
    reference_provider: Any | None = None,
    document_utility_provider: Any | None = None,
    materialization_batch_size: int = 8,
) -> VerifiedAuthoritativeGroundedCanonicalData:
    if not isinstance(source, VerifiedGovernedGroundedDataset):
        raise ValueError("source must be VerifiedGovernedGroundedDataset")
    tokenizer_sha = _sha(tokenizer_sha256, "tokenizer_sha256")
    commit = _commit(source_commit)
    batch_size = _batch_size(materialization_batch_size)
    providers = {
        "teacher_logits": teacher_provider,
        "reference_policy_log_probs": reference_provider,
        "document_lm_utility": document_utility_provider,
    }

    root = safe_advanced_path(
        output_dir,
        label="authoritative grounded canonical output",
        must_exist=False,
    )
    if root.exists():
        raise ValueError("authoritative grounded canonical output must not already exist")
    parent = safe_advanced_path(
        root.parent,
        label="authoritative grounded canonical parent",
        must_exist=True,
        require_directory=True,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'grounded'}-stage-", dir=parent))
    ledger_path = stage / ".identity.sqlite"
    published = False
    ledger: SqliteIdentityLedger | None = None
    try:
        ledger = SqliteIdentityLedger(ledger_path)
        splits: list[AuthoritativeGroundedCanonicalSplit] = []
        total_records = 0
        source_splits = tuple(source.manifest.splits)
        if not source_splits or len(source_splits) > _MAX_SPLITS:
            raise ValueError("source grounded manifest requires bounded non-empty splits")
        if len({item.name for item in source_splits}) != len(source_splits):
            raise ValueError("source grounded manifest split names must be unique")

        for split_manifest in source_splits:
            dataset = source.split(split_manifest.name)
            filename = _split_filename(split_manifest.name)
            destination = stage / filename
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}-",
                suffix=".tmp",
                dir=stage,
            )
            temporary = Path(temporary_name)
            split_digest = hashlib.sha256()
            count = 0
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    for index in range(len(dataset)):
                        example = dataset[index]
                        payload = _payload(
                            example,
                            source_manifest_sha256=source.manifest.manifest_digest,
                            source_receipt_sha256=source.receipt.receipt_sha256,
                        )
                        parsed = parse_authoritative_grounded_example(payload)
                        encoded = _canonical(payload) + b"\n"
                        if len(encoded) > _MAX_LINE_BYTES:
                            raise ValueError("canonical grounded record exceeds line safety bound")
                        payload_sha = hashlib.sha256(encoded).hexdigest()
                        ledger.add_unique(
                            "grounded-example",
                            split_manifest.name,
                            parsed.example_id,
                            payload_sha256=payload_sha,
                        )
                        for evidence in parsed.evidence:
                            ledger.add_set(
                                "grounded-evidence",
                                split_manifest.name,
                                evidence.evidence_id,
                                payload_sha256=_evidence_payload_sha(evidence),
                            )
                        handle.write(encoded)
                        split_digest.update(encoded)
                        count += 1
                        if count % 10_000 == 0:
                            ledger.commit()
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            ledger.commit()
            if count != len(dataset) or count != split_manifest.record_count:
                raise ValueError(
                    f"grounded split {split_manifest.name!r} record count changed during canonical publication"
                )
            split_sha = split_digest.hexdigest()
            if _stream_sha(destination) != split_sha:
                raise RuntimeError("canonical grounded split bytes changed during publication")
            record_digest = ledger.digest_unique(
                "grounded-example",
                scope=split_manifest.name,
            )
            evidence_digest = ledger.digest_set(
                "grounded-evidence",
                scope=split_manifest.name,
            )
            splits.append(
                AuthoritativeGroundedCanonicalSplit(
                    name=split_manifest.name,
                    filename=filename,
                    sha256=split_sha,
                    record_count=count,
                    record_id_sha256=record_digest,
                    evidence_id_sha256=evidence_digest,
                )
            )
            total_records += count

        if total_records <= 0:
            raise ValueError("authoritative grounded canonical data requires at least one record")
        if ledger.count_unique("grounded-example") != total_records:
            raise RuntimeError("global grounded example identity count differs from published records")

        transformation = _digest(
            {
                "schema": "rigorousrag-authoritative-grounded-canonical-transformation/v2",
                "source_manifest_sha256": source.manifest.manifest_digest,
                "source_import_receipt_sha256": source.receipt.receipt_sha256,
                "physical_split_policy": "sha256(logical_split_name)",
                "record_authority": "global-example-sqlite-uniqueness+scoped-evidence-set",
                "cache_keys": "reference=<example_id>|teacher=teacher:<example_id>|utility=utility:<example_id>",
            }
        )
        ordered_splits = tuple(sorted(splits, key=lambda item: item.name))
        manifest = DatasetManifest(
            dataset_id=source.manifest.dataset_id,
            exact_version=source.manifest.exact_version,
            source_locator=source.manifest.source_locator,
            artifact_sha256=source.manifest.artifact_sha256,
            license_identifier=source.manifest.license_identifier,
            license_status=source.manifest.license_status,
            license_evidence=source.manifest.license_evidence,
            loader_name="training.authoritative_grounded_canonical_training_data",
            loader_version="2",
            transformation_sha256=transformation,
            splits=tuple(
                SplitManifest(
                    name=item.name,
                    content_sha256=item.sha256,
                    record_count=item.record_count,
                    record_id_sha256=item.record_id_sha256,
                    query_id_sha256=item.record_id_sha256,
                    document_id_sha256=item.evidence_id_sha256,
                )
                for item in ordered_splits
            ),
            tasks=source.manifest.tasks,
            modalities=source.manifest.modalities,
            card=source.manifest.card,
            metadata={
                **source.manifest.metadata,
                "canonical_record_kind": "grounded_generation",
                "source_manifest_sha256": source.manifest.manifest_digest,
                "canonical_authority": "grounded-v2-atomic-disk-backed",
            },
        )
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

        caches: dict[str, DiskBackedAuthoritativeSafetensorCache] = {}
        materializer = GroundedSupervisionMaterializer(
            reference_provider=reference_provider,
            teacher_provider=teacher_provider,
            document_utility_provider=document_utility_provider,
        )
        for kind, provider in providers.items():
            if provider is None:
                continue
            contract_sha, producer_sha = _provider(
                provider,
                tokenizer_sha256=tokenizer_sha,
                label=f"{kind} provider",
            )
            identity = _cache_identity(
                kind=kind,
                provider_contract_sha256=contract_sha,
                producer_sha256=producer_sha,
                tokenizer_sha256=tokenizer_sha,
                dataset_manifest_sha256=manifest.manifest_digest,
                source_commit=commit,
            )
            caches[kind] = DiskBackedAuthoritativeSafetensorCache(
                stage / "caches" / kind,
                identity,
            )

        if caches:
            for batch in _batches(
                stage,
                ordered_splits,
                manifest_sha256=manifest.manifest_digest,
                batch_size=batch_size,
            ):
                if "teacher_logits" in caches:
                    materialize_teacher_logit_cache(
                        batch,
                        materializer=materializer,
                        cache=caches["teacher_logits"],
                    )
                if "reference_policy_log_probs" in caches:
                    materialize_reference_policy_cache(
                        batch,
                        materializer=materializer,
                        cache=caches["reference_policy_log_probs"],
                    )
                if "document_lm_utility" in caches:
                    materialize_document_utility_cache(
                        batch,
                        materializer=materializer,
                        cache=caches["document_lm_utility"],
                    )

        cache_bindings: list[AuthoritativeGroundedCacheBinding] = []
        prefixes = {
            "teacher_logits": "teacher:",
            "reference_policy_log_probs": "",
            "document_lm_utility": "utility:",
        }
        for kind in sorted(caches):
            cache = caches[kind]
            cache.seal()
            if cache.entry_count != total_records:
                raise ValueError(
                    f"grounded cache {kind} entry count differs from canonical record count"
                )
            expected_key_sha = _key_digest(
                ledger.iter_unique("grounded-example"),
                prefixes[kind],
            )
            actual_count, actual_key_sha = _cache_key_digest(cache.root)
            if actual_count != total_records or actual_key_sha != expected_key_sha:
                raise ValueError(f"grounded cache {kind} key universe differs from canonical examples")
            cache_bindings.append(
                _cache_binding(
                    kind=kind,
                    cache=cache,
                    expected_key_sha256=expected_key_sha,
                )
            )

        ledger.close()
        ledger = None
        for suffix in ("", "-wal", "-shm"):
            Path(str(ledger_path) + suffix).unlink(missing_ok=True)

        unsigned = {
            "schema": "rigorousrag-authoritative-grounded-canonical-receipt/v2",
            "source_manifest_sha256": source.manifest.manifest_digest,
            "source_import_receipt_sha256": source.receipt.receipt_sha256,
            "dataset_manifest_sha256": manifest.manifest_digest,
            "transformation_sha256": transformation,
            "tokenizer_sha256": tokenizer_sha,
            "source_commit": commit,
            "splits": [asdict(item) for item in ordered_splits],
            "caches": [asdict(item) for item in sorted(cache_bindings, key=lambda item: item.kind)],
        }
        receipt = AuthoritativeGroundedCanonicalReceipt(
            source_manifest_sha256=source.manifest.manifest_digest,
            source_import_receipt_sha256=source.receipt.receipt_sha256,
            dataset_manifest_sha256=manifest.manifest_digest,
            transformation_sha256=transformation,
            tokenizer_sha256=tokenizer_sha,
            source_commit=commit,
            splits=ordered_splits,
            caches=tuple(cache_bindings),
            receipt_sha256=_digest(unsigned),
        )
        _atomic(
            stage / "canonical_receipt.json",
            _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256})
            + b"\n",
        )

        expected_top = {
            "dataset_manifest.json",
            "canonical_receipt.json",
            *(item.filename for item in ordered_splits),
        }
        if cache_bindings:
            expected_top.add("caches")
        if {item.name for item in stage.iterdir()} != expected_top:
            raise RuntimeError("grounded canonical staging directory is not closed")
        os.replace(stage, root)
        published = True
        _fsync_directory(parent)
        try:
            verified = verify_authoritative_grounded_canonical_training_data(
                root / "canonical_receipt.json"
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            published = False
            raise
        if verified.receipt.receipt_sha256 != receipt.receipt_sha256:
            shutil.rmtree(root, ignore_errors=True)
            published = False
            raise RuntimeError("grounded canonical identity changed after atomic publication")
        return verified
    finally:
        if ledger is not None:
            ledger.close()
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _parse_split(raw: Any) -> AuthoritativeGroundedCanonicalSplit:
    required = {
        "name",
        "filename",
        "sha256",
        "record_count",
        "record_id_sha256",
        "evidence_id_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("grounded canonical split receipt fields are invalid")
    return AuthoritativeGroundedCanonicalSplit(**dict(raw))


def _parse_cache(raw: Any) -> AuthoritativeGroundedCacheBinding:
    required = {
        "kind",
        "relative_root",
        "producer_sha256",
        "tokenizer_sha256",
        "dataset_manifest_sha256",
        "source_commit",
        "config_sha256",
        "identity_sha256",
        "contract_sha256",
        "authority_json_sha256",
        "authority_db_sha256",
        "entry_count",
        "expected_key_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("grounded canonical cache receipt fields are invalid")
    return AuthoritativeGroundedCacheBinding(**dict(raw))


def verify_authoritative_grounded_canonical_training_data(
    receipt_path: str | Path,
) -> VerifiedAuthoritativeGroundedCanonicalData:
    raw_receipt_path = Path(receipt_path).expanduser()
    if raw_receipt_path.is_symlink():
        raise ValueError("grounded canonical receipt may not be a symlink")
    selected_receipt = safe_advanced_path(
        raw_receipt_path,
        label="authoritative grounded canonical receipt",
        must_exist=True,
        require_file=True,
    )
    root = selected_receipt.parent
    if selected_receipt != root / "canonical_receipt.json":
        raise ValueError("grounded canonical receipt must use canonical filename")
    raw = _strict_json(selected_receipt, "authoritative grounded canonical receipt")
    required = {
        "schema",
        "source_manifest_sha256",
        "source_import_receipt_sha256",
        "dataset_manifest_sha256",
        "transformation_sha256",
        "tokenizer_sha256",
        "source_commit",
        "splits",
        "caches",
        "receipt_sha256",
    }
    if (
        set(raw) != required
        or raw.get("schema") != "rigorousrag-authoritative-grounded-canonical-receipt/v2"
        or not isinstance(raw.get("splits"), list)
        or not isinstance(raw.get("caches"), list)
    ):
        raise ValueError("unsupported authoritative grounded canonical receipt schema")
    receipt = AuthoritativeGroundedCanonicalReceipt(
        source_manifest_sha256=raw["source_manifest_sha256"],
        source_import_receipt_sha256=raw["source_import_receipt_sha256"],
        dataset_manifest_sha256=raw["dataset_manifest_sha256"],
        transformation_sha256=raw["transformation_sha256"],
        tokenizer_sha256=raw["tokenizer_sha256"],
        source_commit=raw["source_commit"],
        splits=tuple(_parse_split(item) for item in raw["splits"]),
        caches=tuple(_parse_cache(item) for item in raw["caches"]),
        receipt_sha256=raw["receipt_sha256"],
    )

    expected_top = {
        "dataset_manifest.json",
        "canonical_receipt.json",
        *(item.filename for item in receipt.splits),
    }
    if receipt.caches:
        expected_top.add("caches")
    actual_top = {item.name for item in root.iterdir()}
    if actual_top != expected_top:
        raise ValueError("grounded canonical publication directory is not closed")
    for name in expected_top:
        child = root / name
        if child.is_symlink():
            raise ValueError(f"grounded canonical publication child {name!r} may not be a symlink")

    manifest_raw = _strict_json(root / "dataset_manifest.json", "grounded canonical manifest")
    if (
        set(manifest_raw) != {"schema", "manifest", "manifest_sha256"}
        or manifest_raw.get("schema") != "rigorousrag-dataset-manifest/v1"
    ):
        raise ValueError("unsupported grounded canonical manifest envelope")
    manifest = _manifest(manifest_raw["manifest"])
    if manifest.manifest_digest != _sha(manifest_raw["manifest_sha256"], "manifest_sha256"):
        raise ValueError("grounded canonical manifest envelope digest mismatch")
    if manifest.manifest_digest != receipt.dataset_manifest_sha256:
        raise ValueError("grounded canonical manifest differs from receipt")
    if manifest.transformation_sha256 != receipt.transformation_sha256:
        raise ValueError("grounded canonical transformation differs from receipt")
    if (
        manifest.loader_name != "training.authoritative_grounded_canonical_training_data"
        or manifest.loader_version != "2"
    ):
        raise ValueError("grounded canonical manifest loader authority is not v2")
    if manifest.metadata.get("source_manifest_sha256") != receipt.source_manifest_sha256:
        raise ValueError("grounded canonical source-manifest lineage differs from receipt")

    ledger_descriptor, ledger_name = tempfile.mkstemp(
        prefix=".grounded-verify-",
        suffix=".sqlite",
        dir=root.parent,
    )
    os.close(ledger_descriptor)
    ledger_path = Path(ledger_name)
    ledger_path.unlink(missing_ok=True)
    ledger = SqliteIdentityLedger(ledger_path)
    cache_objects: list[tuple[str, DiskBackedAuthoritativeSafetensorCache]] = []
    try:
        manifest_splits = {item.name: item for item in manifest.splits}
        if set(manifest_splits) != {item.name for item in receipt.splits}:
            raise ValueError("grounded canonical manifest split universe differs from receipt")
        total_records = 0
        for item in receipt.splits:
            split_path = root / item.filename
            if split_path.is_symlink():
                raise ValueError("grounded canonical split may not be a symlink")
            selected_split = safe_advanced_path(
                split_path,
                label=f"grounded canonical split {item.name}",
                must_exist=True,
                require_file=True,
            )
            if selected_split.parent != root or selected_split.name != item.filename:
                raise ValueError("grounded canonical split escapes publication root")
            if _stream_sha(selected_split) != item.sha256:
                raise ValueError(f"grounded canonical split {item.name} bytes differ from receipt")
            dataset = ManifestBoundAuthoritativeJsonlDataset(
                selected_split,
                expected_sha256=item.sha256,
                dataset_manifest_sha256=manifest.manifest_digest,
                split_name=item.name,
                record_kind="grounded_generation",
                expected_record_count=item.record_count,
            )
            for index in range(len(dataset)):
                example = dataset[index]
                payload = _payload(
                    example,
                    source_manifest_sha256=receipt.source_manifest_sha256,
                    source_receipt_sha256=receipt.source_import_receipt_sha256,
                )
                encoded = _canonical(payload) + b"\n"
                ledger.add_unique(
                    "grounded-example",
                    item.name,
                    example.example_id,
                    payload_sha256=hashlib.sha256(encoded).hexdigest(),
                )
                for evidence in example.evidence:
                    ledger.add_set(
                        "grounded-evidence",
                        item.name,
                        evidence.evidence_id,
                        payload_sha256=_evidence_payload_sha(evidence),
                    )
                if (index + 1) % 10_000 == 0:
                    ledger.commit()
            ledger.commit()
            if ledger.count_unique("grounded-example", scope=item.name) != item.record_count:
                raise ValueError(f"grounded canonical split {item.name} record identity count differs")
            if ledger.digest_unique("grounded-example", scope=item.name) != item.record_id_sha256:
                raise ValueError(f"grounded canonical split {item.name} record digest differs")
            if ledger.digest_set("grounded-evidence", scope=item.name) != item.evidence_id_sha256:
                raise ValueError(f"grounded canonical split {item.name} evidence digest differs")
            manifest_split = manifest_splits[item.name]
            if (
                manifest_split.content_sha256 != item.sha256
                or manifest_split.record_count != item.record_count
                or manifest_split.record_id_sha256 != item.record_id_sha256
                or manifest_split.query_id_sha256 != item.record_id_sha256
                or manifest_split.document_id_sha256 != item.evidence_id_sha256
            ):
                raise ValueError(f"grounded canonical split {item.name} differs from manifest")
            total_records += item.record_count
        if total_records <= 0 or ledger.count_unique("grounded-example") != total_records:
            raise ValueError("grounded canonical global example authority differs from receipt")

        if receipt.caches:
            caches_root = root / "caches"
            if caches_root.is_symlink() or not caches_root.is_dir():
                raise ValueError("grounded canonical caches root is unsafe")
            expected_cache_dirs = {item.kind for item in receipt.caches}
            if {item.name for item in caches_root.iterdir()} != expected_cache_dirs:
                raise ValueError("grounded canonical cache directory is not closed")
        prefixes = {
            "teacher_logits": "teacher:",
            "reference_policy_log_probs": "",
            "document_lm_utility": "utility:",
        }
        for binding in receipt.caches:
            if binding.dataset_manifest_sha256 != manifest.manifest_digest:
                raise ValueError("grounded canonical cache manifest identity differs")
            if binding.tokenizer_sha256 != receipt.tokenizer_sha256:
                raise ValueError("grounded canonical cache tokenizer identity differs")
            if binding.source_commit != receipt.source_commit:
                raise ValueError("grounded canonical cache source revision differs")
            cache_root = root / binding.relative_root
            if cache_root.is_symlink():
                raise ValueError("grounded canonical cache root may not be a symlink")
            selected_cache_root = safe_advanced_path(
                cache_root,
                label=f"grounded canonical {binding.kind} cache",
                must_exist=True,
                require_directory=True,
            )
            if selected_cache_root.parent.parent != root or selected_cache_root.name != binding.kind:
                raise ValueError("grounded canonical cache root escapes publication authority")
            cache = DiskBackedAuthoritativeSafetensorCache(
                selected_cache_root,
                binding.identity(),
            )
            if cache.assert_sealed_integrity() != binding.contract_sha256:
                raise ValueError(f"grounded canonical cache {binding.kind} contract differs")
            authority_json_sha, authority_db_sha = cache.authority_file_sha256s
            if (
                authority_json_sha != binding.authority_json_sha256
                or authority_db_sha != binding.authority_db_sha256
            ):
                raise ValueError(f"grounded canonical cache {binding.kind} authority bytes differ")
            expected_key_sha = _key_digest(
                ledger.iter_unique("grounded-example"),
                prefixes[binding.kind],
            )
            actual_count, actual_key_sha = _cache_key_digest(selected_cache_root)
            if (
                binding.entry_count != total_records
                or cache.entry_count != total_records
                or actual_count != total_records
                or binding.expected_key_sha256 != expected_key_sha
                or actual_key_sha != expected_key_sha
            ):
                raise ValueError(f"grounded canonical cache {binding.kind} key universe differs")
            cache_objects.append((binding.kind, cache))
        return VerifiedAuthoritativeGroundedCanonicalData(
            root=str(root),
            manifest=manifest,
            receipt=receipt,
            caches=tuple(sorted(cache_objects, key=lambda item: item[0])),
        )
    finally:
        ledger.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(ledger_path) + suffix).unlink(missing_ok=True)


__all__ = [
    "AuthoritativeGroundedCacheBinding",
    "AuthoritativeGroundedCanonicalReceipt",
    "AuthoritativeGroundedCanonicalSplit",
    "VerifiedAuthoritativeGroundedCanonicalData",
    "build_authoritative_grounded_canonical_training_data",
    "verify_authoritative_grounded_canonical_training_data",
]
