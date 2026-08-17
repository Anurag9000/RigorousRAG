"""Dataset governance contracts for reproducible retrieval and multimodal evaluation.

This module deliberately separates *benchmark proposals* from *promotable manifests*.
Names such as BEIR or HotpotQA are useful planning targets, but a production experiment
must bind an exact version, license decision, immutable content digest, split digests,
loader/transformation versions and leakage checks before it may be promoted.

No dataset is downloaded, licensed, or asserted to have a particular checksum merely by
appearing in the proposal catalog below.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

_SHA256_LEN = 64
_MAX_SPLITS = 100
_MAX_METADATA = 2_000


def _text(value: Any, label: str, maximum: int = 10_000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if (not allow_empty and not result) or len(result) > maximum:
        raise ValueError(f"{label} is empty or too long")
    if any(ord(ch) == 0 for ch in result):
        raise ValueError(f"{label} contains NUL")
    return result


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    result = _text(value, label, maximum)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} contains control characters")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _identifier(value, label, _SHA256_LEN).lower()
    if len(result) != _SHA256_LEN or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DatasetTask(str, Enum):
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    QUESTION_ANSWERING = "question_answering"
    MULTI_HOP = "multi_hop"
    FACT_VERIFICATION = "fact_verification"
    CITATION = "citation"
    TABLE_QA = "table_qa"
    CHART_QA = "chart_qa"
    DOCUMENT_QA = "document_qa"
    MULTIMODAL_RETRIEVAL = "multimodal_retrieval"
    ADVERSARIAL_SECURITY = "adversarial_security"
    DOMAIN_SPECIFIC = "domain_specific"


class DatasetModality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    PDF = "pdf"
    MULTIMODAL = "multimodal"
    GRAPH = "graph"


class LicenseStatus(str, Enum):
    VERIFIED_ALLOWED = "verified_allowed"
    VERIFIED_RESTRICTED = "verified_restricted"
    UNKNOWN = "unknown"
    PROHIBITED = "prohibited"


class LeakageSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class SplitManifest:
    name: str
    content_sha256: str
    record_count: int
    record_id_sha256: str
    source_group_sha256: str | None = None
    query_id_sha256: str | None = None
    document_id_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "split name", 200))
        for field_name in ("content_sha256", "record_id_sha256"):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        for field_name in ("source_group_sha256", "query_id_sha256", "document_id_sha256"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _sha256(value, field_name))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count < 0:
            raise ValueError("record_count must be a non-negative integer")


@dataclass(frozen=True)
class DatasetCard:
    summary: str
    intended_uses: tuple[str, ...]
    forbidden_uses: tuple[str, ...] = ()
    populations_or_domains: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    pii_notes: str | None = None
    safety_notes: str | None = None
    source_citation: str | None = None
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _text(self.summary, "summary", 100_000))
        for name in (
            "intended_uses",
            "forbidden_uses",
            "populations_or_domains",
            "languages",
            "known_limitations",
        ):
            values = getattr(self, name)
            if len(values) > 1_000:
                raise ValueError(f"{name} is too large")
            object.__setattr__(self, name, tuple(_text(value, name, 10_000) for value in values))
        if not self.intended_uses:
            raise ValueError("at least one intended use is required")
        for name in ("pii_notes", "safety_notes", "source_citation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name, 100_000))


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    exact_version: str
    source_locator: str
    artifact_sha256: str
    license_identifier: str
    license_status: LicenseStatus
    license_evidence: str
    loader_name: str
    loader_version: str
    transformation_sha256: str
    splits: tuple[SplitManifest, ...]
    tasks: tuple[DatasetTask, ...]
    modalities: tuple[DatasetModality, ...]
    card: DatasetCard
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _identifier(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "exact_version", _identifier(self.exact_version, "exact_version", 1_000))
        object.__setattr__(self, "source_locator", _text(self.source_locator, "source_locator", 5_000))
        object.__setattr__(self, "artifact_sha256", _sha256(self.artifact_sha256, "artifact_sha256"))
        object.__setattr__(self, "license_identifier", _identifier(self.license_identifier, "license_identifier", 1_000))
        if not isinstance(self.license_status, LicenseStatus):
            object.__setattr__(self, "license_status", LicenseStatus(self.license_status))
        object.__setattr__(self, "license_evidence", _text(self.license_evidence, "license_evidence", 100_000))
        object.__setattr__(self, "loader_name", _identifier(self.loader_name, "loader_name", 1_000))
        object.__setattr__(self, "loader_version", _identifier(self.loader_version, "loader_version", 1_000))
        object.__setattr__(
            self,
            "transformation_sha256",
            _sha256(self.transformation_sha256, "transformation_sha256"),
        )
        if not self.splits or len(self.splits) > _MAX_SPLITS:
            raise ValueError("splits must be a non-empty bounded tuple")
        if any(not isinstance(split, SplitManifest) for split in self.splits):
            raise ValueError("splits must contain SplitManifest values")
        split_names = [split.name for split in self.splits]
        if len(set(split_names)) != len(split_names):
            raise ValueError("split names must be unique")
        if not self.tasks or any(not isinstance(task, DatasetTask) for task in self.tasks):
            raise ValueError("tasks must contain DatasetTask values")
        if not self.modalities or any(not isinstance(modality, DatasetModality) for modality in self.modalities):
            raise ValueError("modalities must contain DatasetModality values")
        if not isinstance(self.card, DatasetCard):
            raise ValueError("card must be DatasetCard")
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > _MAX_METADATA:
            raise ValueError("metadata must be a bounded mapping")
        cleaned: dict[str, str] = {}
        for key, value in self.metadata.items():
            cleaned[_identifier(key, "metadata key", 300)] = _text(value, "metadata value", 20_000)
        object.__setattr__(self, "metadata", cleaned)

    @property
    def manifest_digest(self) -> str:
        return canonical_digest(asdict(self))

    def assert_promotable(self) -> None:
        """Reject a dataset manifest that is not strong enough for governed promotion."""

        if self.license_status != LicenseStatus.VERIFIED_ALLOWED:
            raise ValueError("dataset promotion requires a verified-allowed license decision")
        placeholders = {"unknown", "latest", "main", "master", "head", "tbd", "todo", "n/a", "none"}
        for label, value in (
            ("exact_version", self.exact_version),
            ("license_identifier", self.license_identifier),
            ("loader_version", self.loader_version),
        ):
            if value.strip().lower() in placeholders:
                raise ValueError(f"{label} must be exact, not a placeholder")
        if not self.license_evidence.strip():
            raise ValueError("license_evidence is required")
        if len({split.content_sha256 for split in self.splits}) != len(self.splits):
            raise ValueError("different named splits may not share the exact same content digest")


@dataclass(frozen=True)
class DatasetProposal:
    """Planning-only benchmark target; deliberately not a DatasetManifest."""

    name: str
    tasks: tuple[DatasetTask, ...]
    modalities: tuple[DatasetModality, ...]
    rationale: str
    family: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "proposal name", 500))
        if not self.tasks or any(not isinstance(task, DatasetTask) for task in self.tasks):
            raise ValueError("proposal tasks are invalid")
        if not self.modalities or any(not isinstance(modality, DatasetModality) for modality in self.modalities):
            raise ValueError("proposal modalities are invalid")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale", 20_000))
        object.__setattr__(self, "family", _identifier(self.family, "family", 500))


@dataclass(frozen=True)
class LeakageFinding:
    left_split: str
    right_split: str
    key_kind: str
    overlaps: tuple[str, ...]
    severity: LeakageSeverity

    def __post_init__(self) -> None:
        for name in ("left_split", "right_split", "key_kind"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 500))
        if not isinstance(self.severity, LeakageSeverity):
            object.__setattr__(self, "severity", LeakageSeverity(self.severity))
        if len(self.overlaps) > 10_000:
            raise ValueError("too many overlap examples")
        object.__setattr__(
            self,
            "overlaps",
            tuple(_identifier(value, "overlap key", 2_000) for value in self.overlaps),
        )


def check_split_leakage(
    split_keys: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    blocking_key_kinds: Sequence[str] = ("record_id", "query_id", "source_group_id"),
    sample_limit: int = 100,
) -> tuple[LeakageFinding, ...]:
    """Detect exact identifier/source-group overlap across supplied splits.

    ``split_keys`` maps split name -> key kind -> values.  The caller chooses the key
    semantics; this function never hashes or guesses content identity.
    """

    if not isinstance(split_keys, Mapping) or len(split_keys) < 2 or len(split_keys) > _MAX_SPLITS:
        raise ValueError("split_keys must contain between two and 100 splits")
    if isinstance(sample_limit, bool) or not isinstance(sample_limit, int) or not 1 <= sample_limit <= 10_000:
        raise ValueError("sample_limit is invalid")
    blocking = {_identifier(value, "blocking key kind", 500) for value in blocking_key_kinds}
    normalized: dict[str, dict[str, set[str]]] = {}
    for split_name, key_groups in split_keys.items():
        selected_split = _identifier(split_name, "split name", 200)
        if not isinstance(key_groups, Mapping) or len(key_groups) > 100:
            raise ValueError("split key groups must be bounded mappings")
        normalized[selected_split] = {}
        for kind, values in key_groups.items():
            selected_kind = _identifier(kind, "key kind", 500)
            if len(values) > _MAX_METADATA * 100_000:
                raise ValueError("split key collection exceeds safety bound")
            normalized[selected_split][selected_kind] = {
                _identifier(value, f"{selected_kind} value", 2_000) for value in values
            }
    findings: list[LeakageFinding] = []
    names = sorted(normalized)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            common_kinds = sorted(set(normalized[left_name]) & set(normalized[right_name]))
            for kind in common_kinds:
                overlap = sorted(normalized[left_name][kind] & normalized[right_name][kind])
                if not overlap:
                    continue
                findings.append(
                    LeakageFinding(
                        left_split=left_name,
                        right_split=right_name,
                        key_kind=kind,
                        overlaps=tuple(overlap[:sample_limit]),
                        severity=LeakageSeverity.BLOCKING if kind in blocking else LeakageSeverity.WARNING,
                    )
                )
    return tuple(findings)


def assert_no_blocking_leakage(findings: Sequence[LeakageFinding]) -> None:
    blocking = [finding for finding in findings if finding.severity == LeakageSeverity.BLOCKING]
    if blocking:
        summary = ", ".join(
            f"{item.left_split}/{item.right_split}:{item.key_kind}" for item in blocking[:20]
        )
        raise ValueError(f"blocking split leakage detected: {summary}")


RECOMMENDED_BENCHMARK_PROPOSALS: tuple[DatasetProposal, ...] = (
    DatasetProposal(
        "BEIR family",
        (DatasetTask.RETRIEVAL,),
        (DatasetModality.TEXT,),
        "Broad zero-shot retrieval coverage across heterogeneous domains; pin individual constituent datasets before use.",
        "retrieval",
    ),
    DatasetProposal(
        "MS MARCO passage/document",
        (DatasetTask.RETRIEVAL, DatasetTask.RERANKING),
        (DatasetModality.TEXT,),
        "Large-scale retrieval and reranking training/evaluation target; exact edition and redistribution terms must be verified.",
        "retrieval",
    ),
    DatasetProposal(
        "Natural Questions / TriviaQA",
        (DatasetTask.QUESTION_ANSWERING, DatasetTask.RETRIEVAL),
        (DatasetModality.TEXT,),
        "Open-domain QA retrieval and answer-support stress testing.",
        "open-domain-qa",
    ),
    DatasetProposal(
        "LoTTE / MIRACL / Mr.TyDi / mMARCO",
        (DatasetTask.RETRIEVAL,),
        (DatasetModality.TEXT,),
        "Long-tail and multilingual retrieval; use language-stratified reporting rather than one aggregate score.",
        "multilingual-retrieval",
    ),
    DatasetProposal(
        "SciFact / SCIDOCS / TREC-COVID / NFCorpus",
        (DatasetTask.RETRIEVAL, DatasetTask.FACT_VERIFICATION, DatasetTask.CITATION),
        (DatasetModality.TEXT,),
        "Scientific/biomedical evidence retrieval, citation, and claim-support evaluation.",
        "scientific",
    ),
    DatasetProposal(
        "HotpotQA / 2WikiMultiHopQA / MuSiQue",
        (DatasetTask.MULTI_HOP, DatasetTask.QUESTION_ANSWERING),
        (DatasetModality.TEXT,),
        "Multi-hop evidence composition and decomposition evaluation.",
        "multi-hop",
    ),
    DatasetProposal(
        "DocVQA / InfographicVQA / PubTables-1M / PubLayNet / ChartQA",
        (DatasetTask.DOCUMENT_QA, DatasetTask.TABLE_QA, DatasetTask.CHART_QA, DatasetTask.MULTIMODAL_RETRIEVAL),
        (DatasetModality.PDF, DatasetModality.TABLE, DatasetModality.IMAGE, DatasetModality.CHART),
        "Layout, table, chart, OCR, and multimodal evidence retrieval/evaluation.",
        "multimodal-document",
    ),
    DatasetProposal(
        "Repository-owned adversarial corpus",
        (DatasetTask.ADVERSARIAL_SECURITY, DatasetTask.CITATION, DatasetTask.RETRIEVAL),
        (DatasetModality.TEXT, DatasetModality.PDF, DatasetModality.MULTIMODAL),
        "Versioned malformed-file, prompt-injection, citation-spoofing, tenant-isolation and exfiltration regression set.",
        "security",
    ),
)


__all__ = [
    "DatasetCard",
    "DatasetManifest",
    "DatasetModality",
    "DatasetProposal",
    "DatasetTask",
    "LeakageFinding",
    "LeakageSeverity",
    "LicenseStatus",
    "RECOMMENDED_BENCHMARK_PROPOSALS",
    "SplitManifest",
    "assert_no_blocking_leakage",
    "canonical_digest",
    "check_split_leakage",
]
