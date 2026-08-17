"""Governed scientific multi-document benchmark adapter contracts.

A benchmark may combine multiple papers only when provenance remains document/generation
scoped.  This module defines a strict local-record adapter and expected evidence graph for
scientific synthesis, contradiction, method comparison and effect extraction.  It does
not download a public dataset or execute retrieval/models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetManifest

_MAX_FILE_BYTES = 1_000_000_000
_MAX_RECORDS = 10_000_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _text(value: Any, label: str, maximum: int = 200_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise ValueError(f"{label} is empty or too long")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ScientificTaskKind(str, Enum):
    SYNTHESIS = "synthesis"
    CONTRADICTION = "contradiction"
    METHOD_COMPARISON = "method_comparison"
    EFFECT_EXTRACTION = "effect_extraction"
    CITATION_SUPPORT = "citation_support"
    MULTI_HOP = "multi_hop"


class ExpectedRelation(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    QUALIFIES = "qualifies"
    LIMITS = "limits"
    COMPARES = "compares"
    DERIVES = "derives"


@dataclass(frozen=True)
class ScientificDocumentRef:
    document_id: str
    generation_id: str
    content_sha256: str
    source_group_id: str

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id", "source_group_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))


@dataclass(frozen=True)
class ExpectedEvidence:
    evidence_id: str
    document_id: str
    generation_id: str
    locator: str
    relation: ExpectedRelation
    claim_id: str

    def __post_init__(self) -> None:
        for name in ("evidence_id", "document_id", "generation_id", "locator", "claim_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 10_000))
        if not isinstance(self.relation, ExpectedRelation):
            object.__setattr__(self, "relation", ExpectedRelation(self.relation))


@dataclass(frozen=True)
class ScientificMultiDocumentExample:
    example_id: str
    query: str
    task: ScientificTaskKind
    documents: tuple[ScientificDocumentRef, ...]
    expected_answer: str | None = None
    expected_evidence: tuple[ExpectedEvidence, ...] = ()
    strata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _identifier(self.example_id, "example_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        if not isinstance(self.task, ScientificTaskKind):
            object.__setattr__(self, "task", ScientificTaskKind(self.task))
        if len(self.documents) < 2 or len(self.documents) > 10_000:
            raise ValueError("scientific multi-document examples require 2..10,000 documents")
        if any(not isinstance(value, ScientificDocumentRef) for value in self.documents):
            raise ValueError("documents must contain ScientificDocumentRef values")
        identities = {(value.document_id, value.generation_id) for value in self.documents}
        if len(identities) != len(self.documents):
            raise ValueError("document/generation identities must be unique")
        if self.expected_answer is not None:
            object.__setattr__(self, "expected_answer", _text(self.expected_answer, "expected_answer"))
        if len(self.expected_evidence) > 100_000 or any(
            not isinstance(value, ExpectedEvidence) for value in self.expected_evidence
        ):
            raise ValueError("expected_evidence must be bounded ExpectedEvidence records")
        for evidence in self.expected_evidence:
            if (evidence.document_id, evidence.generation_id) not in identities:
                raise ValueError("expected evidence references a document outside the example")
        if not isinstance(self.strata, Mapping) or len(self.strata) > 1_000:
            raise ValueError("strata must be a bounded mapping")
        object.__setattr__(
            self,
            "strata",
            {
                _identifier(key, "stratum key", 300): _identifier(value, "stratum value", 5_000)
                for key, value in self.strata.items()
            },
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ScientificBenchmarkCorpus:
    dataset_manifest: DatasetManifest
    split_name: str
    examples: tuple[ScientificMultiDocumentExample, ...]
    source_path_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_manifest, DatasetManifest):
            raise ValueError("dataset_manifest must be DatasetManifest")
        object.__setattr__(self, "split_name", _identifier(self.split_name, "split_name", 300))
        if self.split_name not in {split.name for split in self.dataset_manifest.splits}:
            raise ValueError("split_name is not declared by dataset manifest")
        if not self.examples or len(self.examples) > _MAX_RECORDS:
            raise ValueError("examples must be non-empty and bounded")
        ids = [example.example_id for example in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("example ids must be unique")
        object.__setattr__(self, "source_path_sha256", _sha256(self.source_path_sha256, "source_path_sha256"))

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "dataset_manifest_digest": self.dataset_manifest.manifest_digest,
                "split_name": self.split_name,
                "example_digests": [example.digest for example in self.examples],
                "source_path_sha256": self.source_path_sha256,
            }
        )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_document(value: Any) -> ScientificDocumentRef:
    if not isinstance(value, Mapping):
        raise ValueError("document record must be an object")
    allowed = {"document_id", "generation_id", "content_sha256", "source_group_id"}
    if set(value) - allowed:
        raise ValueError("document record contains unknown fields")
    return ScientificDocumentRef(**value)


def _parse_evidence(value: Any) -> ExpectedEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("evidence record must be an object")
    allowed = {"evidence_id", "document_id", "generation_id", "locator", "relation", "claim_id"}
    if set(value) - allowed:
        raise ValueError("evidence record contains unknown fields")
    return ExpectedEvidence(**value)


def _parse_example(value: Any) -> ScientificMultiDocumentExample:
    if not isinstance(value, Mapping):
        raise ValueError("example must be an object")
    allowed = {"example_id", "query", "task", "documents", "expected_answer", "expected_evidence", "strata"}
    if set(value) - allowed:
        raise ValueError("example contains unknown fields")
    documents = tuple(_parse_document(item) for item in value.get("documents", ()))
    evidence = tuple(_parse_evidence(item) for item in value.get("expected_evidence", ()))
    return ScientificMultiDocumentExample(
        example_id=value["example_id"],
        query=value["query"],
        task=value["task"],
        documents=documents,
        expected_answer=value.get("expected_answer"),
        expected_evidence=evidence,
        strata=value.get("strata", {}),
    )


def load_local_jsonl(
    path: str | Path,
    *,
    dataset_manifest: DatasetManifest,
    split_name: str,
    expected_file_sha256: str,
) -> ScientificBenchmarkCorpus:
    """Strictly load an already-local JSONL benchmark after exact digest verification."""

    selected_path = Path(path).expanduser().resolve(strict=True)
    if not selected_path.is_file() or selected_path.is_symlink():
        raise ValueError("benchmark path must be a regular non-symlink file")
    stat = selected_path.stat()
    if stat.st_size > _MAX_FILE_BYTES:
        raise ValueError("benchmark file exceeds safety limit")
    raw = selected_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _sha256(expected_file_sha256, "expected_file_sha256"):
        raise ValueError("benchmark file digest does not match expected digest")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("benchmark must be valid UTF-8") from exc
    examples: list[ScientificMultiDocumentExample] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant {value}")))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL record at line {line_number}") from exc
        examples.append(_parse_example(value))
        if len(examples) > _MAX_RECORDS:
            raise ValueError("benchmark record count exceeds safety limit")
    dataset_manifest.assert_promotable()
    return ScientificBenchmarkCorpus(dataset_manifest, split_name, tuple(examples), digest)


__all__ = [
    "ExpectedEvidence",
    "ExpectedRelation",
    "ScientificBenchmarkCorpus",
    "ScientificDocumentRef",
    "ScientificMultiDocumentExample",
    "ScientificTaskKind",
    "canonical_digest",
    "load_local_jsonl",
]
