"""Local-only, content-addressed corpus import for retrieval benchmarks.

A query/qrels split is not a complete retrieval benchmark identity when its document corpus is
stored separately. This module converts already-local JSON/JSONL corpus bytes into canonical
full-text documents with stable document ids. It performs no download, indexing, retrieval or
model execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path

_MAX_JSON_BYTES = 2 * 1024 * 1024 * 1024
_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_RECORDS = 200_000_000
_MAX_TEXT = 32 * 1024 * 1024
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


def _identifier(value: Any, label: str, maximum: int = 10_000) -> str:
    selected = str(value).strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, *, allow_empty: bool = False, maximum: int = _MAX_TEXT) -> str:
    selected = str(value)
    if "\x00" in selected or len(selected) > maximum:
        raise ValueError(f"{label} exceeds safety bound or contains NUL")
    selected = selected.strip()
    if not allow_empty and not selected:
        raise ValueError(f"{label} may not be empty")
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


def _strict_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _first(root: Mapping[str, Any], paths: Sequence[str]) -> Any | None:
    for expression in paths:
        node: Any = root
        found = True
        for segment in _identifier(expression, "field path", 1_000).split("."):
            if not isinstance(node, Mapping) or segment not in node:
                found = False
                break
            node = node[segment]
        if found and node is not None and (not isinstance(node, str) or node.strip()):
            return node
    return None


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
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
class BenchmarkCorpusDocument:
    document_id: str
    text: str
    title: str | None = None
    source_group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        object.__setattr__(self, "text", _text(self.text, "document text"))
        if self.title is not None:
            object.__setattr__(self, "title", _text(self.title, "title", allow_empty=True, maximum=1_000_000))
        if self.source_group_id is not None:
            object.__setattr__(self, "source_group_id", _identifier(self.source_group_id, "source_group_id"))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 10_000:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class BenchmarkCorpusProfile:
    name: str
    document_id_paths: tuple[str, ...]
    text_paths: tuple[str, ...]
    title_paths: tuple[str, ...] = ()
    source_group_paths: tuple[str, ...] = ()
    metadata_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "profile name", 300))
        for name in ("document_id_paths", "text_paths", "title_paths", "source_group_paths"):
            object.__setattr__(self, name, tuple(_identifier(item, name, 1_000) for item in getattr(self, name)))
        if not self.document_id_paths or not self.text_paths:
            raise ValueError("profile requires document_id_paths and text_paths")
        object.__setattr__(self, "metadata_paths", {_identifier(key, "metadata key", 300): tuple(_identifier(path, "metadata path", 1_000) for path in paths) for key, paths in self.metadata_paths.items()})

    @property
    def profile_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-benchmark-corpus-profile/v1", **asdict(self)})

    def adapt(self, row: Mapping[str, Any]) -> BenchmarkCorpusDocument:
        document_id, text = _first(row, self.document_id_paths), _first(row, self.text_paths)
        if document_id is None or text is None:
            raise ValueError("corpus row lacks configured document id or text")
        title = _first(row, self.title_paths) if self.title_paths else None
        source_group = _first(row, self.source_group_paths) if self.source_group_paths else None
        metadata = {key: value for key, paths in self.metadata_paths.items() if (value := _first(row, paths)) is not None}
        metadata.update({"corpus_profile": self.name, "corpus_profile_sha256": self.profile_sha256})
        return BenchmarkCorpusDocument(str(document_id), str(text), None if title is None else str(title), None if source_group is None else str(source_group), metadata)


@dataclass(frozen=True)
class BenchmarkCorpusImportSpec:
    source_path: str
    source_sha256: str
    profile: BenchmarkCorpusProfile
    input_format: str = "jsonl"
    expected_record_count: int | None = None

    def __post_init__(self) -> None:
        source = safe_advanced_path(self.source_path, label="benchmark corpus source", must_exist=True, require_file=True)
        object.__setattr__(self, "source_path", str(source))
        object.__setattr__(self, "source_sha256", _sha(self.source_sha256, "source_sha256"))
        if not isinstance(self.profile, BenchmarkCorpusProfile):
            raise ValueError("profile must be BenchmarkCorpusProfile")
        selected = _identifier(self.input_format, "input_format", 20).lower()
        if selected not in {"json", "jsonl"}:
            raise ValueError("input_format must be json or jsonl")
        object.__setattr__(self, "input_format", selected)
        if self.expected_record_count is not None and (isinstance(self.expected_record_count, bool) or not isinstance(self.expected_record_count, int) or self.expected_record_count < 0):
            raise ValueError("expected_record_count must be non-negative")

    @property
    def transformation_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-benchmark-corpus-transform/v1", "input_format": self.input_format, "profile_sha256": self.profile.profile_sha256})


@dataclass(frozen=True)
class GovernedBenchmarkCorpusReceipt:
    source_sha256: str
    transformation_sha256: str
    output_path: str
    output_sha256: str
    record_count: int
    document_id_sha256: str
    source_group_sha256: str | None
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("source_sha256", "transformation_sha256", "output_sha256", "document_id_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.source_group_sha256 is not None:
            object.__setattr__(self, "source_group_sha256", _sha(self.source_group_sha256, "source_group_sha256"))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("corpus receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-governed-benchmark-corpus-receipt/v1", "source_sha256": self.source_sha256, "transformation_sha256": self.transformation_sha256, "output_path": self.output_path, "output_sha256": self.output_sha256, "record_count": self.record_count, "document_id_sha256": self.document_id_sha256, "source_group_sha256": self.source_group_sha256}


def _rows(spec: BenchmarkCorpusImportSpec) -> Iterator[Mapping[str, Any]]:
    source = Path(spec.source_path)
    if _stream_sha(source) != spec.source_sha256:
        raise ValueError("corpus source digest differs from configured bytes")
    if spec.input_format == "jsonl":
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                if len(raw) > _MAX_LINE_BYTES:
                    raise ValueError(f"corpus line {line_number} exceeds safety bound")
                value = _strict_json(raw, f"corpus line {line_number}")
                if not isinstance(value, Mapping):
                    raise ValueError("corpus JSONL rows must be objects")
                yield value
        return
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("corpus JSON exceeds whole-file safety bound")
    value = _strict_json(source.read_bytes(), "corpus JSON")
    if not isinstance(value, list):
        raise ValueError("corpus JSON must contain an array")
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("corpus JSON rows must be objects")
        yield row


def import_governed_benchmark_corpus(spec: BenchmarkCorpusImportSpec, *, output_path: str | Path) -> GovernedBenchmarkCorpusReceipt:
    destination = safe_advanced_path(output_path, label="canonical benchmark corpus output", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("canonical corpus output must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    digest = hashlib.sha256(); seen: set[str] = set(); ids: list[str] = []; groups: list[str] = []; count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in _rows(spec):
                if count >= _MAX_RECORDS:
                    raise ValueError("corpus exceeds record safety bound")
                document = spec.profile.adapt(row)
                if document.document_id in seen:
                    raise ValueError(f"duplicate corpus document id {document.document_id!r}")
                seen.add(document.document_id); ids.append(document.document_id)
                if document.source_group_id is not None: groups.append(document.source_group_id)
                metadata = dict(document.metadata); metadata["source_sha256"] = spec.source_sha256
                encoded = _canonical({"schema": "rigorousrag-benchmark-corpus-document/v1", "document_id": document.document_id, "title": document.title, "text": document.text, "source_group_id": document.source_group_id, "metadata": metadata}) + b"\n"
                if len(encoded) > _MAX_LINE_BYTES:
                    raise ValueError("canonical corpus row exceeds line safety bound")
                handle.write(encoded); digest.update(encoded); count += 1
            handle.flush(); os.fsync(handle.fileno())
        if count <= 0:
            raise ValueError("corpus may not be empty")
        if spec.expected_record_count is not None and count != spec.expected_record_count:
            raise ValueError("corpus count differs from expected_record_count")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    output_sha = digest.hexdigest()
    if _stream_sha(destination) != output_sha:
        raise RuntimeError("canonical corpus changed during publication")
    unsigned = {"schema": "rigorousrag-governed-benchmark-corpus-receipt/v1", "source_sha256": spec.source_sha256, "transformation_sha256": spec.transformation_sha256, "output_path": str(destination), "output_sha256": output_sha, "record_count": count, "document_id_sha256": _id_digest(ids), "source_group_sha256": _id_digest(groups) if groups else None}
    receipt = GovernedBenchmarkCorpusReceipt(**{key: value for key, value in unsigned.items() if key != "schema"}, receipt_sha256=_digest(unsigned))
    _atomic(destination.with_suffix(destination.suffix + ".receipt.json"), _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n")
    return receipt


__all__ = ["BenchmarkCorpusDocument", "BenchmarkCorpusImportSpec", "BenchmarkCorpusProfile", "GovernedBenchmarkCorpusReceipt", "import_governed_benchmark_corpus"]
