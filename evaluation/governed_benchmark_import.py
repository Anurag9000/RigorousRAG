"""Governed local-only benchmark import for RigorousRAG.

This module closes the gap between planning-only benchmark names / permissive in-memory
adapters and promotable, immutable evaluation data.  It never downloads a dataset, infers a
license, runs a model, or invents semantic labels.  Operators provide already-local source
files together with exact SHA-256 values and reviewed governance metadata.

Two adaptation modes are supported:

* an existing :mod:`tools.benchmark_adapters` adapter name; or
* a declarative field-path profile.  Field paths use ``.`` for mapping traversal and ``[]``
  to flatten list-valued segments, for example ``positive_passages[].docid``.

Every imported split is written atomically as canonical ``BenchmarkExample`` JSONL.  The
resulting ``DatasetManifest`` binds exact source bytes, deterministic transformation identity,
canonical split bytes, record/query/document/source-group identifiers and reviewed licensing.
A self-verifying receipt binds the complete import result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from evaluation.dataset_governance import (
    DatasetCard,
    DatasetManifest,
    DatasetModality,
    DatasetTask,
    LicenseStatus,
    SplitManifest,
    canonical_digest,
)
from tools.benchmark_adapters import ADAPTERS, BenchmarkExample, adapt_record

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_JSON_BYTES = 512 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_TEXT = 8_000_000
_MAX_COLLECTION = 100_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, *, allow_empty: bool = False, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        value = str(value)
    if "\x00" in value or len(value) > maximum:
        raise ValueError(f"{label} is invalid or exceeds the character bound")
    selected = value.strip()
    if not allow_empty and not selected:
        raise ValueError(f"{label} may not be empty")
    return selected


def _safe_path(value: str | os.PathLike[str], *, label: str, must_exist: bool, directory: bool | None = None) -> Path:
    raw = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    for component in (raw, *raw.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT):
            raise ValueError(f"{label} may not traverse a symlink or reparse point")
    if must_exist and not raw.exists():
        raise ValueError(f"{label} does not exist")
    if raw.exists():
        if directory is True and not raw.is_dir():
            raise ValueError(f"{label} must be a directory")
        if directory is False and not raw.is_file():
            raise ValueError(f"{label} must be a regular file")
    return raw


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _id_digest(values: Iterable[str]) -> str:
    normalized = sorted({_identifier(value, "identifier", 10_000) for value in values})
    return hashlib.sha256(("\n".join(normalized) + ("\n" if normalized else "")).encode("utf-8")).hexdigest()


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence):
        result: list[Any] = []
        for item in value:
            result.extend(_flatten(item))
            if len(result) > _MAX_COLLECTION:
                raise ValueError("declarative field collection exceeds safety bound")
        return result
    return [value]


def _walk(root: Any, expression: str) -> list[Any]:
    selected = _identifier(expression, "field expression", 1_000)
    nodes = [root]
    for raw_segment in selected.split("."):
        expand = raw_segment.endswith("[]")
        key = raw_segment[:-2] if expand else raw_segment
        key = _identifier(key, "field path segment", 300)
        next_nodes: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping) or key not in node:
                continue
            value = node[key]
            if expand:
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    next_nodes.extend(value)
                elif value is not None:
                    raise ValueError(f"field expression {expression!r} expected an array at {key!r}")
            else:
                next_nodes.append(value)
            if len(next_nodes) > _MAX_COLLECTION:
                raise ValueError("field expression expands beyond safety bound")
        nodes = next_nodes
        if not nodes:
            break
    return nodes


def _first(root: Mapping[str, Any], expressions: Sequence[str]) -> Any | None:
    for expression in expressions:
        values = _walk(root, expression)
        for value in values:
            if value is not None and (not isinstance(value, str) or value.strip()):
                return value
    return None


def _collect(root: Mapping[str, Any], expressions: Sequence[str]) -> tuple[Any, ...]:
    result: list[Any] = []
    for expression in expressions:
        for value in _walk(root, expression):
            result.extend(_flatten(value))
            if len(result) > _MAX_COLLECTION:
                raise ValueError("declarative collection exceeds safety bound")
    return tuple(result)


def _answer_values(values: Sequence[Any]) -> tuple[str, ...]:
    answers: list[str] = []
    stack = list(values)
    while stack:
        value = stack.pop(0)
        if value is None:
            continue
        if isinstance(value, Mapping):
            selected = None
            for key in ("text", "answer", "answers", "value"):
                if key in value:
                    selected = value[key]
                    break
            if selected is not None:
                stack[0:0] = _flatten(selected)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            stack[0:0] = list(value)
            continue
        text = _text(value, "answer", allow_empty=True)
        if text and text not in answers:
            answers.append(text)
        if len(answers) > _MAX_COLLECTION:
            raise ValueError("answer collection exceeds safety bound")
    return tuple(answers)


@dataclass(frozen=True)
class DeclarativeBenchmarkProfile:
    name: str
    id_paths: tuple[str, ...]
    query_paths: tuple[str, ...]
    answer_paths: tuple[str, ...] = ()
    relevant_id_paths: tuple[str, ...] = ()
    context_paths: tuple[str, ...] = ()
    metadata_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constant_metadata: Mapping[str, str] = field(default_factory=dict)
    generate_missing_ids: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "profile name", 300))
        for attr in ("id_paths", "query_paths", "answer_paths", "relevant_id_paths", "context_paths"):
            values = tuple(_identifier(value, attr, 1_000) for value in getattr(self, attr))
            if len(values) > 100:
                raise ValueError(f"{attr} exceeds safety bound")
            object.__setattr__(self, attr, values)
        if not self.query_paths:
            raise ValueError("declarative profile requires at least one query path")
        if not self.id_paths and not self.generate_missing_ids:
            raise ValueError("declarative profile requires id_paths unless generate_missing_ids is enabled")
        metadata: dict[str, tuple[str, ...]] = {}
        if len(self.metadata_paths) > 100:
            raise ValueError("metadata_paths exceeds safety bound")
        for key, paths in self.metadata_paths.items():
            metadata[_identifier(key, "metadata key", 300)] = tuple(_identifier(path, "metadata path", 1_000) for path in paths)
        object.__setattr__(self, "metadata_paths", metadata)
        constants = {_identifier(key, "constant metadata key", 300): _text(value, "constant metadata value", allow_empty=True, maximum=20_000) for key, value in self.constant_metadata.items()}
        object.__setattr__(self, "constant_metadata", constants)

    @property
    def profile_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-declarative-benchmark-profile/v1", **asdict(self)})

    def adapt(self, row: Mapping[str, Any], *, ordinal: int, split_name: str, dataset_id: str) -> BenchmarkExample:
        query_raw = _first(row, self.query_paths)
        if query_raw is None:
            raise ValueError("declarative benchmark row has no query value")
        query = _text(query_raw, "benchmark query")
        identifier_raw = _first(row, self.id_paths)
        generated = False
        if identifier_raw is None:
            if not self.generate_missing_ids:
                raise ValueError("declarative benchmark row has no example id")
            identifier = "generated-" + hashlib.sha256(f"{dataset_id}\n{split_name}\n{ordinal}\n{query}".encode("utf-8")).hexdigest()[:32]
            generated = True
        else:
            identifier = _identifier(str(identifier_raw), "example id", 10_000)
        answers = _answer_values(_collect(row, self.answer_paths))
        relevant = tuple(dict.fromkeys(_identifier(str(value), "relevant id", 10_000) for value in _collect(row, self.relevant_id_paths) if str(value).strip()))
        contexts = tuple(dict.fromkeys(_text(value, "context", allow_empty=True) for value in _collect(row, self.context_paths) if _text(value, "context", allow_empty=True)))
        metadata: dict[str, Any] = dict(self.constant_metadata)
        for key, paths in self.metadata_paths.items():
            value = _first(row, paths)
            if value is not None:
                if isinstance(value, (Mapping, list, tuple)):
                    metadata[key] = value
                else:
                    metadata[key] = _text(value, f"metadata {key}", allow_empty=True, maximum=100_000)
        metadata["import_profile"] = self.name
        metadata["import_profile_sha256"] = self.profile_sha256
        if generated:
            metadata["generated_example_id"] = True
        return BenchmarkExample(identifier, query, answers, relevant, contexts, metadata)


@dataclass(frozen=True)
class BenchmarkSplitImportSpec:
    name: str
    source_path: str
    source_sha256: str
    input_format: str = "jsonl"
    adapter_name: str | None = None
    profile: DeclarativeBenchmarkProfile | None = None
    expected_record_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "split name", 200))
        source = _safe_path(self.source_path, label=f"split {self.name} source", must_exist=True, directory=False)
        object.__setattr__(self, "source_path", str(source))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))
        selected_format = _identifier(self.input_format, "input_format", 20).lower()
        if selected_format not in {"jsonl", "json"}:
            raise ValueError("input_format must be jsonl or json")
        object.__setattr__(self, "input_format", selected_format)
        if (self.adapter_name is None) == (self.profile is None):
            raise ValueError("exactly one of adapter_name or profile must be configured")
        if self.adapter_name is not None:
            adapter = _identifier(self.adapter_name, "adapter_name", 200).lower()
            if adapter not in ADAPTERS:
                raise ValueError(f"unknown benchmark adapter: {adapter}")
            object.__setattr__(self, "adapter_name", adapter)
        if self.profile is not None and not isinstance(self.profile, DeclarativeBenchmarkProfile):
            raise ValueError("profile must be DeclarativeBenchmarkProfile")
        if self.expected_record_count is not None and (isinstance(self.expected_record_count, bool) or not isinstance(self.expected_record_count, int) or self.expected_record_count < 0):
            raise ValueError("expected_record_count must be non-negative")

    @property
    def transformation_component_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-benchmark-split-import/v1",
            "name": self.name,
            "input_format": self.input_format,
            "adapter_name": self.adapter_name,
            "profile_sha256": None if self.profile is None else self.profile.profile_sha256,
        })


@dataclass(frozen=True)
class BenchmarkGovernanceSpec:
    dataset_id: str
    exact_version: str
    source_locator: str
    license_identifier: str
    license_status: LicenseStatus
    license_evidence: str
    tasks: tuple[DatasetTask, ...]
    modalities: tuple[DatasetModality, ...]
    card: DatasetCard
    metadata: Mapping[str, str] = field(default_factory=dict)
    require_promotable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _identifier(self.dataset_id, "dataset_id", 500))
        object.__setattr__(self, "exact_version", _identifier(self.exact_version, "exact_version", 1_000))
        object.__setattr__(self, "source_locator", _text(self.source_locator, "source_locator", maximum=5_000))
        object.__setattr__(self, "license_identifier", _identifier(self.license_identifier, "license_identifier", 1_000))
        if not isinstance(self.license_status, LicenseStatus):
            object.__setattr__(self, "license_status", LicenseStatus(self.license_status))
        object.__setattr__(self, "license_evidence", _text(self.license_evidence, "license_evidence", maximum=100_000))
        if not self.tasks or any(not isinstance(item, DatasetTask) for item in self.tasks):
            raise ValueError("tasks must contain DatasetTask values")
        if not self.modalities or any(not isinstance(item, DatasetModality) for item in self.modalities):
            raise ValueError("modalities must contain DatasetModality values")
        if not isinstance(self.card, DatasetCard):
            raise ValueError("card must be DatasetCard")
        object.__setattr__(self, "metadata", {_identifier(key, "metadata key", 300): _text(value, "metadata value", allow_empty=True, maximum=20_000) for key, value in self.metadata.items()})


@dataclass(frozen=True)
class ImportedSplitReceipt:
    name: str
    source_sha256: str
    output_path: str
    output_sha256: str
    record_count: int
    record_id_sha256: str
    query_id_sha256: str
    document_id_sha256: str | None
    source_group_sha256: str | None
    transformation_component_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "split name", 200))
        for attr in ("source_sha256", "output_sha256", "record_id_sha256", "query_id_sha256", "transformation_component_sha256"):
            object.__setattr__(self, attr, _sha256(getattr(self, attr), attr))
        for attr in ("document_id_sha256", "source_group_sha256"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, _sha256(value, attr))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count < 0:
            raise ValueError("record_count must be non-negative")


@dataclass(frozen=True)
class GovernedBenchmarkImportReceipt:
    dataset_manifest_sha256: str
    dataset_artifact_sha256: str
    transformation_sha256: str
    manifest_path: str
    split_receipts: tuple[ImportedSplitReceipt, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for attr in ("dataset_manifest_sha256", "dataset_artifact_sha256", "transformation_sha256", "receipt_sha256"):
            object.__setattr__(self, attr, _sha256(getattr(self, attr), attr))
        if not self.split_receipts or any(not isinstance(item, ImportedSplitReceipt) for item in self.split_receipts):
            raise ValueError("split_receipts must be a non-empty tuple")
        if _digest(self._unsigned()) != self.receipt_sha256:
            raise ValueError("governed benchmark import receipt digest mismatch")

    def _unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-governed-benchmark-import-receipt/v1",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_artifact_sha256": self.dataset_artifact_sha256,
            "transformation_sha256": self.transformation_sha256,
            "manifest_path": self.manifest_path,
            "split_receipts": [asdict(item) for item in self.split_receipts],
        }


def _iter_rows(spec: BenchmarkSplitImportSpec) -> Iterator[Mapping[str, Any]]:
    source = Path(spec.source_path)
    actual = _stream_sha256(source)
    if actual != spec.source_sha256:
        raise ValueError(f"split {spec.name} source SHA-256 differs from configured immutable bytes")
    if spec.input_format == "jsonl":
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                if len(raw) > _MAX_LINE_BYTES:
                    raise ValueError(f"split {spec.name} line {line_number} exceeds byte safety bound")
                value = _strict_json_bytes(raw, f"split {spec.name} line {line_number}")
                if not isinstance(value, Mapping):
                    raise ValueError(f"split {spec.name} line {line_number} must be a JSON object")
                yield value
        return
    size = source.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"split {spec.name} JSON source exceeds whole-file byte safety bound")
    value = _strict_json_bytes(source.read_bytes(), f"split {spec.name}")
    if not isinstance(value, list):
        raise ValueError(f"split {spec.name} JSON input must contain an array of objects")
    if len(value) > _MAX_RECORDS:
        raise ValueError(f"split {spec.name} exceeds record safety bound")
    for index, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"split {spec.name} JSON row {index} must be an object")
        yield row


def _adapt(spec: BenchmarkSplitImportSpec, row: Mapping[str, Any], *, ordinal: int, dataset_id: str) -> BenchmarkExample:
    if spec.adapter_name is not None:
        result = adapt_record(spec.adapter_name, row)
    else:
        assert spec.profile is not None
        result = spec.profile.adapt(row, ordinal=ordinal, split_name=spec.name, dataset_id=dataset_id)
    if not isinstance(result, BenchmarkExample):
        raise ValueError("benchmark adapter did not return BenchmarkExample")
    example_id = _identifier(result.example_id, "example id", 10_000)
    query = _text(result.query, "query")
    answers = tuple(_text(value, "answer", allow_empty=False) for value in result.answers)
    relevant = tuple(dict.fromkeys(_identifier(value, "relevant id", 10_000) for value in result.relevant_ids))
    contexts = tuple(_text(value, "context", allow_empty=False) for value in result.contexts)
    metadata = dict(result.metadata)
    metadata["source_split"] = spec.name
    metadata["source_sha256"] = spec.source_sha256
    return BenchmarkExample(example_id, query, answers, relevant, contexts, metadata)


def _benchmark_payload(example: BenchmarkExample) -> Mapping[str, Any]:
    return {
        "schema": "rigorousrag-benchmark-example/v1",
        "example_id": example.example_id,
        "query": example.query,
        "answers": list(example.answers),
        "relevant_ids": list(example.relevant_ids),
        "contexts": list(example.contexts),
        "metadata": dict(example.metadata),
    }


def _import_split(spec: BenchmarkSplitImportSpec, *, dataset_id: str, output_dir: Path) -> ImportedSplitReceipt:
    destination = _safe_path(output_dir / f"{spec.name}.benchmark.jsonl", label=f"split {spec.name} output", must_exist=False, directory=None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    output_digest = hashlib.sha256()
    record_ids: list[str] = []
    query_ids: list[str] = []
    document_ids: list[str] = []
    source_groups: list[str] = []
    seen: set[str] = set()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for ordinal, row in enumerate(_iter_rows(spec), start=1):
                if count >= _MAX_RECORDS:
                    raise ValueError(f"split {spec.name} exceeds record safety bound")
                example = _adapt(spec, row, ordinal=ordinal, dataset_id=dataset_id)
                if example.example_id in seen:
                    raise ValueError(f"split {spec.name} contains duplicate example id {example.example_id!r}")
                seen.add(example.example_id)
                payload = _canonical_bytes(_benchmark_payload(example)) + b"\n"
                if len(payload) > _MAX_LINE_BYTES:
                    raise ValueError(f"canonical split {spec.name} record exceeds byte safety bound")
                handle.write(payload)
                output_digest.update(payload)
                record_ids.append(example.example_id)
                query_ids.append(example.example_id)
                document_ids.extend(example.relevant_ids)
                source_group = example.metadata.get("source_group_id") if isinstance(example.metadata, Mapping) else None
                if isinstance(source_group, str) and source_group.strip():
                    source_groups.append(source_group.strip())
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if spec.expected_record_count is not None and count != spec.expected_record_count:
            raise ValueError(f"split {spec.name} record count differs from configured expected_record_count")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    output_sha = output_digest.hexdigest()
    if _stream_sha256(destination) != output_sha:
        raise RuntimeError("canonical benchmark output digest changed after atomic publication")
    return ImportedSplitReceipt(
        name=spec.name,
        source_sha256=spec.source_sha256,
        output_path=str(destination),
        output_sha256=output_sha,
        record_count=count,
        record_id_sha256=_id_digest(record_ids),
        query_id_sha256=_id_digest(query_ids),
        document_id_sha256=_id_digest(document_ids) if document_ids else None,
        source_group_sha256=_id_digest(source_groups) if source_groups else None,
        transformation_component_sha256=spec.transformation_component_sha256,
    )


def import_governed_benchmark(
    governance: BenchmarkGovernanceSpec,
    splits: Sequence[BenchmarkSplitImportSpec],
    *,
    output_dir: str | Path,
) -> tuple[DatasetManifest, GovernedBenchmarkImportReceipt]:
    """Import exact local source bytes into canonical benchmark JSONL plus a real manifest."""
    if not isinstance(governance, BenchmarkGovernanceSpec):
        raise ValueError("governance must be BenchmarkGovernanceSpec")
    if not splits or len(splits) > 100 or any(not isinstance(item, BenchmarkSplitImportSpec) for item in splits):
        raise ValueError("splits must be a non-empty bounded BenchmarkSplitImportSpec sequence")
    names = [item.name for item in splits]
    if len(names) != len(set(names)):
        raise ValueError("split names must be unique")
    root = _safe_path(output_dir, label="benchmark import output directory", must_exist=False, directory=None)
    if root.exists() and not root.is_dir():
        raise ValueError("benchmark import output path must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    receipts = tuple(_import_split(spec, dataset_id=governance.dataset_id, output_dir=root) for spec in splits)
    source_set_sha = _digest({
        "schema": "rigorousrag-benchmark-source-set/v1",
        "dataset_id": governance.dataset_id,
        "exact_version": governance.exact_version,
        "splits": [{"name": item.name, "source_sha256": item.source_sha256} for item in receipts],
    })
    transformation_sha = _digest({
        "schema": "rigorousrag-governed-benchmark-import-transformation/v1",
        "loader_name": "evaluation.governed_benchmark_import",
        "loader_version": "1",
        "split_transformations": [{"name": item.name, "sha256": item.transformation_component_sha256} for item in receipts],
    })
    split_manifests = tuple(
        SplitManifest(
            name=item.name,
            content_sha256=item.output_sha256,
            record_count=item.record_count,
            record_id_sha256=item.record_id_sha256,
            source_group_sha256=item.source_group_sha256,
            query_id_sha256=item.query_id_sha256,
            document_id_sha256=item.document_id_sha256,
        )
        for item in receipts
    )
    manifest = DatasetManifest(
        dataset_id=governance.dataset_id,
        exact_version=governance.exact_version,
        source_locator=governance.source_locator,
        artifact_sha256=source_set_sha,
        license_identifier=governance.license_identifier,
        license_status=governance.license_status,
        license_evidence=governance.license_evidence,
        loader_name="evaluation.governed_benchmark_import",
        loader_version="1",
        transformation_sha256=transformation_sha,
        splits=split_manifests,
        tasks=governance.tasks,
        modalities=governance.modalities,
        card=governance.card,
        metadata={**governance.metadata, "canonical_record_schema": "rigorousrag-benchmark-example/v1"},
    )
    if governance.require_promotable:
        manifest.assert_promotable()
    manifest_path = _safe_path(root / "dataset_manifest.json", label="dataset manifest output", must_exist=False, directory=None)
    manifest_payload = _canonical_bytes({"schema": "rigorousrag-dataset-manifest/v1", "manifest": asdict(manifest), "manifest_sha256": manifest.manifest_digest}) + b"\n"
    _atomic_write(manifest_path, manifest_payload)
    unsigned = {
        "schema": "rigorousrag-governed-benchmark-import-receipt/v1",
        "dataset_manifest_sha256": manifest.manifest_digest,
        "dataset_artifact_sha256": source_set_sha,
        "transformation_sha256": transformation_sha,
        "manifest_path": str(manifest_path),
        "split_receipts": [asdict(item) for item in receipts],
    }
    receipt = GovernedBenchmarkImportReceipt(
        dataset_manifest_sha256=manifest.manifest_digest,
        dataset_artifact_sha256=source_set_sha,
        transformation_sha256=transformation_sha,
        manifest_path=str(manifest_path),
        split_receipts=receipts,
        receipt_sha256=_digest(unsigned),
    )
    _atomic_write(root / "import_receipt.json", _canonical_bytes({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n")
    return manifest, receipt


def _profile_from_json(raw: Any, label: str) -> DeclarativeBenchmarkProfile:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    allowed = {"name", "id_paths", "query_paths", "answer_paths", "relevant_id_paths", "context_paths", "metadata_paths", "constant_metadata", "generate_missing_ids"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    metadata_paths_raw = raw.get("metadata_paths") or {}
    if not isinstance(metadata_paths_raw, Mapping):
        raise ValueError(f"{label}.metadata_paths must be an object")
    def paths(name: str) -> tuple[str, ...]:
        value = raw.get(name) or []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label}.{name} must be an array of strings")
        return tuple(value)
    normalized_metadata: dict[str, tuple[str, ...]] = {}
    for key, value in metadata_paths_raw.items():
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label}.metadata_paths entries must be string arrays")
        normalized_metadata[str(key)] = tuple(value)
    constants = raw.get("constant_metadata") or {}
    if not isinstance(constants, Mapping):
        raise ValueError(f"{label}.constant_metadata must be an object")
    return DeclarativeBenchmarkProfile(
        name=raw.get("name", label),
        id_paths=paths("id_paths"),
        query_paths=paths("query_paths"),
        answer_paths=paths("answer_paths"),
        relevant_id_paths=paths("relevant_id_paths"),
        context_paths=paths("context_paths"),
        metadata_paths=normalized_metadata,
        constant_metadata={str(key): str(value) for key, value in constants.items()},
        generate_missing_ids=bool(raw.get("generate_missing_ids", False)),
    )


def _card_from_json(raw: Any) -> DatasetCard:
    if not isinstance(raw, Mapping):
        raise ValueError("governance.card must be an object")
    allowed = {"summary", "intended_uses", "forbidden_uses", "populations_or_domains", "languages", "pii_notes", "safety_notes", "source_citation", "known_limitations"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"governance.card contains unsupported fields: {sorted(unknown)}")
    def strings(name: str) -> tuple[str, ...]:
        value = raw.get(name) or []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"governance.card.{name} must be an array of strings")
        return tuple(value)
    return DatasetCard(
        summary=raw.get("summary", ""),
        intended_uses=strings("intended_uses"),
        forbidden_uses=strings("forbidden_uses"),
        populations_or_domains=strings("populations_or_domains"),
        languages=strings("languages"),
        pii_notes=raw.get("pii_notes"),
        safety_notes=raw.get("safety_notes"),
        source_citation=raw.get("source_citation"),
        known_limitations=strings("known_limitations"),
    )


def import_governed_benchmark_from_config(path: str | Path) -> tuple[DatasetManifest, GovernedBenchmarkImportReceipt]:
    """Load a strict local JSON import config and execute no-network conversion."""
    source = _safe_path(path, label="benchmark import config", must_exist=True, directory=False)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("benchmark import config exceeds byte safety bound")
    raw = _strict_json_bytes(source.read_bytes(), "benchmark import config")
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark import config must be an object")
    if set(raw) != {"schema", "output_dir", "governance", "splits"} or raw.get("schema") != "rigorousrag-governed-benchmark-import-config/v1":
        raise ValueError("benchmark import config must be rigorousrag-governed-benchmark-import-config/v1")
    governance_raw = raw.get("governance")
    if not isinstance(governance_raw, Mapping):
        raise ValueError("governance must be an object")
    allowed_governance = {"dataset_id", "exact_version", "source_locator", "license_identifier", "license_status", "license_evidence", "tasks", "modalities", "card", "metadata", "require_promotable"}
    unknown = set(governance_raw) - allowed_governance
    if unknown:
        raise ValueError(f"governance contains unsupported fields: {sorted(unknown)}")
    tasks = governance_raw.get("tasks")
    modalities = governance_raw.get("modalities")
    if not isinstance(tasks, list) or not tasks or not isinstance(modalities, list) or not modalities:
        raise ValueError("governance.tasks and governance.modalities must be non-empty arrays")
    metadata = governance_raw.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("governance.metadata must be an object")
    governance = BenchmarkGovernanceSpec(
        dataset_id=governance_raw.get("dataset_id"),
        exact_version=governance_raw.get("exact_version"),
        source_locator=governance_raw.get("source_locator"),
        license_identifier=governance_raw.get("license_identifier"),
        license_status=LicenseStatus(governance_raw.get("license_status")),
        license_evidence=governance_raw.get("license_evidence"),
        tasks=tuple(DatasetTask(item) for item in tasks),
        modalities=tuple(DatasetModality(item) for item in modalities),
        card=_card_from_json(governance_raw.get("card")),
        metadata={str(key): str(value) for key, value in metadata.items()},
        require_promotable=bool(governance_raw.get("require_promotable", False)),
    )
    split_raw = raw.get("splits")
    if not isinstance(split_raw, list) or not split_raw:
        raise ValueError("splits must be a non-empty array")
    splits: list[BenchmarkSplitImportSpec] = []
    for index, item in enumerate(split_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"split {index} must be an object")
        allowed_split = {"name", "source_path", "source_sha256", "input_format", "adapter_name", "profile", "expected_record_count"}
        unknown_split = set(item) - allowed_split
        if unknown_split:
            raise ValueError(f"split {index} contains unsupported fields: {sorted(unknown_split)}")
        profile = _profile_from_json(item["profile"], f"split[{index}].profile") if item.get("profile") is not None else None
        splits.append(BenchmarkSplitImportSpec(
            name=item.get("name"),
            source_path=item.get("source_path"),
            source_sha256=item.get("source_sha256"),
            input_format=item.get("input_format", "jsonl"),
            adapter_name=item.get("adapter_name"),
            profile=profile,
            expected_record_count=item.get("expected_record_count"),
        ))
    return import_governed_benchmark(governance, tuple(splits), output_dir=raw.get("output_dir"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import exact local benchmark bytes into governed RigorousRAG data")
    parser.add_argument("config", help="rigorousrag-governed-benchmark-import-config/v1 JSON file")
    args = parser.parse_args(argv)
    manifest, receipt = import_governed_benchmark_from_config(args.config)
    print(json.dumps({
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_sha256": manifest.manifest_digest,
        "artifact_sha256": manifest.artifact_sha256,
        "record_count": sum(split.record_count for split in manifest.splits),
        "receipt_sha256": receipt.receipt_sha256,
        "manifest_path": receipt.manifest_path,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BenchmarkGovernanceSpec",
    "BenchmarkSplitImportSpec",
    "DeclarativeBenchmarkProfile",
    "GovernedBenchmarkImportReceipt",
    "ImportedSplitReceipt",
    "import_governed_benchmark",
    "import_governed_benchmark_from_config",
]
