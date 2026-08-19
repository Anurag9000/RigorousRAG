"""Strict config-only command for governed retrieval benchmark corpus import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.governed_benchmark_corpus import BenchmarkCorpusImportSpec, BenchmarkCorpusProfile, import_governed_benchmark_corpus
from evaluation.governed_benchmark_corpus_io import verify_governed_benchmark_corpus_receipt
from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="benchmark corpus import config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("benchmark corpus import config exceeds byte safety bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("benchmark corpus import config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("benchmark corpus import config must be an object")
    return value


def _paths(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def run_corpus_import_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    required = {"schema", "source_path", "source_sha256", "input_format", "expected_record_count", "profile", "output_path"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-governed-benchmark-corpus-import-config/v1":
        raise ValueError("config must be rigorousrag-governed-benchmark-corpus-import-config/v1")
    profile_raw = raw["profile"]
    if not isinstance(profile_raw, Mapping):
        raise ValueError("profile must be an object")
    allowed_profile = {"name", "document_id_paths", "text_paths", "title_paths", "source_group_paths", "metadata_paths"}
    if set(profile_raw) - allowed_profile:
        raise ValueError(f"profile has unsupported fields: {sorted(set(profile_raw)-allowed_profile)}")
    metadata_raw = profile_raw.get("metadata_paths", {})
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("profile.metadata_paths must be an object")
    profile = BenchmarkCorpusProfile(
        name=profile_raw.get("name", "corpus"),
        document_id_paths=_paths(profile_raw.get("document_id_paths", []), "document_id_paths"),
        text_paths=_paths(profile_raw.get("text_paths", []), "text_paths"),
        title_paths=_paths(profile_raw.get("title_paths", []), "title_paths"),
        source_group_paths=_paths(profile_raw.get("source_group_paths", []), "source_group_paths"),
        metadata_paths={str(key): _paths(value, f"metadata_paths.{key}") for key, value in metadata_raw.items()},
    )
    expected = raw["expected_record_count"]
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int) or expected < 0):
        raise ValueError("expected_record_count must be non-negative or null")
    spec = BenchmarkCorpusImportSpec(
        source_path=raw["source_path"], source_sha256=raw["source_sha256"], profile=profile,
        input_format=raw["input_format"], expected_record_count=expected,
    )
    receipt = import_governed_benchmark_corpus(spec, output_path=raw["output_path"])
    receipt_path = str(Path(receipt.output_path).with_suffix(Path(receipt.output_path).suffix + ".receipt.json"))
    verified = verify_governed_benchmark_corpus_receipt(receipt_path)
    if verified.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("corpus import read-side verification returned a different receipt identity")
    return {"output_path": receipt.output_path, "output_sha256": receipt.output_sha256, "record_count": receipt.record_count, "document_id_sha256": receipt.document_id_sha256, "receipt_path": receipt_path, "receipt_sha256": receipt.receipt_sha256}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import exact local retrieval corpus bytes into governed canonical JSONL")
    parser.add_argument("config", help="rigorousrag-governed-benchmark-corpus-import-config/v1 JSON file")
    print(json.dumps(run_corpus_import_config(parser.parse_args(argv).config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_corpus_import_config"]
