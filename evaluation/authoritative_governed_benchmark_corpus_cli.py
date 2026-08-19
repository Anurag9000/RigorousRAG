"""Strict CLI for authoritative v2 benchmark corpus publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.authoritative_governed_benchmark_corpus import (
    publish_authoritative_benchmark_corpus,
    verify_authoritative_benchmark_corpus_receipt,
)
from evaluation.governed_benchmark_corpus import BenchmarkCorpusImportSpec, BenchmarkCorpusProfile
from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="authoritative corpus config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("authoritative corpus config exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("authoritative corpus config is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("authoritative corpus config must contain an object")
    return raw


def _paths(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def run_authoritative_corpus_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    required = {"schema", "source_path", "source_sha256", "input_format", "expected_record_count", "profile", "output_dir"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-benchmark-corpus-import-config/v2":
        raise ValueError("config must be rigorousrag-authoritative-benchmark-corpus-import-config/v2")
    profile_raw = raw["profile"]
    if not isinstance(profile_raw, Mapping):
        raise ValueError("profile must be an object")
    allowed_profile = {"name", "document_id_paths", "text_paths", "title_paths", "source_group_paths", "metadata_paths"}
    unknown = set(profile_raw) - allowed_profile
    if unknown:
        raise ValueError(f"profile has unsupported fields: {sorted(unknown)}")
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
    receipt = publish_authoritative_benchmark_corpus(spec, output_dir=raw["output_dir"])
    receipt_path = Path(raw["output_dir"]) / "corpus_receipt.json"
    verified = verify_authoritative_benchmark_corpus_receipt(receipt_path)
    if verified.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("authoritative corpus read-side verification returned a different identity")
    return {
        "output_path": receipt.output_path,
        "output_sha256": receipt.output_sha256,
        "record_count": receipt.record_count,
        "document_id_sha256": receipt.document_id_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt.receipt_sha256,
        "publication_contract_sha256": receipt.publication_contract_sha256,
        "publication_authority": "authoritative-benchmark-corpus/v2",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish exact local retrieval corpus bytes as a closed authoritative v2 corpus")
    parser.add_argument("config", help="rigorousrag-authoritative-benchmark-corpus-import-config/v2 JSON file")
    print(json.dumps(run_authoritative_corpus_config(parser.parse_args(argv).config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_authoritative_corpus_config"]
