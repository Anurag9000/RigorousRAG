"""Strict config-only entry point for disk-backed governed qrels parsing and persistence."""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence
from pathlib import Path

from evaluation.authoritative_governed_qrels import close_authoritative_governed_qrels, load_authoritative_governed_qrels
from evaluation.governed_qrels_io import load_governed_qrels_from_receipt, write_governed_qrels_receipt
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="governed qrels config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("governed qrels config exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ValueError("governed qrels config is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("governed qrels config must contain an object")
    return raw


def run_qrels_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    required = {"schema", "source_path", "source_sha256", "input_format", "minimum_relevance", "query_field", "document_field", "relevance_field", "receipt_path"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-governed-qrels-config/v1":
        raise ValueError("config must be rigorousrag-governed-qrels-config/v1")
    qrels = load_authoritative_governed_qrels(
        raw["source_path"], expected_sha256=raw["source_sha256"], input_format=raw["input_format"], minimum_relevance=raw["minimum_relevance"],
        query_field=raw["query_field"], document_field=raw["document_field"], relevance_field=raw["relevance_field"],
    )
    verified = None
    try:
        write_governed_qrels_receipt(raw["receipt_path"], qrels)
        verified = load_governed_qrels_from_receipt(raw["receipt_path"])
        if verified.receipt.receipt_sha256 != qrels.receipt.receipt_sha256:
            raise RuntimeError("qrels read-side verification returned a different identity")
        return {"receipt_path": raw["receipt_path"], "receipt_sha256": qrels.receipt.receipt_sha256, "pair_count": qrels.receipt.pair_count, "query_count": qrels.receipt.query_count, "document_count": qrels.receipt.document_count, "contract_sha256": qrels.contract_sha256, "storage_authority": "disk-backed-sqlite"}
    finally:
        close_authoritative_governed_qrels(qrels)
        if verified is not None:
            close_authoritative_governed_qrels(verified)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse exact local qrels into a disk-backed governed, restart-verifiable receipt")
    parser.add_argument("config", help="rigorousrag-governed-qrels-config/v1 JSON file")
    print(json.dumps(run_qrels_config(parser.parse_args(argv).config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_qrels_config"]
