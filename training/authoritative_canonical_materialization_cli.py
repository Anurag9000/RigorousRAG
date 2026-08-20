"""Strict config-only CLI for authoritative Grounded/Dynamic canonical-v2 materialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.authoritative_canonical_materialization import (
    run_dynamic_canonical_materialization_config,
    run_grounded_canonical_materialization_config,
)
from training.production_canonical_limits import (
    assert_production_split_count,
    grounded_source_split_count_from_receipt,
)

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(
        path,
        label="canonical materialization config",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("canonical materialization config exceeds byte safety bound")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("canonical materialization config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("canonical materialization config must contain an object")
    return value


def _assert_grounded_source_limit(raw: Mapping[str, Any]) -> None:
    receipt_path = raw.get("source_receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("grounded canonical materialization requires source_receipt_path")
    count = grounded_source_split_count_from_receipt(receipt_path)
    assert_production_split_count(count, label="grounded source split count")


def run_canonical_materialization_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    schema = raw.get("schema")
    if schema == "rigorousrag-authoritative-grounded-canonical-materialization-config/v1":
        _assert_grounded_source_limit(raw)
        result = run_grounded_canonical_materialization_config(raw)
        assert_production_split_count(int(result["split_count"]), label="grounded canonical output split count")
        return result
    if schema == "rigorousrag-authoritative-dynamic-canonical-materialization-config/v1":
        result = run_dynamic_canonical_materialization_config(raw)
        assert_production_split_count(int(result["split_count"]), label="dynamic canonical output split count")
        return result
    raise ValueError("unsupported authoritative canonical materialization config schema")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize authoritative Grounded/Dynamic canonical-v2 training data from exact local artifacts"
    )
    parser.add_argument("config", help="strict canonical materialization JSON config")
    result = run_canonical_materialization_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_canonical_materialization_config"]
