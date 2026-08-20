"""Canonical recipe v2 operator with explicit stage resolution.

V2 is an operator envelope around the stable advanced-training config v1 schema.  It adds two
source-completeness guarantees without changing trainer semantics:

* optional per-curriculum-stage ``max_steps``, ``checkpoint_every_steps`` and ``learning_rate``
  overrides are atomically applied then re-parsed through the canonical v1 config authority;
* a resolved-plan artifact is always emitted, making objective weights/defaults, execution,
  collator, trainability, split/cache/model identities and checkpoint settings explicit.

The resulting training config remains ``rigorousrag-advanced-training-config/v1`` and is consumed
unchanged by ``rigorousrag-advanced-training``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_resolved_plan import (
    apply_curriculum_stage_overrides,
    write_resolved_training_plan,
)
from training.authoritative_canonical_recipe_cli import (
    run_dynamic_canonical_recipe_config,
    run_grounded_canonical_recipe_config,
)

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_GROUNDED_V2 = "rigorousrag-authoritative-grounded-canonical-recipe-config/v2"
_DYNAMIC_V2 = "rigorousrag-authoritative-dynamic-canonical-recipe-config/v2"


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="canonical recipe v2 config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("canonical recipe v2 config exceeds byte safety bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("canonical recipe v2 config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("canonical recipe v2 config must contain an object")
    return value


def _base_v1(raw: Mapping[str, Any], *, grounded: bool) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]] | None, str]:
    selected = dict(raw)
    stage_overrides = selected.pop("curriculum_stages", None)
    resolved_output = selected.pop("resolved_plan_output_path", None)
    if not isinstance(resolved_output, str) or not resolved_output.strip():
        raise ValueError("canonical recipe v2 requires resolved_plan_output_path")
    if stage_overrides is not None and not isinstance(stage_overrides, Mapping):
        raise ValueError("curriculum_stages must be an object when supplied")
    selected["schema"] = (
        "rigorousrag-authoritative-grounded-canonical-recipe-config/v1"
        if grounded
        else "rigorousrag-authoritative-dynamic-canonical-recipe-config/v1"
    )
    return selected, stage_overrides, resolved_output


def _finish(result: Mapping[str, Any], stage_overrides: Mapping[str, Mapping[str, Any]] | None, resolved_output: str) -> Mapping[str, Any]:
    config_path = result.get("config_path")
    if not isinstance(config_path, str) or not config_path:
        raise RuntimeError("canonical recipe v1 operator returned no config_path")
    selected = dict(result)
    if stage_overrides is not None:
        receipt = apply_curriculum_stage_overrides(config_path, stage_overrides)
        selected.update({
            "kind": receipt.kind,
            "run_id": receipt.run_id,
            "config_path": receipt.output_path,
            "config_sha256": receipt.config_sha256,
            "plan_sha256": receipt.plan_sha256,
            "recipe_receipt_sha256": receipt.receipt_sha256,
        })
    resolved = write_resolved_training_plan(config_path, resolved_output)
    if resolved["plan_sha256"] != selected["plan_sha256"]:
        raise RuntimeError("resolved plan identity differs from recipe receipt")
    selected.update({
        "resolved_plan_path": str(Path(resolved_output).expanduser().resolve()),
        "resolved_plan_sha256": resolved["resolved_plan_sha256"],
        "recipe_operator_schema": "rigorousrag-authoritative-canonical-recipe-operator/v2",
    })
    return selected


def run_canonical_recipe_v2_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    schema = raw.get("schema")
    if schema == _GROUNDED_V2:
        base, overrides, resolved_output = _base_v1(raw, grounded=True)
        result = run_grounded_canonical_recipe_config(base)
        return _finish(result, overrides, resolved_output)
    if schema == _DYNAMIC_V2:
        base, overrides, resolved_output = _base_v1(raw, grounded=False)
        result = run_dynamic_canonical_recipe_config(base)
        return _finish(result, overrides, resolved_output)
    raise ValueError("unsupported authoritative canonical recipe v2 schema")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit canonical advanced-RAG training config plus exact resolved plan"
    )
    parser.add_argument("config", help="grounded/dynamic authoritative canonical recipe v2 JSON")
    result = run_canonical_recipe_v2_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_canonical_recipe_v2_config"]
