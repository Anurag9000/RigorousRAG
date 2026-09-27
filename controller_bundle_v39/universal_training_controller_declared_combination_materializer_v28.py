#!/usr/bin/env python3
"""Materialize source-proven repository-declared scientific combinations.

v27 audits literal repository-authored model/dataset/task/loss/method/etc.
combinations.  This v28 layer closes the implementation gap for combinations
that are unambiguously executable through an already-authoritative training
entrypoint.

Safety / scientific-contract rules:
* never form a Cartesian product;
* only consume concrete rows already inventoried by v27;
* only clone an existing *training* job;
* every value in the row must map to a selector that is either already present
  in the command or explicitly declared by the trainer's argparse/click parser;
* all candidate jobs for a row must collapse to one trainer command signature;
* preserve every non-scientific argument, dependency, checkpoint/resume field,
  lifecycle field and artifact declaration from the authoritative template;
* ambiguous rows remain unresolved and therefore fail the existing strict v27
  combination-closure contract rather than being guessed.

This module implements scientific job construction only.  It contains no
resource admission, concurrency, RAM/VRAM/swap gating, pause/resume, retry,
CUDA-OOM fallback or device scheduling.  Materialized jobs flow through the
unchanged byte-pinned OPF_ADP scheduler.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import universal_training_controller_combination_closure_v27 as combinations
import universal_training_controller_current as current
import universal_training_controller_lifecycle as lifecycle

MATERIALIZER_SCHEMA = 1
_CAPTURED = None

_FLAG_DIMENSIONS: Dict[str, str] = {
    "--model": "model", "--model-name": "model", "--model_name": "model",
    "--architecture": "model", "--arch": "model",
    "--backbone": "backbone",
    "--encoder": "encoder", "--decoder": "decoder", "--head": "head",
    "--learner": "learner", "--trainer": "trainer",
    "--loss": "loss", "--loss-name": "loss", "--loss_name": "loss",
    "--objective": "loss", "--criterion": "loss",
    "--dataset": "dataset", "--dataset-name": "dataset", "--dataset_name": "dataset",
    "--data": "dataset", "--data-name": "dataset", "--data_name": "dataset",
    "--datamodule": "dataset", "--data-module": "dataset", "--data_module": "dataset",
    "--corpus": "dataset", "--benchmark": "dataset",
    "--task": "task", "--task-name": "task", "--task_name": "task",
    "--target": "task", "--label-space": "task", "--label_space": "task",
    "--method": "method", "--algorithm": "method", "--strategy": "method",
    "--approach": "method", "--policy": "method", "--agent": "method",
    "--optimizer": "optimizer", "--optimiser": "optimizer",
    "--scheduler": "scheduler", "--lr-scheduler": "scheduler", "--lr_scheduler": "scheduler",
    "--sampler": "sampler",
    "--augmentation": "augmentation", "--transform": "augmentation",
    "--preprocessor": "preprocessor", "--tokenizer": "preprocessor", "--feature": "feature",
    "--ensemble": "ensemble", "--fusion": "ensemble", "--cascade": "ensemble",
    "--stacker": "ensemble", "--blender": "ensemble",
    "--experiment": "pipeline", "--pipeline": "pipeline", "--workflow": "pipeline",
    "--stage": "pipeline", "--recipe": "pipeline",
    "--environment": "environment", "--env": "environment", "--domain": "environment",
    "--scenario": "environment", "--regime": "environment",
    "--evaluator": "evaluation", "--metric": "evaluation", "--scorer": "evaluation",
}
_DIMENSION_FLAGS: Dict[str, Tuple[str, ...]] = {}
for _flag, _dim in _FLAG_DIMENSIONS.items():
    _DIMENSION_FLAGS.setdefault(_dim, tuple())
    _DIMENSION_FLAGS[_dim] = (*_DIMENSION_FLAGS[_dim], _flag)


def _entrypoint(root: Path, command: Sequence[str]) -> Path | None:
    values = [str(value) for value in command]
    for token in values:
        if token.endswith(".py"):
            candidate = Path(token)
            candidate = candidate if candidate.is_absolute() else root / candidate
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root.resolve())
            except Exception:
                continue
            if resolved.is_file():
                return resolved
    for index, token in enumerate(values[:-1]):
        if token == "-m":
            module = values[index + 1].strip()
            if not module or module.startswith("-"):
                continue
            relative = Path(*module.split("."))
            for base in (root, root / "src"):
                for candidate in (base / relative.with_suffix(".py"), base / relative / "__main__.py"):
                    if candidate.is_file():
                        return candidate.resolve()
    return None


def _parser_flags(path: Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    result: set[str] = set()
    # argparse / parser.add_argument / group.add_argument.
    for match in re.finditer(r"\.add_argument\s*\((.*?)\)", text, re.S):
        body = match.group(1)
        for flag in re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", body):
            result.add(flag)
    # click/typer-style option declarations.
    for match in re.finditer(r"(?:click\.option|typer\.Option)\s*\((.*?)\)", text, re.S):
        for flag in re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", match.group(1)):
            result.add(flag)
    return result


def _present_slots(command: Sequence[str]) -> Dict[str, Tuple[int, str, bool]]:
    """Return dimension -> (index, flag, equals_form)."""
    values = [str(value) for value in command]
    result: Dict[str, Tuple[int, str, bool]] = {}
    index = 0
    while index < len(values):
        token = values[index]
        if token in _FLAG_DIMENSIONS and index + 1 < len(values):
            result.setdefault(_FLAG_DIMENSIONS[token], (index, token, False))
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            flag = token.split("=", 1)[0]
            dim = _FLAG_DIMENSIONS.get(flag)
            if dim:
                result.setdefault(dim, (index, flag, True))
        index += 1
    return result


def _supported_dimensions(root: Path, command: Sequence[str]) -> Dict[str, str]:
    present = _present_slots(command)
    result = {dim: slot[1] for dim, slot in present.items()}
    parser_flags = _parser_flags(_entrypoint(root, command))
    for dim, aliases in _DIMENSION_FLAGS.items():
        if dim in result:
            continue
        for flag in aliases:
            if flag in parser_flags:
                result[dim] = flag
                break
    return result


def _normalized_signature(root: Path, command: Sequence[str], row_dimensions: set[str]) -> Tuple[str, ...]:
    values = [str(value) for value in command]
    slots = _present_slots(values)
    for dim in row_dimensions:
        slot = slots.get(dim)
        if slot is None:
            continue
        index, flag, equals_form = slot
        if equals_form:
            values[index] = f"{flag}=<{dim.upper()}>"
        elif index + 1 < len(values):
            values[index + 1] = f"<{dim.upper()}>"
    # A parser-supported but currently absent selector is part of the same
    # trainer signature; it is deliberately not added here.
    return tuple(values)


def _rewrite_command(root: Path, command: Sequence[str], values: Mapping[str, str]) -> list[str] | None:
    result = [str(value) for value in command]
    supported = _supported_dimensions(root, result)
    if not set(values).issubset(supported):
        return None
    slots = _present_slots(result)
    for dim, scientific_value in values.items():
        slot = slots.get(dim)
        if slot is not None:
            index, flag, equals_form = slot
            if equals_form:
                result[index] = f"{flag}={scientific_value}"
            else:
                if index + 1 >= len(result):
                    return None
                result[index + 1] = str(scientific_value)
            continue
        flag = supported[dim]
        result.extend([flag, str(scientific_value)])
    return result


def _job_id(surface: Mapping[str, Any], values: Mapping[str, str]) -> str:
    payload = f"{surface.get('path')}|{surface.get('line')}|{surface.get('fingerprint')}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    summary = "-".join(re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:24] for value in values.values())
    summary = summary.strip("-")[:72] or "declared"
    return f"declared-combo:auto:{summary}:{digest}"


def _records(root: Path, profile: Dict[str, Any]):
    assert _CAPTURED is not None
    records = [dict(row) for row in _CAPTURED(root, profile)]
    if profile.get("disable_declared_combination_materialization") is True:
        return records

    training = [row for row in records if lifecycle._normalized_phase(row) == "training"]
    surfaces = combinations._inventory_surfaces(root)
    seen_ids = {str(row.get("id") or "") for row in records}
    seen_commands = {tuple(str(value) for value in (row.get("command") or [])) for row in records}
    added: list[Dict[str, Any]] = []
    unresolved: list[Dict[str, Any]] = []

    for surface in surfaces:
        scientific = dict(surface.get("values") or {})
        if not scientific:
            continue
        if any(combinations._job_covers_values(job, scientific) for job in records):
            continue

        row_dims = set(scientific)
        candidates: list[tuple[Dict[str, Any], list[str], Tuple[str, ...]]] = []
        for template in training:
            command = [str(value) for value in (template.get("command") or [])]
            if not command:
                continue
            rewritten = _rewrite_command(root, command, scientific)
            if rewritten is None:
                continue
            signature = _normalized_signature(root, command, row_dims)
            candidates.append((template, rewritten, signature))

        signatures = {candidate[2] for candidate in candidates}
        if len(signatures) != 1:
            unresolved.append({
                "path": surface.get("path"),
                "line": surface.get("line"),
                "fingerprint": surface.get("fingerprint"),
                "values": scientific,
                "reason": "no_source_proven_trainer" if not candidates else "ambiguous_trainer_signatures",
                "candidate_count": len(candidates),
                "signature_count": len(signatures),
            })
            continue

        # Multiple curated jobs with the same normalized signature are equivalent
        # templates for selector materialization.  Pick deterministically and
        # preserve every non-selector field from that authoritative job.
        candidates.sort(key=lambda item: str(item[0].get("id") or ""))
        template, command, _ = candidates[0]
        command_key = tuple(command)
        if command_key in seen_commands:
            continue
        clone = dict(template)
        base_id = _job_id(surface, scientific)
        job_id = base_id
        suffix = 2
        while job_id in seen_ids:
            job_id = f"{base_id}:{suffix}"
            suffix += 1
        clone["id"] = job_id
        clone["command"] = command
        for dim, value in scientific.items():
            clone[dim] = value
        clone["declared_combination_materialized"] = True
        clone["declared_combination_materializer_schema"] = MATERIALIZER_SCHEMA
        clone["declared_combination_source"] = {
            "path": surface.get("path"),
            "line": surface.get("line"),
            "symbol": surface.get("symbol"),
            "fingerprint": surface.get("fingerprint"),
            "surface_kind": surface.get("surface_kind"),
            "template_job": template.get("id"),
        }
        records.append(clone)
        training.append(clone)
        seen_ids.add(job_id)
        seen_commands.add(command_key)
        added.append({"job_id": job_id, "values": scientific, "source": clone["declared_combination_source"]})

    profile["declared_combination_materialization"] = {
        "schema": MATERIALIZER_SCHEMA,
        "enabled": True,
        "added_job_count": len(added),
        "added_jobs": added,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "policy": "repository_rows_plus_source_proven_selectors_only",
        "cartesian_products_invented": False,
    }
    return records


def install() -> None:
    global _CAPTURED
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_declared_combo_materializer_v28", False):
        return
    _CAPTURED = current._ORIGINAL_JOB_RECORDS
    _records._training_control_declared_combo_materializer_v28 = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _records


__all__ = ["MATERIALIZER_SCHEMA", "install"]
