#!/usr/bin/env python3
"""Broaden exhaustive workload closure to the full scientific experiment surface.

v24 made model/dataset/task/method/config coverage fail closed.  v25 extends the
same source-proven contract to every statically declared experiment selector that
can materially change scientific behavior: architectures/backbones/heads,
losses/objectives/criteria, data sources and benchmarks, tasks/targets,
algorithms/strategies/policies/agents, optimizers and schedulers, samplers and
preprocessing/tokenization/augmentation/features, ensembles/fusion/cascades,
pipelines/workflows/recipes/stages, environments/scenarios/regimes, trainers,
evaluators/metrics/scorers, and related registries.

This module is deliberately an accounting layer, not a scheduler and not an
unsafe Cartesian-product generator.  Repository-authored job catalogs/configs
remain the compatibility authority.  Missing declared members fail closure so
that the repository-specific catalog or trainer must be repaired explicitly.
All ready jobs continue to execute through the unchanged literal OPF_ADP
scheduler.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_workload_closure as workload

SCIENTIFIC_SURFACE_SCHEMA = 1

# Upper-case repository registries are treated as contractual only when their
# symbol name carries one of these strong semantic tokens.  This intentionally
# avoids generic constants such as DEFAULTS/OPTIONS/CONFIGS.
_SCIENTIFIC_TOKENS = (
    "MODEL|MODELS|ARCH|ARCHS|ARCHITECTURE|ARCHITECTURES|BACKBONE|BACKBONES|"
    "ENCODER|ENCODERS|DECODER|DECODERS|HEAD|HEADS|LEARNER|LEARNERS|TRAINER|TRAINERS|"
    "LOSS|LOSSES|OBJECTIVE|OBJECTIVES|CRITERION|CRITERIA|"
    "DATASET|DATASETS|DATAMODULE|DATAMODULES|DATA_MODULE|DATA_MODULES|DATA_SOURCE|DATA_SOURCES|"
    "CORPUS|CORPORA|BENCHMARK|BENCHMARKS|"
    "TASK|TASKS|TARGET|TARGETS|LABEL_SPACE|LABEL_SPACES|"
    "METHOD|METHODS|ALGORITHM|ALGORITHMS|STRATEGY|STRATEGIES|APPROACH|APPROACHES|"
    "POLICY|POLICIES|AGENT|AGENTS|"
    "OPTIMIZER|OPTIMIZERS|OPTIMISER|OPTIMISERS|SCHEDULER|SCHEDULERS|LR_SCHEDULER|LR_SCHEDULERS|"
    "SAMPLER|SAMPLERS|AUGMENTATION|AUGMENTATIONS|TRANSFORM|TRANSFORMS|"
    "PREPROCESSOR|PREPROCESSORS|TOKENIZER|TOKENIZERS|FEATURE|FEATURES|"
    "ENSEMBLE|ENSEMBLES|FUSION|FUSIONS|CASCADE|CASCADES|STACKER|STACKERS|BLENDER|BLENDERS|"
    "EXPERIMENT|EXPERIMENTS|PIPELINE|PIPELINES|WORKFLOW|WORKFLOWS|STAGE|STAGES|RECIPE|RECIPES|"
    "ENVIRONMENT|ENVIRONMENTS|DOMAIN|DOMAINS|SCENARIO|SCENARIOS|REGIME|REGIMES|"
    "EVALUATOR|EVALUATORS|METRIC|METRICS|SCORER|SCORERS"
)
SCIENTIFIC_REGISTRY_RE = re.compile(r"(?:^|_)(?:" + _SCIENTIFIC_TOKENS + r")(?:_|$)", re.I)

SCIENTIFIC_IDENTITY_KEYS = {
    "model", "model_name", "architecture", "arch", "backbone", "encoder", "decoder", "head",
    "learner", "trainer", "loss", "loss_name", "objective", "criterion",
    "dataset", "dataset_name", "datamodule", "data_module", "data_source", "corpus", "benchmark",
    "task", "task_name", "target", "label_space",
    "method", "algorithm", "strategy", "approach", "policy", "agent",
    "optimizer", "optimiser", "scheduler", "lr_scheduler",
    "sampler", "augmentation", "transform", "preprocessor", "tokenizer", "feature",
    "ensemble", "fusion", "cascade", "stacker", "blender",
    "experiment", "pipeline", "workflow", "stage", "recipe",
    "environment", "domain", "scenario", "regime",
    "evaluator", "metric", "metrics", "scorer",
}
SCIENTIFIC_TRAINING_KEYS = {
    "training", "trainer", "optimizer", "optimiser", "learning_rate", "lr", "epochs", "max_epochs",
    "steps", "max_steps", "batch_size", "micro_batch_size", "gradient_accumulation_steps",
    "loss", "criterion", "objective", "scheduler", "lr_scheduler", "weight_decay", "warmup_steps",
    "checkpoint", "checkpointing", "resume", "early_stopping", "patience", "min_delta", "precision",
}
SCIENTIFIC_CONFIG_HINT_RE = re.compile(
    r"(?:config|recipe|experiment|train|training|model|architecture|backbone|loss|objective|dataset|task|"
    r"benchmark|method|algorithm|strategy|optimizer|scheduler|ensemble|pipeline|workflow|environment|scenario|regime)",
    re.I,
)
SCIENTIFIC_ALL_FLAGS = {
    "--all", "--all-models", "--all-model", "--all-backbones", "--all-architectures", "--all-heads",
    "--all-losses", "--all-objectives", "--all-criteria", "--all-datasets", "--all-dataset",
    "--all-benchmarks", "--all-tasks", "--all-targets", "--all-methods", "--all-algorithms",
    "--all-strategies", "--all-policies", "--all-agents", "--all-optimizers", "--all-optimisers",
    "--all-schedulers", "--all-samplers", "--all-augmentations", "--all-transforms",
    "--all-preprocessors", "--all-tokenizers", "--all-features", "--all-ensembles", "--all-fusions",
    "--all-cascades", "--all-pipelines", "--all-workflows", "--all-recipes", "--all-benchmarks",
    "--all-environments", "--all-domains", "--all-scenarios", "--all-regimes", "--all-metrics",
    "--all-evaluators", "--all-scorers",
}


def _literal_members(value: ast.AST | None) -> list[str]:
    if isinstance(value, ast.Dict):
        return sorted({str(key.value) for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, (str, int, float))})
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return sorted({str(item.value) for item in value.elts if isinstance(item, ast.Constant) and isinstance(item.value, (str, int, float))})
    return []


def _scientific_registry_findings(path: Path, rel: str) -> list[Dict[str, Any]]:
    """Enumerate strong static registries and keep dynamic assignments visible.

    Unlike the v22 scanner, a strong symbol assigned from a call/comprehension is
    retained with ``members=[]``.  This prevents a dynamic registry from
    disappearing from the audit merely because its members cannot be enumerated
    safely without executing repository code.
    """
    if path.suffix.lower() != ".py":
        return []
    text = workload._read(path)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []
    rows: list[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if not isinstance(target, ast.Name) or not SCIENTIFIC_REGISTRY_RE.search(target.id):
                continue
            key = (target.id, int(getattr(node, "lineno", 0)))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "path": rel,
                "symbol": target.id,
                "line": key[1],
                "members": _literal_members(value),
                "enumerable": isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)),
                "assignment_kind": type(value).__name__ if value is not None else "None",
            })
    return rows


def _normalize_dynamic_covers(profile: Mapping[str, Any]) -> set[str]:
    raw = profile.get("dynamic_registry_covers") or []
    if isinstance(raw, str):
        raw = [raw]
    result: set[str] = set()
    for item in raw if isinstance(raw, (list, tuple, set)) else []:
        if isinstance(item, str):
            result.add(item.strip())
        elif isinstance(item, Mapping):
            path = str(item.get("path") or "").strip()
            symbol = str(item.get("symbol") or "").strip()
            if path and symbol:
                result.add(f"{path}:{symbol}")
            elif path:
                result.add(path)
    return {value for value in result if value}


def install_primitives() -> None:
    workload.REGISTRY_RE = SCIENTIFIC_REGISTRY_RE
    workload.IDENTITY_KEYS = set(workload.IDENTITY_KEYS) | SCIENTIFIC_IDENTITY_KEYS
    workload.TRAINING_KEYS = set(workload.TRAINING_KEYS) | SCIENTIFIC_TRAINING_KEYS
    workload.CONFIG_HINT_RE = SCIENTIFIC_CONFIG_HINT_RE
    workload._registry_findings = _scientific_registry_findings
    registry_members.ALL_FLAGS = set(registry_members.ALL_FLAGS) | SCIENTIFIC_ALL_FLAGS


def install_contract() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        inventory = (report.get("registry_member_inventory") or {}) if isinstance(report, Mapping) else {}
        non_enumerable = list(inventory.get("non_enumerable_registry_surfaces") or []) if isinstance(inventory, Mapping) else []
        covers = _normalize_dynamic_covers(profile)
        unresolved = []
        for row in non_enumerable:
            if not isinstance(row, Mapping):
                continue
            path = str(row.get("path") or "")
            symbol = str(row.get("symbol") or "")
            key = f"{path}:{symbol}"
            if key in covers or path in covers:
                continue
            unresolved.append(dict(row))
        require_dynamic = bool(profile.get("require_dynamic_registry_accounting", profile.get("strict_coverage", True)))
        controls = dict(report.get("strict_controls") or {})
        controls["require_dynamic_registry_accounting"] = require_dynamic
        report.update({
            "scientific_surface_schema": SCIENTIFIC_SURFACE_SCHEMA,
            "scientific_registry_pattern": SCIENTIFIC_REGISTRY_RE.pattern,
            "unaccounted_dynamic_scientific_registries": unresolved,
            "declared_dynamic_registry_covers": sorted(covers),
            "strict_dynamic_scientific_registry_pass": not unresolved,
            "strict_controls": controls,
        })
        if require_dynamic and unresolved:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = [
    "SCIENTIFIC_SURFACE_SCHEMA", "SCIENTIFIC_REGISTRY_RE", "SCIENTIFIC_IDENTITY_KEYS",
    "SCIENTIFIC_TRAINING_KEYS", "SCIENTIFIC_ALL_FLAGS", "install_primitives", "install_contract",
]
