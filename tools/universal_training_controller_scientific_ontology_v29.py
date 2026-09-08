#!/usr/bin/env python3
"""Scientific ontology expansion for universal controller v29.

The v25-v28 controller stack already closes repository-declared models,
architectures/backbones/heads, losses/objectives, datasets/benchmarks, tasks,
methods/algorithms/strategies/policies, optimizers/schedulers, preprocessing,
ensembles, pipelines/workflows/recipes, environments, trainers/evaluators,
metrics and concrete repository-authored combinations.

v29 closes the remaining taxonomy gap that occurs in repositories where those
choices are grouped behind an intermediate scientific identity rather than a
literal ``model``/``method`` selector: model/architecture/network/learner
families, model/architecture/network types, implementation/model variants,
capabilities/functionalities, and regularization families.

The layer mutates only the static scientific vocabulary consumed by the existing
v25-v28 inventory, selector, member and combination-closure machinery.  It does
not create a Cartesian product and contains no resource scheduling.  All ready
jobs continue to execute through the unchanged byte-pinned OPF_ADP scheduler.
"""
from __future__ import annotations

import re
from typing import Any, Dict

import universal_training_controller_combination_closure_v27 as combinations
import universal_training_controller_declared_combination_materializer_v28 as materializer
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_scientific_surface_v25 as scientific
import universal_training_controller_selector_closure_v26 as selectors
import universal_training_controller_workload_closure as workload
import universal_training_controller_current as current

ONTOLOGY_SCHEMA = 1

_ADDED_TOKENS = (
    "MODEL_FAMILY|MODEL_FAMILIES|ARCHITECTURE_FAMILY|ARCHITECTURE_FAMILIES|"
    "NETWORK_FAMILY|NETWORK_FAMILIES|LEARNER_FAMILY|LEARNER_FAMILIES|"
    "METHOD_FAMILY|METHOD_FAMILIES|ALGORITHM_FAMILY|ALGORITHM_FAMILIES|"
    "MODEL_TYPE|MODEL_TYPES|ARCHITECTURE_TYPE|ARCHITECTURE_TYPES|NETWORK_TYPE|NETWORK_TYPES|"
    "LEARNER_TYPE|LEARNER_TYPES|MODEL_VARIANT|MODEL_VARIANTS|ARCHITECTURE_VARIANT|"
    "ARCHITECTURE_VARIANTS|IMPLEMENTATION_VARIANT|IMPLEMENTATION_VARIANTS|VARIANT|VARIANTS|"
    "CAPABILITY|CAPABILITIES|FUNCTIONALITY|FUNCTIONALITIES|"
    "REGULARIZER|REGULARIZERS|REGULARISER|REGULARISERS|REGULARIZATION|REGULARISATION"
)

_IDENTITY_KEYS = {
    "family", "model_family", "architecture_family", "network_family", "learner_family",
    "method_family", "algorithm_family", "model_type", "architecture_type", "network_type",
    "learner_type", "variant", "model_variant", "architecture_variant", "implementation_variant",
    "capability", "functionality", "regularizer", "regulariser", "regularization", "regularisation",
}

_ALL_FLAGS = {
    "--all-families", "--all-model-families", "--all-architecture-families", "--all-network-families",
    "--all-model-types", "--all-architecture-types", "--all-network-types", "--all-learner-types",
    "--all-variants", "--all-model-variants", "--all-architecture-variants",
    "--all-capabilities", "--all-functionalities", "--all-regularizers", "--all-regularisers",
}

_KEY_ALIASES: Dict[str, str] = {
    "family": "family",
    "model_family": "family",
    "architecture_family": "family",
    "network_family": "family",
    "learner_family": "family",
    "method_family": "family",
    "algorithm_family": "family",
    "model_type": "model_type",
    "architecture_type": "model_type",
    "network_type": "model_type",
    "learner_type": "model_type",
    "variant": "variant",
    "model_variant": "variant",
    "architecture_variant": "variant",
    "implementation_variant": "variant",
    "capability": "capability",
    "functionality": "capability",
    "regularizer": "regularizer",
    "regulariser": "regularizer",
    "regularization": "regularizer",
    "regularisation": "regularizer",
}

_FLAG_DIMENSIONS: Dict[str, str] = {
    "--family": "family", "--model-family": "family", "--model_family": "family",
    "--architecture-family": "family", "--architecture_family": "family",
    "--network-family": "family", "--network_family": "family",
    "--method-family": "family", "--method_family": "family",
    "--algorithm-family": "family", "--algorithm_family": "family",
    "--model-type": "model_type", "--model_type": "model_type",
    "--architecture-type": "model_type", "--architecture_type": "model_type",
    "--network-type": "model_type", "--network_type": "model_type",
    "--learner-type": "model_type", "--learner_type": "model_type",
    "--variant": "variant", "--model-variant": "variant", "--model_variant": "variant",
    "--architecture-variant": "variant", "--architecture_variant": "variant",
    "--implementation-variant": "variant", "--implementation_variant": "variant",
    "--capability": "capability", "--functionality": "capability",
    "--regularizer": "regularizer", "--regulariser": "regularizer",
    "--regularization": "regularizer", "--regularisation": "regularizer",
}


def _extend_selector_regex() -> None:
    old = selectors._SELECTOR_TOKEN_RE.pattern
    # Preserve the established boundary convention while adding only strong
    # scientific compound names plus the explicit variant/capability vocabulary.
    extra = (
        r"model_family|model_families|architecture_family|architecture_families|"
        r"network_family|network_families|learner_family|learner_families|"
        r"method_family|method_families|algorithm_family|algorithm_families|"
        r"model_type|model_types|architecture_type|architecture_types|network_type|network_types|"
        r"learner_type|learner_types|model_variant|model_variants|architecture_variant|"
        r"architecture_variants|implementation_variant|implementation_variants|variant|variants|"
        r"capability|capabilities|functionality|functionalities|regularizer|regularizers|"
        r"regulariser|regularisers|regularization|regularisation"
    )
    marker = r")(?:_|$)"
    if marker in old:
        old = old.replace(marker, "|" + extra + marker, 1)
    selectors._SELECTOR_TOKEN_RE = re.compile(old, re.I)


def _extend_component_regex() -> None:
    old = selectors._COMPONENT_DIR_RE.pattern
    marker = r")(?:/|$)"
    extra = (
        r"model_families?|architecture_families?|network_families?|learner_families?|"
        r"model_types?|architecture_types?|network_types?|learner_types?|"
        r"model_variants?|architecture_variants?|implementation_variants?|variants?|"
        r"capabilities|functionalities|regularizers?|regularisers?"
    )
    if marker in old:
        old = old.replace(marker, "|" + extra + marker, 1)
    selectors._COMPONENT_DIR_RE = re.compile(old, re.I)


def _rebuild_materializer_dimensions() -> None:
    materializer._FLAG_DIMENSIONS.update(_FLAG_DIMENSIONS)
    rebuilt: Dict[str, tuple[str, ...]] = {}
    for flag, dimension in materializer._FLAG_DIMENSIONS.items():
        rebuilt.setdefault(dimension, tuple())
        rebuilt[dimension] = (*rebuilt[dimension], flag)
    materializer._DIMENSION_FLAGS = rebuilt


def install_primitives() -> None:
    # v25 registry/config vocabulary.
    tokens = scientific._SCIENTIFIC_TOKENS
    if _ADDED_TOKENS not in tokens:
        tokens = tokens + "|" + _ADDED_TOKENS
    scientific._SCIENTIFIC_TOKENS = tokens
    scientific.SCIENTIFIC_REGISTRY_RE = re.compile(
        r"(?:^|_)(?:" + tokens + r")(?:_|$)", re.I
    )
    scientific.SCIENTIFIC_IDENTITY_KEYS = set(scientific.SCIENTIFIC_IDENTITY_KEYS) | _IDENTITY_KEYS
    scientific.SCIENTIFIC_TRAINING_KEYS = set(scientific.SCIENTIFIC_TRAINING_KEYS) | {
        "regularizer", "regulariser", "regularization", "regularisation",
    }
    scientific.SCIENTIFIC_ALL_FLAGS = set(scientific.SCIENTIFIC_ALL_FLAGS) | _ALL_FLAGS
    scientific.SCIENTIFIC_CONFIG_HINT_RE = re.compile(
        scientific.SCIENTIFIC_CONFIG_HINT_RE.pattern.rstrip(")")
        + r"|family|model_type|architecture_type|variant|capability|functionality|regulari[sz])",
        re.I,
    )

    workload.REGISTRY_RE = scientific.SCIENTIFIC_REGISTRY_RE
    workload.IDENTITY_KEYS = set(workload.IDENTITY_KEYS) | _IDENTITY_KEYS
    workload.TRAINING_KEYS = set(workload.TRAINING_KEYS) | {
        "regularizer", "regulariser", "regularization", "regularisation",
    }
    workload.CONFIG_HINT_RE = scientific.SCIENTIFIC_CONFIG_HINT_RE
    registry_members.ALL_FLAGS = set(registry_members.ALL_FLAGS) | _ALL_FLAGS

    # v26 selector/component vocabulary.
    _extend_selector_regex()
    _extend_component_regex()

    # v27 concrete repository-authored combination vocabulary.
    combinations._KEY_ALIASES.update(_KEY_ALIASES)
    combinations._PRIMARY_DIMENSIONS.update({"family", "model_type", "variant", "capability", "regularizer"})

    # v28 execution materialization selectors.
    _rebuild_materializer_dimensions()


def install_contract() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root, profile, jobs):
        report = original_report(root, profile, jobs)
        controls = dict(report.get("strict_controls") or {})
        require = bool(profile.get("require_scientific_ontology_accounting", profile.get("strict_coverage", True)))
        controls["require_scientific_ontology_accounting"] = require
        report.update({
            "scientific_ontology_schema": ONTOLOGY_SCHEMA,
            "scientific_ontology_added_dimensions": [
                "family", "model_type", "variant", "capability", "regularizer"
            ],
            "scientific_ontology_cartesian_products_invented": False,
            "strict_controls": controls,
        })
        # Coverage itself is already enforced by the downstream v25-v28 member,
        # selector and combination layers using the expanded vocabulary above.
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["ONTOLOGY_SCHEMA", "install_primitives", "install_contract"]
