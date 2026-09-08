#!/usr/bin/env python3
"""Universal controller v33: role/paradigm/protocol scientific-choice closure.

v25-v32 already close the ordinary model/dataset/task/loss/optimizer/pipeline
surface plus architecture, adaptation, compression, retrieval, generative/RL,
continual-learning, graph/diffusion, probabilistic, federated/meta-learning,
search/ablation and data-protocol choices.  v33 closes another practical naming
gap: repositories often expose the same scientifically material alternatives as
roles, paradigms or protocols rather than under literal MODEL/METHOD/TASK names.

The added vocabulary includes model roles (classifier/regressor/detector/etc.),
open-world/discovery/transfer/domain-adaptation and supervision paradigms,
explicit ensemble algorithms, data split/sampling/mining/feature transforms,
RL reward/advantage/exploration choices, stopping/checkpoint policies,
evaluation/validation/test protocols, decision rules and deployment/export
choices.  These are accounting dimensions only.  The controller never invents
arbitrary Cartesian products; compatible combinations remain repository-authored.
All ready jobs still run exclusively through the exact byte-pinned OPF_ADP
scheduler inherited by the controller stack.
"""
from __future__ import annotations

import re
from typing import Dict

import universal_training_controller_combination_closure_v27 as combinations
import universal_training_controller_declared_combination_materializer_v28 as materializer
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_scientific_surface_v25 as scientific
import universal_training_controller_selector_closure_v26 as selectors
import universal_training_controller_workload_closure as workload
import universal_training_controller_current as current

ONTOLOGY_V33_SCHEMA = 1

_ADDED_TOKENS = (
    "NETWORK|NETWORKS|CLASSIFIER|CLASSIFIERS|REGRESSOR|REGRESSORS|DETECTOR|DETECTORS|"
    "SEGMENTER|SEGMENTERS|PREDICTOR|PREDICTORS|ESTIMATOR|ESTIMATORS|FORECASTER|FORECASTERS|"
    "CLUSTERER|CLUSTERERS|ASSIGNER|ASSIGNERS|"
    "QUERY_STRATEGY|QUERY_STRATEGIES|DISCOVERY_METHOD|DISCOVERY_METHODS|OOD_METHOD|OOD_METHODS|"
    "OPEN_SET_METHOD|OPEN_SET_METHODS|NOVELTY_METHOD|NOVELTY_METHODS|"
    "TRANSFER_METHOD|TRANSFER_METHODS|DOMAIN_ADAPTATION_METHOD|DOMAIN_ADAPTATION_METHODS|"
    "TEST_TIME_ADAPTATION_METHOD|TEST_TIME_ADAPTATION_METHODS|"
    "SELF_SUPERVISED_METHOD|SELF_SUPERVISED_METHODS|SEMI_SUPERVISED_METHOD|SEMI_SUPERVISED_METHODS|"
    "WEAK_SUPERVISION_METHOD|WEAK_SUPERVISION_METHODS|MULTITASK_METHOD|MULTITASK_METHODS|"
    "PSEUDO_LABELER|PSEUDO_LABELERS|CONSISTENCY_METHOD|CONSISTENCY_METHODS|"
    "CONTRASTIVE_METHOD|CONTRASTIVE_METHODS|"
    "BAGGER|BAGGERS|BOOSTER|BOOSTERS|VOTER|VOTERS|MIXTURE_MODEL|MIXTURE_MODELS|"
    "EXPERT_MIXTURE|EXPERT_MIXTURES|"
    "DATA_SPLITTER|DATA_SPLITTERS|BATCH_SAMPLER|BATCH_SAMPLERS|NEGATIVE_SAMPLER|NEGATIVE_SAMPLERS|"
    "HARD_MINER|HARD_MINERS|RESAMPLER|RESAMPLERS|NORMALIZER|NORMALIZERS|NORMALISER|NORMALISERS|"
    "SCALER|SCALERS|VECTORIZER|VECTORIZERS|VECTORIZER|VECTORIZERS|FEATURIZER|FEATURIZERS|"
    "FEATURISER|FEATURISERS|MASKING_STRATEGY|MASKING_STRATEGIES|"
    "REWARD_FUNCTION|REWARD_FUNCTIONS|ADVANTAGE_ESTIMATOR|ADVANTAGE_ESTIMATORS|"
    "EXPLORATION_STRATEGY|EXPLORATION_STRATEGIES|ACTION_SELECTOR|ACTION_SELECTORS|"
    "STOPPING_RULE|STOPPING_RULES|EARLY_STOPPING_POLICY|EARLY_STOPPING_POLICIES|"
    "CHECKPOINT_POLICY|CHECKPOINT_POLICIES|"
    "EVALUATION_PROTOCOL|EVALUATION_PROTOCOLS|VALIDATION_PROTOCOL|VALIDATION_PROTOCOLS|"
    "TEST_PROTOCOL|TEST_PROTOCOLS|CV_SCHEME|CV_SCHEMES|"
    "THRESHOLDER|THRESHOLDERS|DECISION_RULE|DECISION_RULES|"
    "EXPORTER|EXPORTERS|EXPORT_FORMAT|EXPORT_FORMATS|SERIALIZER|SERIALIZERS|"
    "COMPILER|COMPILERS|INFERENCE_BACKEND|INFERENCE_BACKENDS|PRECISION_MODE|PRECISION_MODES"
)

_IDENTITY_KEYS = {
    "network", "classifier", "regressor", "detector", "segmenter", "predictor", "estimator", "forecaster",
    "clusterer", "assigner", "query_strategy", "discovery_method", "ood_method", "open_set_method", "novelty_method",
    "transfer_method", "domain_adaptation_method", "test_time_adaptation_method", "self_supervised_method",
    "semi_supervised_method", "weak_supervision_method", "multitask_method", "pseudo_labeler",
    "consistency_method", "contrastive_method", "bagger", "booster", "voter", "mixture_model", "expert_mixture",
    "data_splitter", "batch_sampler", "negative_sampler", "hard_miner", "resampler", "normalizer", "normaliser",
    "scaler", "vectorizer", "vectorizer", "featurizer", "featuriser", "masking_strategy",
    "reward_function", "advantage_estimator", "exploration_strategy", "action_selector",
    "stopping_rule", "early_stopping_policy", "checkpoint_policy", "evaluation_protocol", "validation_protocol",
    "test_protocol", "cv_scheme", "thresholder", "decision_rule", "exporter", "export_format", "serializer",
    "compiler", "inference_backend", "precision_mode",
}

_KEY_ALIASES: Dict[str, str] = {
    **{key: "model_role" for key in (
        "network", "classifier", "regressor", "detector", "segmenter", "predictor", "estimator", "forecaster",
        "clusterer", "assigner",
    )},
    **{key: "open_world_method" for key in (
        "query_strategy", "discovery_method", "ood_method", "open_set_method", "novelty_method",
    )},
    **{key: "adaptation_paradigm" for key in (
        "transfer_method", "domain_adaptation_method", "test_time_adaptation_method",
    )},
    **{key: "supervision_paradigm" for key in (
        "self_supervised_method", "semi_supervised_method", "weak_supervision_method", "multitask_method",
        "pseudo_labeler", "consistency_method", "contrastive_method",
    )},
    **{key: "ensemble_algorithm" for key in (
        "bagger", "booster", "voter", "mixture_model", "expert_mixture",
    )},
    **{key: "data_sampling_component" for key in (
        "data_splitter", "batch_sampler", "negative_sampler", "hard_miner", "resampler",
    )},
    **{key: "data_transform_component" for key in (
        "normalizer", "normaliser", "scaler", "vectorizer", "vectorizer", "featurizer", "featuriser", "masking_strategy",
    )},
    **{key: "rl_learning_component" for key in (
        "reward_function", "advantage_estimator", "exploration_strategy", "action_selector",
    )},
    **{key: "training_control_component" for key in (
        "stopping_rule", "early_stopping_policy", "checkpoint_policy",
    )},
    **{key: "evaluation_protocol" for key in (
        "evaluation_protocol", "validation_protocol", "test_protocol", "cv_scheme",
    )},
    **{key: "decision_component" for key in ("thresholder", "decision_rule")},
    **{key: "deployment_component" for key in (
        "exporter", "export_format", "serializer", "compiler", "inference_backend", "precision_mode",
    )},
}

_FLAG_DIMENSIONS: Dict[str, str] = {}
for _key, _dimension in _KEY_ALIASES.items():
    _FLAG_DIMENSIONS["--" + _key.replace("_", "-")] = _dimension
    _FLAG_DIMENSIONS["--" + _key] = _dimension

_ALL_FLAGS = {
    "--all-networks", "--all-classifiers", "--all-regressors", "--all-detectors", "--all-segmenters",
    "--all-predictors", "--all-estimators", "--all-forecasters", "--all-clusterers",
    "--all-query-strategies", "--all-discovery-methods", "--all-ood-methods", "--all-open-set-methods",
    "--all-transfer-methods", "--all-domain-adaptation-methods", "--all-test-time-adaptation-methods",
    "--all-self-supervised-methods", "--all-semi-supervised-methods", "--all-pseudo-labelers",
    "--all-contrastive-methods", "--all-baggers", "--all-boosters", "--all-voters", "--all-mixture-models",
    "--all-data-splitters", "--all-batch-samplers", "--all-negative-samplers", "--all-hard-miners",
    "--all-resamplers", "--all-normalizers", "--all-scalers", "--all-vectorizers", "--all-featurizers",
    "--all-reward-functions", "--all-advantage-estimators", "--all-exploration-strategies",
    "--all-stopping-rules", "--all-checkpoint-policies", "--all-evaluation-protocols", "--all-cv-schemes",
    "--all-decision-rules", "--all-exporters", "--all-export-formats", "--all-inference-backends",
}


def _extend_selector_regex() -> None:
    old = selectors._SELECTOR_TOKEN_RE.pattern
    extra = (
        r"network|networks|classifier|classifiers|regressor|regressors|detector|detectors|segmenter|segmenters|"
        r"predictor|predictors|estimator|estimators|forecaster|forecasters|clusterer|clusterers|assigner|assigners|"
        r"query_strategy|discovery_method|ood_method|open_set_method|novelty_method|transfer_method|"
        r"domain_adaptation_method|test_time_adaptation_method|self_supervised_method|semi_supervised_method|"
        r"weak_supervision_method|multitask_method|pseudo_labeler|consistency_method|contrastive_method|"
        r"bagger|booster|voter|mixture_model|expert_mixture|data_splitter|batch_sampler|negative_sampler|hard_miner|"
        r"resampler|normalizer|normaliser|scaler|vectorizer|vectorizer|featurizer|featuriser|masking_strategy|"
        r"reward_function|advantage_estimator|exploration_strategy|action_selector|stopping_rule|"
        r"early_stopping_policy|checkpoint_policy|evaluation_protocol|validation_protocol|test_protocol|cv_scheme|"
        r"thresholder|decision_rule|exporter|export_format|serializer|compiler|inference_backend|precision_mode"
    )
    marker = r")(?:_|$)"
    if marker in old:
        old = old.replace(marker, "|" + extra + marker, 1)
    selectors._SELECTOR_TOKEN_RE = re.compile(old, re.I)


def _extend_component_regex() -> None:
    old = selectors._COMPONENT_DIR_RE.pattern
    extra = (
        r"networks?|classifiers?|regressors?|detectors?|segmenters?|predictors?|forecasters?|clusterers?|"
        r"open_world|ood|open_set|discovery|transfer|domain_adaptation|test_time_adaptation|self_supervised|"
        r"semi_supervised|weak_supervision|multitask|pseudo_labels?|contrastive|bagging|boosting|voting|mixtures?|"
        r"data_sampling|negative_sampling|mining|resampling|normalization|scaling|vectorizers?|featurizers?|"
        r"reward_functions?|advantage|exploration|stopping|checkpoints?|evaluation_protocols?|decision_rules?|"
        r"export|serialization|inference_backends?"
    )
    marker = r")(?:/|$)"
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
    tokens = scientific._SCIENTIFIC_TOKENS
    if _ADDED_TOKENS not in tokens:
        tokens = tokens + "|" + _ADDED_TOKENS
    scientific._SCIENTIFIC_TOKENS = tokens
    scientific.SCIENTIFIC_REGISTRY_RE = re.compile(r"(?:^|_)(?:" + tokens + r")(?:_|$)", re.I)
    scientific.SCIENTIFIC_IDENTITY_KEYS = set(scientific.SCIENTIFIC_IDENTITY_KEYS) | _IDENTITY_KEYS
    scientific.SCIENTIFIC_TRAINING_KEYS = set(scientific.SCIENTIFIC_TRAINING_KEYS) | {
        "stopping_rule", "early_stopping_policy", "checkpoint_policy", "batch_sampler", "negative_sampler",
        "hard_miner", "masking_strategy", "reward_function", "advantage_estimator",
    }
    scientific.SCIENTIFIC_ALL_FLAGS = set(scientific.SCIENTIFIC_ALL_FLAGS) | _ALL_FLAGS
    scientific.SCIENTIFIC_CONFIG_HINT_RE = re.compile(
        scientific.SCIENTIFIC_CONFIG_HINT_RE.pattern.rstrip(")")
        + r"|classifier|regressor|detector|segmenter|predictor|forecaster|cluster|query_strategy|discovery|ood|"
          r"open_set|transfer|domain_adaptation|test_time_adaptation|self_supervised|semi_supervised|"
          r"weak_supervision|multitask|pseudo_label|contrastive|bagging|boosting|voting|mixture|negative_sampler|"
          r"hard_miner|resampler|normalizer|scaler|vectorizer|featurizer|masking|reward_function|advantage|"
          r"exploration|stopping|checkpoint_policy|evaluation_protocol|validation_protocol|test_protocol|cv_scheme|"
          r"decision_rule|export_format|inference_backend|precision_mode)",
        re.I,
    )

    workload.REGISTRY_RE = scientific.SCIENTIFIC_REGISTRY_RE
    workload.IDENTITY_KEYS = set(workload.IDENTITY_KEYS) | _IDENTITY_KEYS
    workload.TRAINING_KEYS = set(workload.TRAINING_KEYS) | {
        "stopping_rule", "early_stopping_policy", "checkpoint_policy", "batch_sampler", "negative_sampler",
        "hard_miner", "masking_strategy", "reward_function", "advantage_estimator",
    }
    workload.CONFIG_HINT_RE = scientific.SCIENTIFIC_CONFIG_HINT_RE
    registry_members.ALL_FLAGS = set(registry_members.ALL_FLAGS) | _ALL_FLAGS

    _extend_selector_regex()
    _extend_component_regex()
    combinations._KEY_ALIASES.update(_KEY_ALIASES)
    combinations._PRIMARY_DIMENSIONS.update(set(_KEY_ALIASES.values()))
    _rebuild_materializer_dimensions()


def install_contract() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root, profile, jobs):
        report = original_report(root, profile, jobs)
        controls = dict(report.get("strict_controls") or {})
        require = bool(profile.get("require_role_paradigm_protocol_accounting", profile.get("strict_coverage", True)))
        controls["require_role_paradigm_protocol_accounting"] = require
        report.update({
            "scientific_ontology_v33_schema": ONTOLOGY_V33_SCHEMA,
            "role_paradigm_protocol_dimensions": sorted(set(_KEY_ALIASES.values())),
            "role_paradigm_protocol_cartesian_products_invented": False,
            "strict_controls": controls,
        })
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["ONTOLOGY_V33_SCHEMA", "install_primitives", "install_contract"]
