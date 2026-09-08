#!/usr/bin/env python3
"""Universal controller v32: full repository scientific-choice closure.

v25-v31 already account for the ordinary model/dataset/task/loss/optimizer/
pipeline surface and for architecture, adaptation, compression, retrieval,
generative/RL, continual-learning, diffusion, graph and data-protocol choices.
This final vocabulary layer closes additional scientifically material selectors
that commonly live under different names and would otherwise evade a literal
model/method/loss scan:

* regularizers, weighting/balancing and gradient transforms/clipping;
* Bayesian/probabilistic priors, posteriors, likelihoods, distributions and
  variational/MCMC inference families;
* federated aggregation/client-selection and meta-learning inner/outer loops;
* hyperparameter search spaces, tuners, sweeps, ablations, seeds/folds/CV;
* loaders, collators, batchers, materializers, downloaders and provenance;
* decoding/search/generation policies and model/checkpoint-selection policies;
* kernels, distances, similarities, projections, bottlenecks and latent spaces;
* corruptions, perturbations, imputers and robustness/noise models;
* modalities and modality-specific encoders/decoders/fusion selectors;
* teacher/student/EMA/target-network and objective-weighting components.

The layer is source accounting only.  It never manufactures arbitrary
model x dataset x task Cartesian products, never changes repository scientific
semantics, and contains no resource scheduling.  Compatible combinations remain
repository-authored.  Every ready job still runs exclusively through the exact
byte-pinned OPF_ADP scheduler inherited by the controller stack.
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

ONTOLOGY_V32_SCHEMA = 1

_ADDED_TOKENS = (
    "REGULARIZER|REGULARIZERS|REGULARISER|REGULARISERS|REGULARIZATION|REGULARISATION|"
    "WEIGHTING|WEIGHTINGS|REWEIGHTING|BALANCER|BALANCERS|BALANCING|"
    "GRADIENT_TRANSFORM|GRADIENT_TRANSFORMS|GRADIENT_CLIPPER|GRADIENT_CLIPPERS|CLIPPER|CLIPPERS|"
    "PRIOR|PRIORS|POSTERIOR|POSTERIORS|LIKELIHOOD|LIKELIHOODS|DISTRIBUTION|DISTRIBUTIONS|"
    "VARIATIONAL_FAMILY|VARIATIONAL_FAMILIES|MCMC_KERNEL|MCMC_KERNELS|PROBABILISTIC_MODEL|PROBABILISTIC_MODELS|"
    "FEDERATED_AGGREGATOR|FEDERATED_AGGREGATORS|CLIENT_SELECTOR|CLIENT_SELECTORS|CLIENT_SAMPLER|CLIENT_SAMPLERS|"
    "META_LEARNER|META_LEARNERS|INNER_LOOP|INNER_LOOPS|OUTER_LOOP|OUTER_LOOPS|"
    "SEARCH_SPACE|SEARCH_SPACES|TUNER|TUNERS|SWEEP|SWEEPS|ABLATION|ABLATIONS|SEED|SEEDS|FOLD|FOLDS|"
    "CROSS_VALIDATION|CROSS_VALIDATIONS|"
    "DATALOADER|DATALOADERS|DATA_LOADER|DATA_LOADERS|COLLATOR|COLLATORS|BATCHER|BATCHERS|"
    "MATERIALIZER|MATERIALIZERS|MATERIALISER|MATERIALISERS|DOWNLOADER|DOWNLOADERS|PROVENANCE|PROVENANCES|"
    "DECODING|DECODER_POLICY|DECODER_POLICIES|GENERATION_POLICY|GENERATION_POLICIES|SEARCH_POLICY|SEARCH_POLICIES|"
    "MODEL_SELECTOR|MODEL_SELECTORS|CHECKPOINT_SELECTOR|CHECKPOINT_SELECTORS|SELECTION_POLICY|SELECTION_POLICIES|"
    "KERNEL|KERNELS|DISTANCE|DISTANCES|SIMILARITY|SIMILARITIES|PROJECTION|PROJECTIONS|"
    "BOTTLENECK|BOTTLENECKS|LATENT|LATENTS|LATENT_SPACE|LATENT_SPACES|"
    "CORRUPTION|CORRUPTIONS|PERTURBATION|PERTURBATIONS|IMPUTER|IMPUTERS|ROBUSTNESS|ROBUSTNESS_METHOD|ROBUSTNESS_METHODS|"
    "NOISE_MODEL|NOISE_MODELS|MODALITY|MODALITIES|MODALITY_ENCODER|MODALITY_ENCODERS|MODALITY_DECODER|MODALITY_DECODERS|"
    "TEACHER|TEACHERS|STUDENT|STUDENTS|EMA_MODEL|EMA_MODELS|TARGET_NETWORK|TARGET_NETWORKS|"
    "LOSS_WEIGHT|LOSS_WEIGHTS|OBJECTIVE_WEIGHT|OBJECTIVE_WEIGHTS|TEMPERATURE_POLICY|TEMPERATURE_POLICIES"
)

_IDENTITY_KEYS = {
    "regularizer", "regulariser", "regularization", "regularisation", "weighting", "reweighting", "balancer", "balancing",
    "gradient_transform", "gradient_clipper", "clipper",
    "prior", "posterior", "likelihood", "distribution", "variational_family", "mcmc_kernel", "probabilistic_model",
    "federated_aggregator", "client_selector", "client_sampler", "meta_learner", "inner_loop", "outer_loop",
    "search_space", "tuner", "sweep", "ablation", "seed", "fold", "cross_validation",
    "dataloader", "data_loader", "collator", "batcher", "materializer", "materialiser", "downloader", "provenance",
    "decoding", "decoder_policy", "generation_policy", "search_policy", "model_selector", "checkpoint_selector", "selection_policy",
    "kernel", "distance", "similarity", "projection", "bottleneck", "latent", "latent_space",
    "corruption", "perturbation", "imputer", "robustness", "robustness_method", "noise_model",
    "modality", "modality_encoder", "modality_decoder", "teacher", "student", "ema_model", "target_network",
    "loss_weight", "objective_weight", "temperature_policy",
}

_KEY_ALIASES: Dict[str, str] = {
    **{key: "regularization" for key in ("regularizer", "regulariser", "regularization", "regularisation", "weighting", "reweighting", "balancer", "balancing", "loss_weight", "objective_weight")},
    **{key: "gradient_control" for key in ("gradient_transform", "gradient_clipper", "clipper")},
    **{key: "probabilistic_component" for key in ("prior", "posterior", "likelihood", "distribution", "variational_family", "mcmc_kernel", "probabilistic_model")},
    **{key: "federated_component" for key in ("federated_aggregator", "client_selector", "client_sampler")},
    **{key: "meta_learning_component" for key in ("meta_learner", "inner_loop", "outer_loop")},
    **{key: "search_component" for key in ("search_space", "tuner", "sweep", "ablation", "seed", "fold", "cross_validation")},
    **{key: "data_pipeline_component" for key in ("dataloader", "data_loader", "collator", "batcher", "materializer", "materialiser", "downloader", "provenance")},
    **{key: "decoding_component" for key in ("decoding", "decoder_policy", "generation_policy", "search_policy", "temperature_policy")},
    **{key: "selection_component" for key in ("model_selector", "checkpoint_selector", "selection_policy")},
    **{key: "kernel_component" for key in ("kernel", "distance", "similarity")},
    **{key: "representation_component" for key in ("projection", "bottleneck", "latent", "latent_space")},
    **{key: "robustness_component" for key in ("corruption", "perturbation", "imputer", "robustness", "robustness_method", "noise_model")},
    **{key: "modality_component" for key in ("modality", "modality_encoder", "modality_decoder")},
    **{key: "teacher_student_component" for key in ("teacher", "student", "ema_model", "target_network")},
}

_FLAG_DIMENSIONS: Dict[str, str] = {}
for _key, _dimension in _KEY_ALIASES.items():
    _FLAG_DIMENSIONS["--" + _key.replace("_", "-")] = _dimension
    _FLAG_DIMENSIONS["--" + _key] = _dimension

_ALL_FLAGS = {
    "--all-regularizers", "--all-regularisers", "--all-gradient-transforms", "--all-priors", "--all-posteriors",
    "--all-likelihoods", "--all-distributions", "--all-variational-families", "--all-mcmc-kernels",
    "--all-federated-aggregators", "--all-client-selectors", "--all-meta-learners", "--all-search-spaces",
    "--all-tuners", "--all-sweeps", "--all-ablations", "--all-seeds", "--all-folds",
    "--all-dataloaders", "--all-collators", "--all-materializers", "--all-downloaders",
    "--all-decoder-policies", "--all-generation-policies", "--all-model-selectors", "--all-checkpoint-selectors",
    "--all-kernels", "--all-distances", "--all-similarities", "--all-projections", "--all-bottlenecks",
    "--all-latent-spaces", "--all-corruptions", "--all-perturbations", "--all-imputers", "--all-noise-models",
    "--all-modalities", "--all-teachers", "--all-students", "--all-target-networks",
}


def _extend_selector_regex() -> None:
    old = selectors._SELECTOR_TOKEN_RE.pattern
    extra = (
        r"regularizer|regulariser|regularization|regularisation|weighting|reweighting|balancer|balancing|"
        r"gradient_transform|gradient_clipper|clipper|prior|posterior|likelihood|distribution|variational_family|mcmc_kernel|"
        r"probabilistic_model|federated_aggregator|client_selector|client_sampler|meta_learner|inner_loop|outer_loop|"
        r"search_space|tuner|sweep|ablation|seed|fold|cross_validation|dataloader|data_loader|collator|batcher|"
        r"materializer|materialiser|downloader|provenance|decoding|decoder_policy|generation_policy|search_policy|"
        r"model_selector|checkpoint_selector|selection_policy|kernel|distance|similarity|projection|bottleneck|latent|latent_space|"
        r"corruption|perturbation|imputer|robustness|robustness_method|noise_model|modality|modality_encoder|modality_decoder|"
        r"teacher|student|ema_model|target_network|loss_weight|objective_weight|temperature_policy"
    )
    marker = r")(?:_|$)"
    if marker in old:
        old = old.replace(marker, "|" + extra + marker, 1)
    selectors._SELECTOR_TOKEN_RE = re.compile(old, re.I)


def _extend_component_regex() -> None:
    old = selectors._COMPONENT_DIR_RE.pattern
    extra = (
        r"regularizers?|regularisation|regularization|gradients?|bayesian|probabilistic|priors?|posteriors?|likelihoods?|"
        r"federated|meta_learning|search_spaces?|sweeps?|ablations?|dataloaders?|collators?|materializers?|downloaders?|"
        r"decoding|generation|selection|kernels?|distances?|similarities?|projections?|bottlenecks?|latents?|"
        r"corruptions?|perturbations?|imputers?|robustness|modalities|teachers?|students?|target_networks?"
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
        "regularizer", "regularization", "regularisation", "gradient_transform", "gradient_clipper",
        "prior", "posterior", "likelihood", "distribution", "variational_family", "mcmc_kernel",
        "loss_weight", "objective_weight", "fold", "cross_validation",
    }
    scientific.SCIENTIFIC_ALL_FLAGS = set(scientific.SCIENTIFIC_ALL_FLAGS) | _ALL_FLAGS
    scientific.SCIENTIFIC_CONFIG_HINT_RE = re.compile(
        scientific.SCIENTIFIC_CONFIG_HINT_RE.pattern.rstrip(")")
        + r"|regulariz|reweight|gradient|prior|posterior|likelihood|distribution|variational|mcmc|federated|client|"
          r"meta_learner|inner_loop|outer_loop|search_space|tuner|sweep|ablation|seed|fold|cross_validation|"
          r"dataloader|collator|batcher|materializ|download|provenance|decod|generation|selection|kernel|distance|"
          r"similarity|projection|bottleneck|latent|corruption|perturbation|imputer|robust|noise_model|modality|"
          r"teacher|student|target_network|loss_weight|objective_weight)",
        re.I,
    )

    workload.REGISTRY_RE = scientific.SCIENTIFIC_REGISTRY_RE
    workload.IDENTITY_KEYS = set(workload.IDENTITY_KEYS) | _IDENTITY_KEYS
    workload.TRAINING_KEYS = set(workload.TRAINING_KEYS) | {
        "regularizer", "regularization", "regularisation", "gradient_transform", "gradient_clipper",
        "prior", "posterior", "likelihood", "distribution", "variational_family", "mcmc_kernel",
        "loss_weight", "objective_weight", "fold", "cross_validation",
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
        require = bool(profile.get("require_full_scientific_choice_accounting", profile.get("strict_coverage", True)))
        controls["require_full_scientific_choice_accounting"] = require
        report.update({
            "scientific_ontology_v32_schema": ONTOLOGY_V32_SCHEMA,
            "full_scientific_choice_dimensions": sorted(set(_KEY_ALIASES.values())),
            "full_scientific_choice_cartesian_products_invented": False,
            "strict_controls": controls,
        })
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["ONTOLOGY_V32_SCHEMA", "install_primitives", "install_contract"]
