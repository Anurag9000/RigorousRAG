#!/usr/bin/env python3
"""Universal controller v31: extended scientific component ontology closure.

v25-v30 close the common experiment surface and declarative selector forms.  A
real repository can still hide scientifically material choices behind component
names that are not literally called model/method/loss/task.  v31 expands the
same fail-closed accounting contract to those remaining strong component names:

* experts, routers/gates and modular network pieces;
* embeddings, attention, pooling, normalization, activations and initialization;
* adapters/PEFT/prompt variants and compression/quantization/pruning;
* distillation, calibration, uncertainty and explanation/attribution methods;
* retrieval, reranking, indexing, chunking and reader components;
* generator/discriminator/critic/reward/value/world-model/planner components;
* replay/memory, curriculum and acquisition strategies;
* diffusion denoisers/score models/noise schedules and graph aggregation/readout;
* data partition/split/stream/episode/trajectory protocols;
* post-processing, constraints and penalties.

This is an inventory/closure layer only.  It does not invent model x dataset x
task Cartesian products and contains no resource scheduling.  Compatibility is
still established by repository-authored jobs/configs/selectors, while all ready
jobs execute exclusively through the exact byte-pinned OPF_ADP scheduler.
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

ONTOLOGY_V31_SCHEMA = 1

_ADDED_TOKENS = (
    "NECK|NECKS|STEM|STEMS|EXPERT|EXPERTS|ROUTER|ROUTERS|GATE|GATES|BLOCK|BLOCKS|LAYER|LAYERS|"
    "EMBEDDING|EMBEDDINGS|EMBEDDER|EMBEDDERS|ATTENTION|ATTENTIONS|POOLING|POOLINGS|POOLER|POOLERS|"
    "NORMALIZATION|NORMALIZATIONS|NORMALISATION|NORMALISATIONS|NORM|NORMS|ACTIVATION|ACTIVATIONS|"
    "INITIALIZER|INITIALIZERS|INITIALISER|INITIALISERS|INITIALIZATION|INITIALISATION|"
    "ADAPTER|ADAPTERS|PEFT|LORA|PROMPT_METHOD|PROMPT_METHODS|PROMPT_TUNING|PREFIX_TUNING|"
    "QUANTIZER|QUANTIZERS|QUANTIZATION|QUANTISATION|PRUNER|PRUNERS|PRUNING|"
    "COMPRESSOR|COMPRESSORS|COMPRESSION|SPARSIFIER|SPARSIFIERS|SPARSIFICATION|"
    "DISTILLER|DISTILLERS|DISTILLATION|CALIBRATOR|CALIBRATORS|CALIBRATION|"
    "UNCERTAINTY|UNCERTAINTY_METHOD|UNCERTAINTY_METHODS|INFERENCE_METHOD|INFERENCE_METHODS|"
    "EXPLAINER|EXPLAINERS|EXPLANATION|EXPLANATIONS|ATTRIBUTION|ATTRIBUTIONS|"
    "RETRIEVER|RETRIEVERS|RERANKER|RERANKERS|INDEXER|INDEXERS|CHUNKER|CHUNKERS|READER|READERS|"
    "GENERATOR|GENERATORS|DISCRIMINATOR|DISCRIMINATORS|CRITIC|CRITICS|"
    "REWARD_MODEL|REWARD_MODELS|VALUE_MODEL|VALUE_MODELS|WORLD_MODEL|WORLD_MODELS|PLANNER|PLANNERS|"
    "REPLAY_BUFFER|REPLAY_BUFFERS|MEMORY_BUFFER|MEMORY_BUFFERS|CURRICULUM|CURRICULA|"
    "ACQUISITION|ACQUISITIONS|ACQUISITION_FUNCTION|ACQUISITION_FUNCTIONS|"
    "DENOISER|DENOISERS|SCORE_MODEL|SCORE_MODELS|NOISE_SCHEDULE|NOISE_SCHEDULES|"
    "MESSAGE_PASSING|MESSAGE_PASSINGS|AGGREGATOR|AGGREGATORS|READOUT|READOUTS|"
    "DATA_SPLIT|DATA_SPLITS|PARTITION|PARTITIONS|STREAM_PROTOCOL|STREAM_PROTOCOLS|"
    "EPISODE_PROTOCOL|EPISODE_PROTOCOLS|TRAJECTORY_SOURCE|TRAJECTORY_SOURCES|"
    "POSTPROCESSOR|POSTPROCESSORS|CONSTRAINT|CONSTRAINTS|PENALTY|PENALTIES"
)

_IDENTITY_KEYS = {
    "neck", "stem", "expert", "router", "gate", "block", "layer",
    "embedding", "embedder", "attention", "pooling", "pooler", "normalization", "normalisation", "norm",
    "activation", "initializer", "initialiser", "initialization", "initialisation",
    "adapter", "adapter_type", "peft", "peft_method", "lora", "prompt_method", "prompt_tuning", "prefix_tuning",
    "quantizer", "quantization", "quantisation", "pruner", "pruning", "compressor", "compression",
    "sparsifier", "sparsification", "distiller", "distillation", "calibrator", "calibration",
    "uncertainty", "uncertainty_method", "inference_method", "explainer", "explanation", "attribution",
    "retriever", "reranker", "indexer", "chunker", "reader",
    "generator", "discriminator", "critic", "reward_model", "value_model", "world_model", "planner",
    "replay_buffer", "memory_buffer", "curriculum", "acquisition", "acquisition_function",
    "denoiser", "score_model", "noise_schedule", "message_passing", "aggregator", "readout",
    "data_split", "partition", "stream_protocol", "episode_protocol", "trajectory_source",
    "postprocessor", "constraint", "penalty",
}

_KEY_ALIASES: Dict[str, str] = {
    **{key: "component" for key in ("neck", "stem", "expert", "router", "gate", "block", "layer")},
    **{key: "representation" for key in ("embedding", "embedder", "attention", "pooling", "pooler", "normalization", "normalisation", "norm", "activation", "initializer", "initialiser", "initialization", "initialisation")},
    **{key: "adaptation" for key in ("adapter", "adapter_type", "peft", "peft_method", "lora", "prompt_method", "prompt_tuning", "prefix_tuning")},
    **{key: "compression" for key in ("quantizer", "quantization", "quantisation", "pruner", "pruning", "compressor", "compression", "sparsifier", "sparsification")},
    **{key: "analysis_method" for key in ("distiller", "distillation", "calibrator", "calibration", "uncertainty", "uncertainty_method", "inference_method", "explainer", "explanation", "attribution")},
    **{key: "retrieval_component" for key in ("retriever", "reranker", "indexer", "chunker", "reader")},
    **{key: "generative_component" for key in ("generator", "discriminator", "critic", "reward_model", "value_model", "world_model", "planner", "denoiser", "score_model", "noise_schedule")},
    **{key: "continual_component" for key in ("replay_buffer", "memory_buffer", "curriculum", "acquisition", "acquisition_function")},
    **{key: "graph_component" for key in ("message_passing", "aggregator", "readout")},
    **{key: "data_protocol" for key in ("data_split", "partition", "stream_protocol", "episode_protocol", "trajectory_source")},
    **{key: "postprocess_component" for key in ("postprocessor", "constraint", "penalty")},
}

_FLAG_DIMENSIONS: Dict[str, str] = {}
for _key, _dim in _KEY_ALIASES.items():
    _FLAG_DIMENSIONS["--" + _key.replace("_", "-")] = _dim
    _FLAG_DIMENSIONS["--" + _key] = _dim

_ALL_FLAGS = {
    "--all-experts", "--all-routers", "--all-attentions", "--all-embeddings", "--all-adapters",
    "--all-peft", "--all-quantizers", "--all-pruners", "--all-distillers", "--all-calibrators",
    "--all-uncertainty-methods", "--all-explainers", "--all-retrievers", "--all-rerankers",
    "--all-indexers", "--all-generators", "--all-critics", "--all-world-models", "--all-planners",
    "--all-replay-buffers", "--all-curricula", "--all-acquisition-functions", "--all-denoisers",
    "--all-score-models", "--all-noise-schedules", "--all-aggregators", "--all-readouts",
    "--all-data-splits", "--all-partitions", "--all-postprocessors",
}


def _extend_selector_regex() -> None:
    old = selectors._SELECTOR_TOKEN_RE.pattern
    extra = (
        r"neck|necks|stem|stems|expert|experts|router|routers|gate|gates|block|blocks|layer|layers|"
        r"embedding|embeddings|embedder|embedders|attention|attentions|pooling|poolings|pooler|poolers|"
        r"normalization|normalisation|norm|norms|activation|activations|initializer|initialiser|"
        r"adapter|adapters|adapter_type|peft|peft_method|lora|prompt_method|prompt_tuning|prefix_tuning|"
        r"quantizer|quantizers|quantization|quantisation|pruner|pruners|pruning|compressor|compression|"
        r"sparsifier|sparsification|distiller|distillation|calibrator|calibration|uncertainty|uncertainty_method|"
        r"inference_method|explainer|explanation|attribution|retriever|retrievers|reranker|rerankers|indexer|"
        r"indexers|chunker|chunkers|reader|readers|generator|generators|discriminator|discriminators|critic|"
        r"critics|reward_model|value_model|world_model|planner|replay_buffer|memory_buffer|curriculum|"
        r"acquisition|acquisition_function|denoiser|score_model|noise_schedule|message_passing|aggregator|readout|"
        r"data_split|partition|stream_protocol|episode_protocol|trajectory_source|postprocessor|constraint|penalty"
    )
    marker = r")(?:_|$)"
    if marker in old:
        old = old.replace(marker, "|" + extra + marker, 1)
    selectors._SELECTOR_TOKEN_RE = re.compile(old, re.I)


def _extend_component_regex() -> None:
    old = selectors._COMPONENT_DIR_RE.pattern
    extra = (
        r"experts?|routers?|attention|embeddings?|adapters?|peft|quantization|pruning|compression|"
        r"distillation|calibration|uncertainty|explainers?|retrievers?|rerankers?|indexers?|generators?|"
        r"critics?|world_models?|planners?|replay_buffers?|curricula|acquisition|denoisers?|score_models?|"
        r"noise_schedules?|message_passing|aggregators?|readouts?|postprocessors?"
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
        "distillation", "quantization", "quantisation", "pruning", "compression", "calibration", "constraint", "penalty"
    }
    scientific.SCIENTIFIC_ALL_FLAGS = set(scientific.SCIENTIFIC_ALL_FLAGS) | _ALL_FLAGS
    scientific.SCIENTIFIC_CONFIG_HINT_RE = re.compile(
        scientific.SCIENTIFIC_CONFIG_HINT_RE.pattern.rstrip(")")
        + r"|expert|router|attention|embedding|adapter|peft|quant|prun|compress|distill|calibr|uncert|explain|"
          r"retriev|rerank|index|generator|critic|reward_model|value_model|world_model|planner|replay|curricul|"
          r"acquisition|denois|score_model|noise_schedule|message_pass|aggregat|readout|partition|postprocess|constraint|penalty)",
        re.I,
    )

    workload.REGISTRY_RE = scientific.SCIENTIFIC_REGISTRY_RE
    workload.IDENTITY_KEYS = set(workload.IDENTITY_KEYS) | _IDENTITY_KEYS
    workload.TRAINING_KEYS = set(workload.TRAINING_KEYS) | {
        "distillation", "quantization", "quantisation", "pruning", "compression", "calibration", "constraint", "penalty"
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
        require = bool(profile.get("require_extended_scientific_component_accounting", profile.get("strict_coverage", True)))
        controls["require_extended_scientific_component_accounting"] = require
        report.update({
            "scientific_ontology_v31_schema": ONTOLOGY_V31_SCHEMA,
            "extended_scientific_component_dimensions": sorted(set(_KEY_ALIASES.values())),
            "extended_scientific_component_cartesian_products_invented": False,
            "strict_controls": controls,
        })
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["ONTOLOGY_V31_SCHEMA", "install_primitives", "install_contract"]
