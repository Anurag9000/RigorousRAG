"""Concrete local-only supervision providers for advanced RAG materialization.

These adapters implement the provider protocols used by the cache/trajectory materializers.
They never download models or data. Callers inject already-admitted local model/tokenizer
objects; model execution occurs only when a materialization method is explicitly invoked.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from training.advanced_rag_data import DynamicRagEpisodeStep, GroundedGenerationExample
from training.dynamic_record_identity import dynamic_step_identity, dynamic_step_pair
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalAction


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("local supervision providers require optional PyTorch")


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("local supervision model exposes no parameters") from exc


def _move(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    _require_torch()
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _output_logits(output: Any) -> Any:
    logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
    if logits is None or getattr(logits, "ndim", 0) != 3:
        raise ValueError("language model must expose [B,T,V] logits")
    return logits


def _causal_inputs(tokenizer: Any, prompt: str, answer: str, *, max_length: int) -> tuple[dict[str, Any], Any, int]:
    prefix = prompt + "\n\n"
    encoded = dict(tokenizer(prefix + answer, truncation=True, max_length=max_length, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True))
    offsets = encoded.pop("offset_mapping")[0].tolist()
    positions = [index for index, pair in enumerate(offsets) if int(pair[1]) > int(pair[0]) and int(pair[1]) > len(prefix)]
    if answer and not positions:
        raise ValueError("answer is entirely truncated while materializing causal supervision")
    attention = encoded.get("attention_mask")
    valid = int(attention[0].long().sum().item()) if attention is not None else int(encoded["input_ids"].size(1))
    answer_start = min(positions) if positions else max(0, valid - 1)
    labels = torch.full_like(encoded["input_ids"], -100)
    for position in range(max(0, answer_start - 1), max(0, valid - 1)):
        labels[0, position] = encoded["input_ids"][0, position + 1]
    return encoded, labels, answer_start


def _seq2seq_inputs(tokenizer: Any, prompt: str, answer: str, *, max_length: int) -> tuple[dict[str, Any], Any]:
    encoder = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=True)
    target = tokenizer(answer, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=True)
    labels = target["input_ids"].clone()
    attention = target.get("attention_mask")
    if attention is not None:
        labels = labels.masked_fill(~attention.to(dtype=torch.bool), -100)
    return dict(encoder), labels


def _sequence_log_probability(logits: Any, labels: Any) -> float:
    _require_torch()
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits/labels sequence shapes differ")
    active = labels.ne(-100)
    if not bool(active.any().detach().item()):
        return 0.0
    safe = labels.long().masked_fill(~active, 0)
    selected = F.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return float(selected[active].sum().detach().cpu())


@dataclass(frozen=True)
class LocalLanguageModelSupervisionConfig:
    generator_family: str
    max_length: int = 2048
    normalize_document_utility_by_tokens: bool = True
    evidence_prefix: str = "\n\nEvidence:\n"
    answer_prefix: str = "\n\nAnswer:\n"

    def __post_init__(self) -> None:
        if self.generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError("max_length must be positive")
        for name in ("evidence_prefix", "answer_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ValueError(f"{name} must be non-empty text")

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-local-lm-supervision-config/v1", **asdict(self)})


class _LocalLmProviderBase:
    def __init__(self, model: Any, tokenizer: Any, *, model_sha256: str, tokenizer_sha256: str, config: LocalLanguageModelSupervisionConfig) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.model_sha256 = _sha(model_sha256, "model_sha256")
        self.tokenizer_sha256 = _sha(tokenizer_sha256, "tokenizer_sha256")

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": f"rigorousrag-{type(self).__name__}/v1", "model_sha256": self.model_sha256, "tokenizer_sha256": self.tokenizer_sha256, "config_sha256": self.config.config_sha256})

    def _logits_and_labels(self, prompt: str, answer: str) -> tuple[Any, Any]:
        _require_torch()
        device = _device(self.model)
        if self.config.generator_family == "causal_lm":
            inputs, labels, _ = _causal_inputs(self.tokenizer, prompt, answer, max_length=self.config.max_length)
            with torch.no_grad():
                logits = _output_logits(self.model(**_move(inputs, device), return_dict=True))
            return logits, labels.to(device)
        encoder, labels = _seq2seq_inputs(self.tokenizer, prompt, answer, max_length=self.config.max_length)
        with torch.no_grad():
            logits = _output_logits(self.model(**_move(encoder, device), labels=labels.to(device), return_dict=True))
        return logits, labels.to(device)


class LocalSequenceReferenceProvider(_LocalLmProviderBase):
    """Compute frozen reference-policy chosen/rejected sequence log probabilities."""
    def sequence_log_probabilities(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[tuple[float, float]]:
        result = []
        self.model.eval()
        for example in examples:
            if example.chosen_answer is None or example.rejected_answer is None:
                raise ValueError(f"example {example.example_id} lacks chosen/rejected answers")
            chosen_logits, chosen_labels = self._logits_and_labels(example.prompt, example.chosen_answer)
            rejected_logits, rejected_labels = self._logits_and_labels(example.prompt, example.rejected_answer)
            result.append((_sequence_log_probability(chosen_logits, chosen_labels), _sequence_log_probability(rejected_logits, rejected_labels)))
        return tuple(result)


class LocalTeacherLogitProvider(_LocalLmProviderBase):
    """Produce unpadded teacher logits aligned with the configured student tokenizer path."""
    def token_logits(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[Any]:
        _require_torch()
        result = []
        self.model.eval()
        device = _device(self.model)
        for example in examples:
            if self.config.generator_family == "causal_lm":
                inputs, _, _ = _causal_inputs(self.tokenizer, example.prompt, example.answer, max_length=self.config.max_length)
                with torch.no_grad():
                    logits = _output_logits(self.model(**_move(inputs, device), return_dict=True))
            else:
                encoder, labels = _seq2seq_inputs(self.tokenizer, example.prompt, example.answer, max_length=self.config.max_length)
                with torch.no_grad():
                    logits = _output_logits(self.model(**_move(encoder, device), labels=labels.to(device), return_dict=True))
            result.append(logits[0].detach().cpu().contiguous())
        return tuple(result)


class LocalDocumentUtilityProvider(_LocalLmProviderBase):
    """Score each evidence candidate by target-answer LM utility for retriever coupling."""
    def document_log_likelihoods(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[Any]:
        _require_torch()
        self.model.eval()
        results = []
        for example in examples:
            scores = []
            for evidence in example.evidence:
                conditioned_prompt = example.prompt + self.config.evidence_prefix + evidence.text + self.config.answer_prefix
                logits, labels = self._logits_and_labels(conditioned_prompt, example.answer)
                score = _sequence_log_probability(logits, labels)
                if self.config.normalize_document_utility_by_tokens:
                    count = int(labels.ne(-100).sum().detach().cpu().item())
                    if count > 0:
                        score /= count
                if not math.isfinite(score):
                    raise ValueError("document LM utility is non-finite")
                scores.append(score)
            results.append(torch.tensor(scores, dtype=torch.float32))
        return tuple(results)


class LocalDynamicPolicyValueProvider:
    """Read retrieval-value targets from a frozen local DynamicRagPolicyModel."""
    def __init__(self, model: Any, architecture: DynamicPolicyArchitecture, *, model_sha256: str) -> None:
        self.model, self.architecture = model, architecture
        self.model_sha256 = _sha(model_sha256, "model_sha256")

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-local-dynamic-value-provider/v1", "model_sha256": self.model_sha256, "architecture_sha256": self.architecture.architecture_sha256})

    def values(self, steps: Sequence[DynamicRagEpisodeStep]) -> Sequence[float]:
        _require_torch()
        if not steps:
            return ()
        features = torch.tensor([[float(step.features[name]) for name in self.architecture.feature_names] for step in steps], dtype=torch.float32, device=_device(self.model))
        self.model.eval()
        with torch.no_grad():
            output = self.model(features=features)
        value = output.get("retrieval_value") if isinstance(output, Mapping) else getattr(output, "retrieval_value", None)
        if value is None or value.ndim != 1 or value.numel() != len(steps):
            raise ValueError("dynamic value model must expose one retrieval_value per step")
        values = tuple(float(item) for item in value.detach().cpu().tolist())
        if any(not math.isfinite(item) for item in values):
            raise ValueError("dynamic value provider returned non-finite value")
        return values


def _counterfactual_key(value: Any) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        episode, step = dynamic_step_pair(value[0], value[1])
        return dynamic_step_identity(episode, step)
    if isinstance(value, str):
        selected = value.strip()
        prefix = "dynamic-step:"
        if selected.startswith(prefix):
            digest = selected[len(prefix):].lower()
            if len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest):
                return prefix + digest
    raise ValueError("counterfactual utility key must be an (episode_id, step_id) tuple or canonical dynamic-step:<sha256> identity")


class LoggedCounterfactualUtilityProvider:
    """Content-bound counterfactual utilities supplied by an admitted offline simulator/log.

    Delimiter-concatenated ``episode:step`` keys are intentionally rejected. Callers must use
    exact two-tuples or canonical ``dynamic-step:<sha256>`` identities so distinct legal IDs
    cannot alias one another.
    """
    def __init__(self, utilities: Mapping[Any, Mapping[str, float]], *, source_sha256: str) -> None:
        self.source_sha256 = _sha(source_sha256, "source_sha256")
        normalized: dict[str, dict[DynamicRetrievalAction, float]] = {}
        for raw_key, mapping in utilities.items():
            key = _counterfactual_key(raw_key)
            if key in normalized or not isinstance(mapping, Mapping) or not mapping:
                raise ValueError("counterfactual utility mapping is invalid or duplicated")
            actions: dict[DynamicRetrievalAction, float] = {}
            for raw_action, raw_value in mapping.items():
                action = DynamicRetrievalAction(raw_action)
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError("counterfactual utility must be finite")
                if action in actions:
                    raise ValueError("counterfactual utility mapping repeats an action")
                actions[action] = value
            normalized[key] = actions
        self.utilities = normalized

    @property
    def contract_sha256(self) -> str:
        serializable = {key: {action.value: value for action, value in sorted(mapping.items(), key=lambda item: item[0].value)} for key, mapping in sorted(self.utilities.items())}
        return _digest({"schema": "rigorousrag-logged-counterfactual-utility-provider/v2", "source_sha256": self.source_sha256, "identity_semantics": "canonical_dynamic_step_sha256", "utilities": serializable})

    def action_utilities(self, step: DynamicRagEpisodeStep) -> Mapping[DynamicRetrievalAction, float]:
        key = dynamic_step_identity(step.episode_id, step.step_id)
        if key not in self.utilities:
            raise ValueError(f"counterfactual utility source lacks canonical step {key}")
        return dict(self.utilities[key])


__all__ = [
    "LocalDocumentUtilityProvider",
    "LocalDynamicPolicyValueProvider",
    "LocalLanguageModelSupervisionConfig",
    "LocalSequenceReferenceProvider",
    "LocalTeacherLogitProvider",
    "LoggedCounterfactualUtilityProvider",
]
