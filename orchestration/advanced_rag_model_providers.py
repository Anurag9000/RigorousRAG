"""Serving adapters for loaded advanced-RAG artifacts.

These adapters deliberately expose only the repository's existing narrow provider protocols.
The grounded adapter performs bounded local language-model generation; it does not assign
server citation ids, call tools/retrievers, or publish output. The dynamic adapter only
scores the closed action vocabulary. Existing authoritative orchestration retains all release,
retrieval, evidence, schema, fencing and publication authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from orchestration.dynamic_rag_runtime import DynamicPolicyProvider
from security.model_output_authority import ClosedOutputSchema
from tools.trusted_generation_context import ChatMessage
from training.advanced_rag_runtime_loading import LoadedDynamicPolicyArtifact, LoadedGroundedArtifact
from training.dynamic_retrieval_policy import DynamicRetrievalAction, DynamicRetrievalFeatures

_MAX_OUTPUT_CHARS = 10_000_000


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("advanced RAG model providers require optional PyTorch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _positive_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


@dataclass(frozen=True)
class LocalGroundedGenerationConfig:
    max_input_tokens: int = 8192
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_input_tokens", _positive_int(self.max_input_tokens, "max_input_tokens", 10_000_000))
        object.__setattr__(self, "max_new_tokens", _positive_int(self.max_new_tokens, "max_new_tokens", 1_000_000))
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be boolean")
        temperature = _finite(self.temperature, "temperature")
        top_p = _finite(self.top_p, "top_p")
        penalty = _finite(self.repetition_penalty, "repetition_penalty")
        if temperature <= 0.0 or not 0.0 < top_p <= 1.0 or penalty <= 0.0:
            raise ValueError("generation temperature/top_p/repetition_penalty are out of bounds")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "repetition_penalty", penalty)

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-local-grounded-generation-config/v1", **asdict(self)})


class LocalGroundedGeneratorProvider:
    """Bounded local implementation of ``GroundedGeneratorProvider``."""
    def __init__(self, loaded: LoadedGroundedArtifact, config: LocalGroundedGenerationConfig = LocalGroundedGenerationConfig()) -> None:
        if not isinstance(loaded, LoadedGroundedArtifact):
            raise ValueError("loaded must be LoadedGroundedArtifact")
        if not isinstance(config, LocalGroundedGenerationConfig):
            raise ValueError("config must be LocalGroundedGenerationConfig")
        self.loaded, self.config = loaded, config

    @property
    def artifact_sha256(self) -> str:
        return self.loaded.manifest.artifact_sha256

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-local-grounded-generator-provider/v1",
                "artifact_sha256": self.artifact_sha256,
                "generator_family": self.loaded.manifest.generator_family,
                "tokenizer_sha256": self.loaded.manifest.tokenizer_sha256,
                "generation_config_sha256": self.config.config_sha256,
            }
        )

    def _render(self, messages: Sequence[ChatMessage]) -> str:
        if not messages:
            raise ValueError("grounded generation requires released messages")
        tokenizer = self.loaded.tokenizer
        records = []
        for message in messages:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("released messages must expose string role/content")
            records.append({"role": role, "content": content})
        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            return str(template(records, tokenize=False, add_generation_prompt=True))
        return "\n".join(f"<{item['role']}>\n{item['content']}" for item in records) + "\n<assistant>\n"

    def generate(self, messages: Sequence[ChatMessage], *, request_sha256: str, output_schema_sha256: str) -> str:
        _require_torch()
        _sha(request_sha256, "request_sha256"); _sha(output_schema_sha256, "output_schema_sha256")
        prompt = self._render(messages)
        tokenizer = self.loaded.tokenizer
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=False)
        input_ids = encoded.get("input_ids")
        if input_ids is None or input_ids.ndim != 2 or input_ids.size(0) != 1:
            raise ValueError("tokenizer must return one [1,T] input_ids row")
        if input_ids.size(1) > self.config.max_input_tokens:
            raise ValueError("released generation context exceeds configured model input budget")
        base_model = self.loaded.model.base_model
        try:
            device = next(base_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        model_inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
        generation = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if self.config.do_sample:
            generation["temperature"] = self.config.temperature
            generation["top_p"] = self.config.top_p
        base_model.eval()
        with torch.no_grad():
            output = base_model.generate(**model_inputs, **generation)
        if not torch.is_tensor(output) or output.ndim != 2 or output.size(0) != 1:
            raise ValueError("local language model generate() returned an invalid token tensor")
        tokens = output[0]
        if self.loaded.manifest.generator_family == "causal_lm":
            tokens = tokens[input_ids.size(1):]
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not isinstance(text, str) or not text.strip() or len(text) > _MAX_OUTPUT_CHARS or "\x00" in text:
            raise ValueError("generated output is empty, oversized or contains NUL")
        return text


class LocalDynamicPolicyProvider(DynamicPolicyProvider):
    """Inference adapter exposing only closed dynamic-retrieval action scores."""
    def __init__(self, loaded: LoadedDynamicPolicyArtifact) -> None:
        if not isinstance(loaded, LoadedDynamicPolicyArtifact):
            raise ValueError("loaded must be LoadedDynamicPolicyArtifact")
        self.loaded = loaded

    @property
    def artifact_sha256(self) -> str:
        return self.loaded.manifest.artifact_sha256

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-local-dynamic-policy-provider/v1",
                "artifact_sha256": self.artifact_sha256,
                "architecture_sha256": self.loaded.manifest.architecture_sha256,
                "budget_sha256": self.loaded.manifest.budget_sha256,
                "score_semantics": "raw_action_logits",
            }
        )

    def action_scores(self, features: DynamicRetrievalFeatures, *, snapshot_sha256: str) -> Mapping[DynamicRetrievalAction, float]:
        _require_torch(); _sha(snapshot_sha256, "snapshot_sha256")
        if not isinstance(features, DynamicRetrievalFeatures):
            raise ValueError("features must be DynamicRetrievalFeatures")
        architecture = self.loaded.model.config
        vector = features.vector(architecture.feature_names)
        try:
            device = next(self.loaded.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        tensor = torch.tensor([vector], dtype=torch.float32, device=device)
        self.loaded.model.eval()
        with torch.no_grad():
            output = self.loaded.model(features=tensor)
        logits = output.get("action_logits") if isinstance(output, Mapping) else None
        if logits is None or logits.ndim != 2 or logits.shape != (1, len(architecture.actions)):
            raise ValueError("dynamic policy model returned invalid action logits")
        values = logits[0].detach().float().cpu().tolist()
        return {action: float(values[index]) for index, action in enumerate(architecture.actions)}


__all__ = [
    "LocalDynamicPolicyProvider",
    "LocalGroundedGenerationConfig",
    "LocalGroundedGeneratorProvider",
]
