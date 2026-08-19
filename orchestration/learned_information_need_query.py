"""Learned information-need query construction for bounded dynamic RAG.

The provider is request-scoped. It receives already-released request text explicitly, binds
that text to ``DynamicRuntimeSnapshot.request_sha256``, verifies that the loaded policy came
from the exact configured generator/tokenizer/training-input identity, and uses the trained
information-need selector only to choose source tokens. Retrieval and query release remain
server-owned downstream authorities.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from orchestration.dynamic_rag_runtime import DynamicRuntimeSnapshot, InformationNeedQueryProvider
from orchestration.reference_dynamic_features import GeneratorHiddenStateAdapter
from training.advanced_rag_config import DynamicConfiguredRun
from training.advanced_rag_run_binding import dynamic_training_input_identity
from training.advanced_rag_runtime_loading import LoadedDynamicPolicyArtifact
from training.local_artifact_loading import load_local_language_model, load_local_tokenizer

_MAX_CONTEXT_CHARS = 5_000_000
_MAX_QUERY_CHARS = 100_000


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("learned information-need query construction requires optional PyTorch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


@dataclass(frozen=True)
class LearnedInformationNeedQueryConfig:
    probability_threshold: float = 0.5
    minimum_selected_tokens: int = 4
    maximum_selected_tokens: int = 48
    maximum_query_characters: int = 4_000
    include_generated_context: bool = True

    def __post_init__(self) -> None:
        threshold = _finite(self.probability_threshold, "probability_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0,1]")
        object.__setattr__(self, "probability_threshold", threshold)
        for name, minimum, maximum in (
            ("minimum_selected_tokens", 1, 10_000),
            ("maximum_selected_tokens", 1, 100_000),
            ("maximum_query_characters", 1, _MAX_QUERY_CHARS),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is out of bounds")
        if self.minimum_selected_tokens > self.maximum_selected_tokens:
            raise ValueError("minimum_selected_tokens may not exceed maximum_selected_tokens")
        if not isinstance(self.include_generated_context, bool):
            raise ValueError("include_generated_context must be boolean")

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-learned-information-need-query-config/v1", **asdict(self)})


class LocalLearnedInformationNeedQueryProvider(InformationNeedQueryProvider):
    """Request-bound implementation of the dynamic runtime query-provider protocol."""
    def __init__(
        self,
        *,
        loaded_policy: LoadedDynamicPolicyArtifact,
        run_config: DynamicConfiguredRun,
        released_request_text: str,
        config: LearnedInformationNeedQueryConfig = LearnedInformationNeedQueryConfig(),
    ) -> None:
        if not isinstance(loaded_policy, LoadedDynamicPolicyArtifact):
            raise ValueError("loaded_policy must be LoadedDynamicPolicyArtifact")
        if not isinstance(run_config, DynamicConfiguredRun):
            raise ValueError("run_config must be DynamicConfiguredRun")
        if not isinstance(released_request_text, str) or not released_request_text or "\x00" in released_request_text or len(released_request_text) > _MAX_CONTEXT_CHARS:
            raise ValueError("released_request_text is invalid or oversized")
        if not isinstance(config, LearnedInformationNeedQueryConfig):
            raise ValueError("config must be LearnedInformationNeedQueryConfig")
        manifest = loaded_policy.manifest
        expected_inputs = dynamic_training_input_identity(run_config)
        checks = {
            "kind": manifest.kind == "dynamic_rag_policy",
            "plan": manifest.plan_sha256 == run_config.plan.plan_sha256,
            "training_input": manifest.training_input_sha256 == expected_inputs.input_sha256,
            "generator": manifest.base_model_sha256 == run_config.generator.expected_sha256,
            "retrieval_stack": manifest.retrieval_stack_sha256 == run_config.plan.retrieval_stack_sha256,
            "budget": manifest.budget_sha256 == run_config.plan.budget.budget_sha256,
        }
        failures = [name for name, matched in checks.items() if not matched]
        if failures:
            raise ValueError(f"dynamic artifact differs from query-provider run config: {','.join(failures)}")
        self.loaded_policy = loaded_policy
        self.run_config = run_config
        self.released_request_text = released_request_text
        self.request_sha256 = _sha_text(released_request_text)
        self.config = config
        self.generator = load_local_language_model(run_config.generator)
        self.tokenizer = load_local_tokenizer(run_config.tokenizer)
        self.hidden_adapter = GeneratorHiddenStateAdapter(
            self.generator,
            self.tokenizer,
            generator_sha256=run_config.generator.expected_sha256,
            tokenizer_sha256=run_config.tokenizer.expected_sha256,
            generator_family=run_config.generator.artifact_kind,
            max_length=run_config.collator.context_max_length,
        )

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-local-learned-information-need-query-provider/v1",
                "dynamic_artifact_sha256": self.loaded_policy.manifest.artifact_sha256,
                "training_input_sha256": self.loaded_policy.manifest.training_input_sha256,
                "generator_sha256": self.run_config.generator.expected_sha256,
                "tokenizer_sha256": self.run_config.tokenizer.expected_sha256,
                "generator_family": self.run_config.generator.artifact_kind,
                "hidden_adapter_contract_sha256": self.hidden_adapter.contract_sha256,
                "request_sha256": self.request_sha256,
                "selection_config_sha256": self.config.config_sha256,
            }
        )

    def _context(self, snapshot: DynamicRuntimeSnapshot) -> str:
        if snapshot.request_sha256 != self.request_sha256:
            raise ValueError("dynamic snapshot request identity differs from request-bound query provider")
        if self.config.include_generated_context and snapshot.generated_text:
            value = self.released_request_text + "\n\n" + snapshot.generated_text
        else:
            value = self.released_request_text
        if len(value) > _MAX_CONTEXT_CHARS:
            raise ValueError("information-need context exceeds character safety bound")
        return value

    def build_query(self, snapshot: DynamicRuntimeSnapshot) -> str:
        _require_torch()
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        context = self._context(snapshot)
        hidden = self.hidden_adapter.encode([context])
        token_hidden = hidden["token_hidden"]
        state_hidden = hidden["state_hidden"]
        attention_mask = hidden["attention_mask"].to(dtype=torch.bool)
        architecture = self.loaded_policy.model.config
        if token_hidden.size(-1) != architecture.context_hidden_size:
            raise ValueError("upstream generator hidden width differs from trained dynamic-policy architecture")
        try:
            device = next(self.loaded_policy.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.loaded_policy.model.eval()
        with torch.no_grad():
            logits = self.loaded_policy.model.need_selector(
                token_hidden.to(device), state_hidden.to(device), attention_mask.to(device)
            )[0].detach().float().cpu()
        encoded = self.tokenizer(
            context,
            padding=False,
            truncation=True,
            max_length=self.run_config.collator.context_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"][0]
        special = encoded.get("special_tokens_mask")
        special_mask = special[0].to(dtype=torch.bool) if special is not None else torch.zeros_like(input_ids, dtype=torch.bool)
        visible = encoded.get("attention_mask")
        visible_mask = visible[0].to(dtype=torch.bool) if visible is not None else torch.ones_like(input_ids, dtype=torch.bool)
        if logits.numel() != input_ids.numel():
            raise ValueError("query tokenization differs from hidden-state tokenization")
        selectable = visible_mask & ~special_mask
        indices = torch.nonzero(selectable, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("information-need context contains no selectable tokens")
        probabilities = torch.sigmoid(logits)
        thresholded = [int(index) for index in indices if float(probabilities[index]) >= self.config.probability_threshold]
        ranked = sorted((int(index) for index in indices), key=lambda index: (-float(probabilities[index]), index))
        selected = thresholded
        if len(selected) < self.config.minimum_selected_tokens:
            selected = ranked[: self.config.minimum_selected_tokens]
        if len(selected) > self.config.maximum_selected_tokens:
            selected = sorted(selected, key=lambda index: (-float(probabilities[index]), index))[: self.config.maximum_selected_tokens]
        selected = sorted(set(selected))
        if not selected:
            raise ValueError("information-need selector produced no usable query tokens")

        groups: list[list[int]] = []
        for index in selected:
            if not groups or index != groups[-1][-1] + 1:
                groups.append([index])
            else:
                groups[-1].append(index)
        pieces = []
        for group in groups:
            piece = self.tokenizer.decode(input_ids[group].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            if piece:
                pieces.append(piece)
        query = " ".join(pieces).strip()
        if not query:
            raise ValueError("selected information-need tokens decoded to empty text")
        if len(query) > self.config.maximum_query_characters:
            query = query[: self.config.maximum_query_characters].rstrip()
        if not query or "\x00" in query:
            raise ValueError("learned information-need query is invalid")
        return query


__all__ = ["LearnedInformationNeedQueryConfig", "LocalLearnedInformationNeedQueryProvider"]
