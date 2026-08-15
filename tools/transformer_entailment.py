"""Injected transformer NLI adapter for the claim-entailment gate.

The adapter receives an already-loaded tokenizer/model and a reviewed label mapping. It
never downloads or trains a model. Unsupported label layouts fail closed rather than
assuming that a particular integer index means entailment.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from tools.claim_entailment import EntailmentProvider, EntailmentScore


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _row(value: Any) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("NLI output must contain one batch item")
        value = value[0]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or not 2 <= len(value) <= 64:
        raise ValueError("NLI output has an invalid class dimension")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("NLI logits contain a non-numeric value")
        parsed = float(item)
        if not math.isfinite(parsed):
            raise ValueError("NLI logits contain a non-finite value")
        output.append(parsed)
    return tuple(output)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    maximum = max(logits)
    values = [math.exp(item - maximum) for item in logits]
    total = sum(values)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("NLI softmax normalization failed")
    return tuple(item / total for item in values)


class InjectedTransformerEntailmentProvider(EntailmentProvider):
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        provider_version: str,
        label_by_index: Mapping[int, str],
        device: str = "cpu",
        max_length: int = 1024,
    ) -> None:
        if tokenizer is None or model is None:
            raise ValueError("tokenizer and model must be supplied explicitly")
        self.tokenizer = tokenizer
        self.model = model
        self.provider_version = _text(provider_version, "provider_version", 128)
        self.device = _text(device, "device", 64)
        if isinstance(max_length, bool) or not isinstance(max_length, int) or not 16 <= max_length <= 32_768:
            raise ValueError("max_length is invalid")
        self.max_length = max_length
        if not isinstance(label_by_index, Mapping) or not 2 <= len(label_by_index) <= 64:
            raise ValueError("label_by_index must be a bounded mapping")
        normalized: dict[int, str] = {}
        for raw_index, raw_label in label_by_index.items():
            if isinstance(raw_index, bool):
                raise ValueError("NLI label index is invalid")
            index = int(raw_index)
            if index < 0:
                raise ValueError("NLI label index is invalid")
            label = _text(raw_label, "NLI label", 32).lower()
            aliases = {
                "entail": "entailment",
                "entailed": "entailment",
                "contradict": "contradiction",
                "contradicted": "contradiction",
                "unknown": "neutral",
            }
            label = aliases.get(label, label)
            if label not in {"entailment", "neutral", "contradiction"}:
                raise ValueError("NLI label mapping contains an unsupported label")
            normalized[index] = label
        if "entailment" not in normalized.values() or "contradiction" not in normalized.values():
            raise ValueError("NLI label mapping must explicitly identify entailment and contradiction")
        self.label_by_index = normalized

    def _batch_to_device(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {key: (value.to(self.device) if hasattr(value, "to") else value) for key, value in batch.items()}
        return batch

    def score(self, claim: str, evidence: str) -> EntailmentScore:
        hypothesis = _text(claim, "claim", 5000)
        premise = _text(evidence, "evidence", 20_000)
        batch = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = self._batch_to_device(batch)
        try:
            import torch  # type: ignore
        except Exception:
            torch = None
        if torch is not None:
            with torch.inference_mode():
                output = self.model(**batch) if isinstance(batch, Mapping) else self.model(batch)
        else:
            output = self.model(**batch) if isinstance(batch, Mapping) else self.model(batch)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, Mapping):
            logits = output.get("logits")
        if logits is None:
            raise RuntimeError("NLI model did not return logits")
        probabilities = _softmax(_row(logits))
        by_label = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for index, probability in enumerate(probabilities):
            label = self.label_by_index.get(index)
            if label is None:
                continue
            by_label[label] += probability
        label = max(by_label, key=lambda key: (by_label[key], key))
        return EntailmentScore(label, by_label[label], self.provider_version)


__all__ = ["InjectedTransformerEntailmentProvider"]
