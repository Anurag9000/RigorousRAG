"""Concrete local-only generator hidden-state provider for dynamic RAG supervision.

The dynamic canonical-data authority intentionally depends on the narrow
``BoundGeneratorHiddenStateProvider`` protocol.  This module supplies the repository-owned
implementation for already-admitted local Hugging Face causal/seq2seq language models and
matching tokenizers.  It never downloads a model, never trusts remote code and does not execute
anything on import; callers explicitly load admitted local artifacts and invoke ``encode``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional execution dependency.
    torch = None  # type: ignore[assignment]


_HEX = frozenset("0123456789abcdef")
_MAX_LENGTH = 10_000_000
_MAX_BATCH = 4096


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("local dynamic hidden-state materialization requires optional PyTorch")


def _device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("local generator exposes no parameters") from exc


def _move(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    _require_torch()
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _hidden(output: Any) -> Any:
    """Return a final hidden sequence as [B,T,H] from a model/encoder output."""
    _require_torch()
    value = (
        output.get("last_hidden_state")
        if isinstance(output, Mapping)
        else getattr(output, "last_hidden_state", None)
    )
    if value is None:
        hidden_states = (
            output.get("hidden_states")
            if isinstance(output, Mapping)
            else getattr(output, "hidden_states", None)
        )
        if hidden_states:
            value = hidden_states[-1]
    if value is None:
        value = (
            output.get("encoder_last_hidden_state")
            if isinstance(output, Mapping)
            else getattr(output, "encoder_last_hidden_state", None)
        )
    if value is None or not torch.is_tensor(value) or value.ndim != 3:
        raise ValueError("local generator must expose a final [B,T,H] hidden sequence")
    return value


@dataclass(frozen=True)
class LocalDynamicHiddenStateConfig:
    generator_family: str
    max_length: int = 2048
    pooling: str = "last_visible"
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        if self.generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")
        if (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or not 1 <= self.max_length <= _MAX_LENGTH
        ):
            raise ValueError("max_length must be a positive bounded integer")
        if self.pooling not in {"last_visible", "mean_visible"}:
            raise ValueError("pooling must be last_visible or mean_visible")
        if self.pad_to_multiple_of is not None:
            value = self.pad_to_multiple_of
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("pad_to_multiple_of must be a positive integer or None")

    @property
    def config_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-local-dynamic-hidden-state-config/v1",
                **asdict(self),
            }
        )


class LocalGeneratorHiddenStateProvider:
    """Bound local generator/tokenizer implementation of hidden-state supervision.

    ``encode`` accepts a bounded batch of exact context strings and returns:
    ``token_hidden`` [B,T,H], ``state_hidden`` [B,H], and ``attention_mask`` [B,T].
    Encoder-decoder models are evaluated through their encoder directly so no synthetic decoder
    inputs are required.  The canonical v2 materializer currently calls this with batch size one,
    while the batch contract keeps the provider reusable for later batched materialization.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        generator_sha256: str,
        tokenizer_sha256: str,
        config: LocalDynamicHiddenStateConfig,
    ) -> None:
        if not isinstance(config, LocalDynamicHiddenStateConfig):
            raise ValueError("config must be LocalDynamicHiddenStateConfig")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.generator_sha256 = _sha(generator_sha256, "generator_sha256")
        self.tokenizer_sha256 = _sha(tokenizer_sha256, "tokenizer_sha256")

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-local-generator-hidden-state-provider/v1",
                "generator_sha256": self.generator_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "config_sha256": self.config.config_sha256,
                "output_contract": "token_hidden[B,T,H]+state_hidden[B,H]+attention_mask[B,T]",
            }
        )

    def encode(self, texts: Sequence[str]) -> Mapping[str, Any]:
        _require_torch()
        selected = tuple(texts)
        if not selected or len(selected) > _MAX_BATCH:
            raise ValueError(f"hidden-state provider requires 1..{_MAX_BATCH} texts")
        if any(not isinstance(text, str) or not text or "\x00" in text for text in selected):
            raise ValueError("hidden-state provider texts must be non-empty NUL-free strings")

        encoded = self.tokenizer(
            list(selected),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("local tokenizer must return input_ids and attention_mask")
        attention = encoded["attention_mask"]
        if not torch.is_tensor(attention) or attention.ndim != 2:
            raise ValueError("local tokenizer attention_mask must have shape [B,T]")
        if attention.size(0) != len(selected):
            raise ValueError("local tokenizer batch dimension differs from requested texts")
        visible = attention.to(dtype=torch.bool)
        if bool((~visible.any(dim=1)).any().item()):
            raise ValueError("local tokenizer produced an empty visible sequence")

        self.model.eval()
        device = _device(self.model)
        model_inputs = _move(dict(encoded), device)
        with torch.no_grad():
            if self.config.generator_family == "seq2seq_lm":
                get_encoder = getattr(self.model, "get_encoder", None)
                if not callable(get_encoder):
                    raise ValueError("seq2seq local generator must expose get_encoder()")
                encoder = get_encoder()
                output = encoder(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                output = self.model(
                    **model_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
        token_hidden = _hidden(output)
        if token_hidden.size(0) != len(selected) or token_hidden.size(1) != attention.size(1):
            raise ValueError("generator hidden sequence does not align with tokenizer attention mask")

        visible_device = visible.to(device=token_hidden.device)
        if self.config.pooling == "last_visible":
            positions = torch.arange(
                token_hidden.size(1),
                device=token_hidden.device,
            ).unsqueeze(0).expand(token_hidden.size(0), -1)
            last_visible = positions.masked_fill(~visible_device, -1).max(dim=1).values
            if bool((last_visible < 0).any().item()):
                raise ValueError("local tokenizer produced no visible token for state pooling")
            rows = torch.arange(token_hidden.size(0), device=token_hidden.device)
            state_hidden = token_hidden[rows, last_visible]
        else:
            weights = visible_device.to(dtype=token_hidden.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            state_hidden = (token_hidden * weights).sum(dim=1) / denominator

        return {
            "token_hidden": token_hidden.detach().cpu().contiguous(),
            "state_hidden": state_hidden.detach().cpu().contiguous(),
            "attention_mask": attention.detach().cpu().contiguous(),
        }


__all__ = [
    "LocalDynamicHiddenStateConfig",
    "LocalGeneratorHiddenStateProvider",
]
