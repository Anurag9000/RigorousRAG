"""Concrete ColBERT-style token-vector encoder using a pinned transformer model."""
from __future__ import annotations

from typing import Sequence

from tools.retrieval_model_contracts import RetrievalModelSpec, finite_vector


class HuggingFaceLateInteractionBackend:
    def __init__(self, spec: RetrievalModelSpec, *, device: str = "cpu", normalize: bool = True) -> None:
        if spec.mode != "late-interaction":
            raise ValueError("spec.mode must be late-interaction")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("torch and transformers are required for late interaction retrieval") from exc
        kwargs = {
            "revision": spec.revision,
            "local_files_only": not spec.allow_download,
            "trust_remote_code": spec.trust_remote_code,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_name, **kwargs)
            self.model = AutoModel.from_pretrained(spec.model_name, **kwargs).to(device)
        except Exception as exc:
            raise RuntimeError("pinned late-interaction model could not be loaded") from exc
        self.model.eval()
        self.spec = spec
        self.device = device
        self.normalize = bool(normalize)
        self.torch = torch

    def encode_tokens(self, texts: Sequence[str]) -> Sequence[Sequence[Sequence[float]]]:
        if not texts or len(texts) > 4096:
            raise ValueError("texts must be a bounded non-empty sequence")
        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.spec.max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**encoded)
            hidden = output.last_hidden_state
            if self.normalize:
                hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=-1)
        masks = encoded.get("attention_mask")
        rows = []
        for index, matrix in enumerate(hidden.detach().cpu().tolist()):
            mask = None if masks is None else masks[index].detach().cpu().tolist()
            tokens = []
            for token_index, vector in enumerate(matrix):
                if mask is not None and not bool(mask[token_index]):
                    continue
                tokens.append(finite_vector(vector))
            if not tokens:
                raise RuntimeError("late-interaction model returned no active token vectors")
            rows.append(tokens)
        return rows


__all__ = ["HuggingFaceLateInteractionBackend"]
