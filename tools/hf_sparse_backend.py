"""Concrete SPLADE-style sparse encoder using a pinned masked-language model."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from tools.retrieval_model_contracts import RetrievalModelSpec


class HuggingFaceSparseBackend:
    def __init__(self, spec: RetrievalModelSpec, *, device: str = "cpu") -> None:
        if spec.mode != "sparse":
            raise ValueError("spec.mode must be sparse")
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("torch and transformers are required for sparse retrieval") from exc
        kwargs = {
            "revision": spec.revision,
            "local_files_only": not spec.allow_download,
            "trust_remote_code": spec.trust_remote_code,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_name, **kwargs)
            self.model = AutoModelForMaskedLM.from_pretrained(spec.model_name, **kwargs).to(device)
        except Exception as exc:
            raise RuntimeError("pinned sparse retrieval model could not be loaded") from exc
        self.model.eval()
        self.spec = spec
        self.device = device
        self.torch = torch

    def encode_sparse(self, texts: Sequence[str]) -> Sequence[Mapping[str, float]]:
        if not texts or len(texts) > 4096:
            raise ValueError("texts must be a bounded non-empty sequence")
        encoded = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=self.spec.max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            weights = self.torch.log1p(self.torch.relu(logits)).max(dim=1).values
        rows = []
        for vector in weights:
            count = min(self.spec.max_terms, int(vector.shape[-1]))
            values, indices = self.torch.topk(vector, k=count)
            row: dict[str, float] = {}
            for raw_value, raw_index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                value = float(raw_value)
                if not math.isfinite(value) or value <= 0.0:
                    continue
                term = str(self.tokenizer.convert_ids_to_tokens(int(raw_index))).strip()
                if term and len(term) <= 500:
                    row[term] = max(row.get(term, 0.0), value)
            rows.append(row)
        return rows


__all__ = ["HuggingFaceSparseBackend"]
