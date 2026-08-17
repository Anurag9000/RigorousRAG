"""Verified-local Hugging Face image+text entailment adapter.

The adapter targets reviewed multimodal sequence-classification checkpoints whose logits
can be mapped explicitly to entailment/neutral/contradiction.  It never chooses model
names, revisions or labels heuristically: the caller supplies a verified local artifact
binding, immutable semantic model identity and an exact class-index mapping.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any, Mapping

from evaluation.multimodal_support import (
    MultimodalEvidence,
    MultimodalSupportScore,
)
from evaluation.semantic_support import ModelIdentity, SemanticLabel, SemanticProbabilities
from models.local_hf_adapters import LocalArtifactBinding

_MAX_PIXELS = 100_000_000


def _positive_int(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class MultimodalLabelMapping:
    entailment_index: int
    neutral_index: int
    contradiction_index: int

    def __post_init__(self) -> None:
        values = tuple(_positive_int(value + 1, "class index", 1_000_000) - 1 for value in (
            self.entailment_index,
            self.neutral_index,
            self.contradiction_index,
        ))
        if len(set(values)) != 3:
            raise ValueError("semantic class indices must be unique")
        object.__setattr__(self, "entailment_index", values[0])
        object.__setattr__(self, "neutral_index", values[1])
        object.__setattr__(self, "contradiction_index", values[2])

    @property
    def maximum_index(self) -> int:
        return max(self.entailment_index, self.neutral_index, self.contradiction_index)


class LocalHFMultimodalEntailmentScorer:
    """Local-only visual entailment scorer with explicit class semantics."""

    def __init__(
        self,
        *,
        binding: LocalArtifactBinding,
        model_identity: ModelIdentity,
        label_mapping: MultimodalLabelMapping,
        device: str = "auto",
        max_text_tokens: int = 512,
        max_pixels: int = 40_000_000,
    ) -> None:
        if not isinstance(binding, LocalArtifactBinding):
            raise ValueError("binding must be LocalArtifactBinding")
        binding.verify()
        if not isinstance(model_identity, ModelIdentity):
            raise ValueError("model_identity must be ModelIdentity")
        if model_identity.artifact_sha256 is None:
            raise ValueError("multimodal model identity requires artifact_sha256")
        if model_identity.artifact_sha256 != binding.model_tree_sha256:
            raise ValueError("model identity artifact digest differs from local artifact binding")
        if not isinstance(label_mapping, MultimodalLabelMapping):
            raise ValueError("label_mapping must be MultimodalLabelMapping")
        self.binding = binding
        self._model_identity = model_identity
        self.label_mapping = label_mapping
        self.device_name = str(device).strip().lower()
        if not self.device_name:
            raise ValueError("device must be non-empty")
        self.max_text_tokens = _positive_int(max_text_tokens, "max_text_tokens", 1_000_000)
        self.max_pixels = _positive_int(max_pixels, "max_pixels", _MAX_PIXELS)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

    @property
    def model_identity(self) -> ModelIdentity:
        return self._model_identity

    def _load(self) -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional model dependency.
            raise RuntimeError("multimodal entailment requires optional torch + transformers") from exc
        if self._device is None:
            selected = self.device_name
            if selected == "auto":
                selected = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = torch.device(selected)
            if self._device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for multimodal entailment but unavailable")
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.binding.model_root,
                local_files_only=True,
                trust_remote_code=False,
            ).to(self._device)
            self._model.eval()
            num_labels = getattr(getattr(self._model, "config", None), "num_labels", None)
            if isinstance(num_labels, int) and self.label_mapping.maximum_index >= num_labels:
                raise RuntimeError("configured semantic label index exceeds model output classes")
        return torch, self._processor, self._model, self._device

    def _decode_image(self, value: bytes) -> Any:
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional image dependency.
            raise RuntimeError("multimodal entailment requires Pillow for bounded image decoding") from exc
        try:
            with Image.open(io.BytesIO(value)) as opened:
                width, height = opened.size
                if width < 1 or height < 1 or width * height > self.max_pixels:
                    raise RuntimeError("visual evidence dimensions exceed configured safety bound")
                opened.load()
                return opened.convert("RGB")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("visual evidence image could not be decoded") from exc

    def _render_text(self, claim_text: str, evidence: MultimodalEvidence) -> str:
        selected = claim_text.strip()
        if not selected:
            raise ValueError("claim_text must be non-empty")
        if evidence.evidence_text is None:
            return selected
        # The text supplement is explicit evidence, not hidden OCR.  Delimiters make
        # the model input reproducible and prevent accidental claim/evidence inversion.
        return f"Claim: {selected}\nEvidence text: {evidence.evidence_text}"

    def score(self, claim_id: str, claim_text: str, evidence: MultimodalEvidence) -> MultimodalSupportScore:
        if not isinstance(evidence, MultimodalEvidence):
            raise ValueError("evidence must be MultimodalEvidence")
        if not isinstance(claim_id, str) or not claim_id.strip():
            raise ValueError("claim_id must be non-empty")
        torch, processor, model, device = self._load()
        image = self._decode_image(evidence.image_bytes)
        rendered = self._render_text(claim_text, evidence)
        try:
            encoded = processor(
                images=image,
                text=rendered,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_text_tokens,
            )
        except Exception as exc:
            raise RuntimeError("multimodal processor failed to encode evidence") from exc
        if not isinstance(encoded, Mapping) or not encoded:
            raise RuntimeError("multimodal processor returned an invalid model batch")
        try:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] <= self.label_mapping.maximum_index:
                raise RuntimeError("multimodal classifier returned an incompatible logit shape")
            probabilities = torch.softmax(logits[0].float(), dim=-1).detach().cpu()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("multimodal entailment inference failed") from exc
        semantic = SemanticProbabilities(
            entailment=float(probabilities[self.label_mapping.entailment_index]),
            neutral=float(probabilities[self.label_mapping.neutral_index]),
            contradiction=float(probabilities[self.label_mapping.contradiction_index]),
        )
        evidence_text_sha = None
        if evidence.evidence_text is not None:
            evidence_text_sha = hashlib.sha256(evidence.evidence_text.encode("utf-8")).hexdigest()
        return MultimodalSupportScore(
            claim_id=claim_id.strip(),
            claim_sha256=hashlib.sha256(claim_text.strip().encode("utf-8")).hexdigest(),
            anchor=evidence.anchor,
            probabilities=semantic,
            model=self.model_identity,
            evidence_text_sha256=evidence_text_sha,
        )


__all__ = ["LocalHFMultimodalEntailmentScorer", "MultimodalLabelMapping"]
