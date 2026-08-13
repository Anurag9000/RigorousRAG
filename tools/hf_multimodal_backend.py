"""Pinned CLIP/SigLIP-style multimodal execution with injected image decoding."""
from typing import Callable, Sequence

from tools.retrieval_model_contracts import MultimodalInput, RetrievalModelSpec, finite_vector


class HuggingFaceMultimodalBackend:
    def __init__(self, spec: RetrievalModelSpec, *, image_decoder: Callable[[bytes], object], device: str = "cpu"):
        if spec.mode != "multimodal" or not callable(image_decoder):
            raise ValueError("multimodal spec and image_decoder are required")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise RuntimeError("torch and transformers are required for multimodal retrieval") from exc
        kwargs = {"revision": spec.revision, "local_files_only": not spec.allow_download,
                  "trust_remote_code": spec.trust_remote_code}
        self.processor = AutoProcessor.from_pretrained(spec.model_name, **kwargs)
        self.model = AutoModel.from_pretrained(spec.model_name, **kwargs).to(device).eval()
        if not callable(getattr(self.model, "get_text_features", None)) or not callable(getattr(self.model, "get_image_features", None)):
            raise RuntimeError("model must expose text and image feature methods")
        self.torch, self.decoder, self.device = torch, image_decoder, device

    def encode_multimodal(self, items: Sequence[MultimodalInput]) -> Sequence[Sequence[float]]:
        rows = []
        for item in items:
            vectors = []
            with self.torch.no_grad():
                if item.text is not None:
                    inputs = self.processor(text=[item.text], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    vectors.append(self.model.get_text_features(**inputs)[0])
                if item.image_bytes is not None:
                    inputs = self.processor(images=self.decoder(item.image_bytes), return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    vectors.append(self.model.get_image_features(**inputs)[0])
            value = vectors[0] if len(vectors) == 1 else self.torch.stack(vectors).mean(dim=0)
            value = self.torch.nn.functional.normalize(value, p=2, dim=-1)
            rows.append(finite_vector(value.detach().cpu().tolist()))
        return rows


__all__ = ["HuggingFaceMultimodalBackend"]
