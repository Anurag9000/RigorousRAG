"""Injected transformer-style backend for page-native late interaction.

The backend receives an already-created processor/model; it never calls from_pretrained,
downloads weights, or chooses remote code.  It supports ColPali/ColQwen-like processors
that expose query/image batches and models that return token/patch embeddings.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping, Sequence

from tools.page_late_interaction import PageEmbeddingBackend

_MAX_QUERY_CHARS = 20_000
_MAX_PAGE_BYTES = 100_000_000
_MAX_TOKENS = 16_384
_MAX_DIM = 8_192


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _to_python_matrix(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    # Drop a singleton batch dimension.
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence) and value[0] and isinstance(value[0][0], Sequence):
        if len(value) != 1:
            raise ValueError(f"{label} must contain exactly one batch item")
        value = value[0]
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or not 1 <= len(value) <= _MAX_TOKENS:
        raise ValueError(f"{label} has an invalid token/patch count")
    rows: list[tuple[float, ...]] = []
    dimension = None
    for raw_row in value:
        if isinstance(raw_row, (str, bytes, bytearray)) or not isinstance(raw_row, Sequence):
            raise ValueError(f"{label} contains an invalid row")
        if dimension is None:
            dimension = len(raw_row)
            if not 1 <= dimension <= _MAX_DIM:
                raise ValueError(f"{label} has an invalid embedding dimension")
        if len(raw_row) != dimension:
            raise ValueError(f"{label} rows have inconsistent dimensions")
        row: list[float] = []
        for raw in raw_row:
            if isinstance(raw, bool):
                raise ValueError(f"{label} contains a non-numeric value")
            parsed = float(raw)
            if not math.isfinite(parsed):
                raise ValueError(f"{label} contains a non-finite value")
            row.append(parsed)
        rows.append(tuple(row))
    return tuple(rows)


class InjectedTransformerPageBackend(PageEmbeddingBackend):
    """Adapter for already-loaded page retrieval processor/model objects."""

    def __init__(
        self,
        *,
        processor: Any,
        model: Any,
        model_id: str,
        device: str = "cpu",
        embedding_field: str = "embeddings",
        query_processor_method: str = "process_queries",
        image_processor_method: str = "process_images",
    ) -> None:
        if processor is None or model is None:
            raise ValueError("processor and model must be supplied explicitly")
        self.processor = processor
        self.model = model
        self._model_id = _text(model_id, "model_id", 300)
        self.device = _text(device, "device", 64)
        self.embedding_field = _text(embedding_field, "embedding_field", 100)
        self.query_processor_method = _text(query_processor_method, "query_processor_method", 100)
        self.image_processor_method = _text(image_processor_method, "image_processor_method", 100)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _move_batch(self, batch: Any) -> Any:
        if hasattr(batch, "to"):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {key: (value.to(self.device) if hasattr(value, "to") else value) for key, value in batch.items()}
        return batch

    def _call_model(self, batch: Any) -> Any:
        moved = self._move_batch(batch)
        # Torch inference mode is optional; importing torch must never trigger model loading.
        try:
            import torch  # type: ignore
        except Exception:
            torch = None
        if torch is not None:
            with torch.inference_mode():
                output = self.model(**moved) if isinstance(moved, Mapping) else self.model(moved)
        else:
            output = self.model(**moved) if isinstance(moved, Mapping) else self.model(moved)
        if isinstance(output, Mapping):
            value = output.get(self.embedding_field)
            if value is None:
                value = output.get("last_hidden_state")
        else:
            value = getattr(output, self.embedding_field, None)
            if value is None:
                value = getattr(output, "last_hidden_state", None)
            if value is None and not isinstance(output, (str, bytes, bytearray)):
                value = output
        if value is None:
            raise RuntimeError("page retrieval model did not expose token/patch embeddings")
        return value

    def embed_query(self, query: str) -> Sequence[Sequence[float]]:
        selected = _text(query, "query", _MAX_QUERY_CHARS)
        method = getattr(self.processor, self.query_processor_method, None)
        if callable(method):
            batch = method([selected])
        elif callable(self.processor):
            batch = self.processor(text=[selected], return_tensors="pt", padding=True)
        else:
            raise RuntimeError("processor does not support query processing")
        return _to_python_matrix(self._call_model(batch), "query embeddings")

    def embed_page(self, rendered_page: bytes, *, page_number: int) -> Sequence[Sequence[float]]:
        if not isinstance(rendered_page, bytes) or not rendered_page or len(rendered_page) > _MAX_PAGE_BYTES:
            raise ValueError("rendered_page is empty or exceeds the byte limit")
        if isinstance(page_number, bool) or not isinstance(page_number, int) or not 1 <= page_number <= 100_000:
            raise ValueError("page_number is invalid")
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("Pillow is required to decode rendered page bytes") from exc
        with Image.open(io.BytesIO(rendered_page)) as image:
            image.verify()
        with Image.open(io.BytesIO(rendered_page)) as image:
            prepared = image.convert("RGB")
            method = getattr(self.processor, self.image_processor_method, None)
            if callable(method):
                batch = method([prepared])
            elif callable(self.processor):
                batch = self.processor(images=[prepared], return_tensors="pt")
            else:
                raise RuntimeError("processor does not support image processing")
            return _to_python_matrix(self._call_model(batch), "page embeddings")


__all__ = ["InjectedTransformerPageBackend"]
