"""Concrete local-only scientific document model adapters.

The structured document IR is model-agnostic; this module supplies executable adapters
for common Hugging Face object-detection and vision-encoder-decoder families plus a
Tesseract OCR adapter.  Model/processor directories are verified local artifacts and all
``from_pretrained`` calls use ``local_files_only=True`` with remote code disabled.
Nothing loads or runs until an adapter method is explicitly invoked.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from models.local_hf_adapters import LocalArtifactBinding
from scientific.document_structure import (
    BoundingBox,
    DocumentRegion,
    FigureStructure,
    FormulaRecord,
    ReadingOrderEdge,
    RegionKind,
    SourceAnchor,
    StructuredDocument,
    StructuredTable,
    TableCell,
)


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _require_vision_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except Exception as exc:  # pragma: no cover - optional scientific dependency.
        raise RuntimeError("local layout/table adapters require torch + transformers") from exc
    return torch, AutoImageProcessor, AutoModelForObjectDetection


def _device(torch_module: Any, requested: str) -> Any:
    selected = requested.strip().lower()
    if selected == "auto":
        selected = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(selected)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _image_size(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("page image must expose PIL-compatible .size")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("page image dimensions must be positive")
    return width, height


def _normalize_box(box: Sequence[float], width: int, height: int) -> BoundingBox:
    if len(box) != 4:
        raise ValueError("detection box must contain four coordinates")
    left, top, right, bottom = (float(value) for value in box)
    left = min(max(left / width, 0.0), 1.0)
    right = min(max(right / width, 0.0), 1.0)
    top = min(max(top / height, 0.0), 1.0)
    bottom = min(max(bottom / height, 0.0), 1.0)
    if right <= left or bottom <= top:
        raise ValueError("normalized detection box has non-positive extent")
    return BoundingBox(left, top, right, bottom)


def _pixel_crop(image: Any, box: BoundingBox) -> Any:
    width, height = _image_size(image)
    return image.crop(
        (
            int(round(box.left * width)),
            int(round(box.top * height)),
            int(round(box.right * width)),
            int(round(box.bottom * height)),
        )
    )


class OCRProvider(Protocol):
    def recognize(self, image: Any) -> tuple[str, float | None]: ...


class TesseractOCRProvider:
    """Concrete OCR adapter using the already-declared pytesseract runtime dependency."""

    def __init__(self, *, language: str | None = None, config: str = "") -> None:
        self.language = language
        self.config = config

    def recognize(self, image: Any) -> tuple[str, float | None]:
        try:
            import pytesseract
            from pytesseract import Output
        except Exception as exc:  # pragma: no cover - external binary/python dependency.
            raise RuntimeError("Tesseract OCR execution requires pytesseract and a configured Tesseract binary") from exc
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            config=self.config,
            output_type=Output.DICT,
        )
        words: list[str] = []
        confidences: list[float] = []
        for text, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
            selected = str(text).strip()
            if not selected:
                continue
            words.append(selected)
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if confidence >= 0.0:
                confidences.append(min(max(confidence / 100.0, 0.0), 1.0))
        return " ".join(words), (sum(confidences) / len(confidences) if confidences else None)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: BoundingBox


class LocalHFObjectDetector:
    """Verified local object-detection model used by layout and table adapters."""

    def __init__(
        self,
        binding: LocalArtifactBinding,
        *,
        device: str = "auto",
        threshold: float = 0.5,
    ) -> None:
        binding.verify()
        self.binding = binding
        self.device_name = device
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        torch, AutoImageProcessor, AutoModelForObjectDetection = _require_vision_stack()
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.binding.model_root,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._processor, self._model

    def detect(self, image: Any) -> tuple[Detection, ...]:
        torch, processor, model = self._load()
        width, height = _image_size(image)
        device = _device(torch, self.device_name)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        if not hasattr(processor, "post_process_object_detection"):
            raise RuntimeError("configured processor lacks post_process_object_detection")
        result = processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]
        id2label = getattr(model.config, "id2label", {}) or {}
        detections: list[Detection] = []
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            label = str(id2label.get(int(label_id), int(label_id)))
            detections.append(
                Detection(label, float(score.detach().cpu()), _normalize_box(box.detach().cpu().tolist(), width, height))
            )
        detections.sort(key=lambda item: (item.box.top, item.box.left, -item.score, item.label))
        return tuple(detections)


@dataclass(frozen=True)
class LayoutAdapterConfig:
    label_mapping: Mapping[str, RegionKind]
    ocr_kinds: frozenset[RegionKind] = frozenset(
        {RegionKind.TITLE, RegionKind.HEADING, RegionKind.PARAGRAPH, RegionKind.LIST, RegionKind.CAPTION, RegionKind.FOOTNOTE}
    )

    def __post_init__(self) -> None:
        mapping: dict[str, RegionKind] = {}
        for label, kind in self.label_mapping.items():
            selected_label = _identifier(label, "layout label", 500).casefold()
            mapping[selected_label] = kind if isinstance(kind, RegionKind) else RegionKind(kind)
        if not mapping:
            raise ValueError("layout label mapping may not be empty")
        object.__setattr__(self, "label_mapping", mapping)
        object.__setattr__(self, "ocr_kinds", frozenset(RegionKind(value) for value in self.ocr_kinds))


class LocalHFLayoutAdapter:
    def __init__(
        self,
        detector: LocalHFObjectDetector,
        config: LayoutAdapterConfig,
        *,
        ocr: OCRProvider | None = None,
    ) -> None:
        self.detector = detector
        self.config = config
        self.ocr = ocr

    def regions(
        self,
        image: Any,
        *,
        document_id: str,
        generation_id: str,
        page: int,
        extraction_artifact_id: str,
    ) -> tuple[DocumentRegion, ...]:
        anchor = SourceAnchor(document_id, generation_id, page, extraction_artifact_id)
        regions: list[DocumentRegion] = []
        counter = 0
        for detection in self.detector.detect(image):
            kind = self.config.label_mapping.get(detection.label.casefold())
            if kind is None:
                continue
            counter += 1
            text = None
            confidence = detection.score
            if self.ocr is not None and kind in self.config.ocr_kinds:
                text, ocr_confidence = self.ocr.recognize(_pixel_crop(image, detection.box))
                text = text or None
                if ocr_confidence is not None:
                    confidence = min(confidence, ocr_confidence)
            regions.append(
                DocumentRegion(
                    region_id=f"p{page}-r{counter}",
                    kind=kind,
                    box=detection.box,
                    anchor=anchor,
                    text=text,
                    confidence=confidence,
                    metadata={"detector_label": detection.label},
                )
            )
        return tuple(regions)


@dataclass(frozen=True)
class TableStructureConfig:
    row_labels: frozenset[str] = frozenset({"table row", "row"})
    column_labels: frozenset[str] = frozenset({"table column", "column"})
    header_labels: frozenset[str] = frozenset({"table column header", "column header"})

    def __post_init__(self) -> None:
        for name in ("row_labels", "column_labels", "header_labels"):
            object.__setattr__(self, name, frozenset(str(value).casefold().strip() for value in getattr(self, name)))


class LocalHFTableStructureAdapter:
    """Convert row/column object detections into a rectangular table cell topology."""

    def __init__(
        self,
        detector: LocalHFObjectDetector,
        *,
        ocr: OCRProvider | None = None,
        config: TableStructureConfig = TableStructureConfig(),
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.config = config

    @staticmethod
    def _intersection(left: BoundingBox, right: BoundingBox) -> BoundingBox | None:
        x1, y1 = max(left.left, right.left), max(left.top, right.top)
        x2, y2 = min(left.right, right.right), min(left.bottom, right.bottom)
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    def extract(self, table_image: Any, *, table_region_id: str) -> StructuredTable:
        detections = self.detector.detect(table_image)
        rows = [value for value in detections if value.label.casefold() in self.config.row_labels]
        columns = [value for value in detections if value.label.casefold() in self.config.column_labels]
        headers = [value for value in detections if value.label.casefold() in self.config.header_labels]
        rows.sort(key=lambda value: (value.box.top, value.box.left))
        columns.sort(key=lambda value: (value.box.left, value.box.top))
        if not rows or not columns:
            raise ValueError("table structure detector did not produce both row and column detections")
        cells: list[TableCell] = []
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                box = self._intersection(row.box, column.box)
                if box is None:
                    continue
                text = ""
                confidence = min(row.score, column.score)
                if self.ocr is not None:
                    text, ocr_confidence = self.ocr.recognize(_pixel_crop(table_image, box))
                    if ocr_confidence is not None:
                        confidence = min(confidence, ocr_confidence)
                is_header = any(self._intersection(box, header.box) is not None for header in headers)
                cells.append(
                    TableCell(
                        cell_id=f"{table_region_id}-r{row_index}c{column_index}",
                        table_region_id=table_region_id,
                        row_start=row_index,
                        row_span=1,
                        column_start=column_index,
                        column_span=1,
                        text=text,
                        box=box,
                        is_header=is_header,
                        confidence=confidence,
                    )
                )
        if not cells:
            raise ValueError("row/column detections yielded no intersecting table cells")
        return StructuredTable(table_region_id, tuple(cells), len(rows), len(columns))


class LocalHFFormulaOCRAdapter:
    """Local vision-encoder-decoder formula OCR producing LaTeX/plain-text records."""

    def __init__(
        self,
        binding: LocalArtifactBinding,
        *,
        device: str = "auto",
        max_new_tokens: int = 256,
    ) -> None:
        binding.verify()
        self.binding = binding
        self.device_name = device
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoProcessor, VisionEncoderDecoderModel
        except Exception as exc:  # pragma: no cover - optional scientific dependency.
            raise RuntimeError("formula OCR requires torch + transformers") from exc
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.binding.model_root,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._processor, self._model

    def recognize(self, image: Any, *, formula_id: str, region_id: str) -> FormulaRecord:
        torch, processor, model = self._load()
        device = _device(torch, self.device_name)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = getattr(inputs, "pixel_values", None)
        if pixel_values is None and isinstance(inputs, Mapping):
            pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("formula processor did not return pixel_values")
        with torch.inference_mode():
            generated = model.generate(pixel_values.to(device), max_new_tokens=self.max_new_tokens)
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
            text = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        else:
            raise RuntimeError("formula processor cannot decode generated token ids")
        selected = " ".join(str(text).split())
        if not selected:
            raise ValueError("formula OCR produced empty output")
        return FormulaRecord(formula_id=formula_id, region_id=region_id, latex=selected)


def geometric_reading_order(regions: Sequence[DocumentRegion]) -> tuple[ReadingOrderEdge, ...]:
    """Deterministic page/top/left fallback order for model outputs lacking order edges."""

    ordered = sorted(
        regions,
        key=lambda region: (
            region.anchor.page,
            round(region.box.top, 6),
            round(region.box.left, 6),
            round(region.box.bottom, 6),
            region.region_id,
        ),
    )
    return tuple(
        ReadingOrderEdge(left.region_id, right.region_id, "geometric page/top/left fallback", confidence=None)
        for left, right in zip(ordered, ordered[1:])
    )


def associate_figures_and_captions(regions: Sequence[DocumentRegion]) -> tuple[FigureStructure, ...]:
    """Associate each caption to the nearest same-page figure with deterministic geometry."""

    figures = [region for region in regions if region.kind == RegionKind.FIGURE]
    captions = [region for region in regions if region.kind == RegionKind.CAPTION]
    assignments: dict[str, list[str]] = {figure.region_id: [] for figure in figures}
    for caption in captions:
        same_page = [figure for figure in figures if figure.anchor.page == caption.anchor.page]
        if not same_page:
            continue
        def distance(figure: DocumentRegion) -> tuple[float, float, str]:
            vertical = min(abs(caption.box.top - figure.box.bottom), abs(figure.box.top - caption.box.bottom))
            figure_center = (figure.box.left + figure.box.right) / 2.0
            caption_center = (caption.box.left + caption.box.right) / 2.0
            return vertical, abs(figure_center - caption_center), figure.region_id
        selected = min(same_page, key=distance)
        assignments[selected.region_id].append(caption.region_id)
    return tuple(
        FigureStructure(figure.region_id, tuple(sorted(assignments[figure.region_id])))
        for figure in figures
        if assignments[figure.region_id]
    )


def assemble_structured_document(
    *,
    document_id: str,
    generation_id: str,
    regions: Sequence[DocumentRegion],
    tables: Sequence[StructuredTable] = (),
    formulas: Sequence[FormulaRecord] = (),
) -> StructuredDocument:
    """Assemble a validated document with deterministic fallback order/figure links."""

    selected_regions = tuple(regions)
    return StructuredDocument(
        document_id=document_id,
        generation_id=generation_id,
        regions=selected_regions,
        reading_edges=geometric_reading_order(selected_regions),
        tables=tuple(tables),
        formulas=tuple(formulas),
        figures=associate_figures_and_captions(selected_regions),
    )


__all__ = [
    "Detection",
    "LayoutAdapterConfig",
    "LocalHFFormulaOCRAdapter",
    "LocalHFLayoutAdapter",
    "LocalHFObjectDetector",
    "LocalHFTableStructureAdapter",
    "OCRProvider",
    "TableStructureConfig",
    "TesseractOCRProvider",
    "assemble_structured_document",
    "associate_figures_and_captions",
    "geometric_reading_order",
]
