"""Supply-chain-admitted construction paths for local scientific/model adapters.

Low-level local adapters remain reusable for research and unit-level composition.  Serving
or authoritative extraction code should construct model-backed scientific adapters through
this module so exact model/processor trees are re-hashed and re-bound to admitted artifact
proofs immediately before adapter construction.  Construction remains lazy: no model is
loaded merely by importing or calling these factories.
"""

from __future__ import annotations

from models.admitted_local_artifacts import AdmittedLocalArtifactBinding, require_admitted_local_binding
from models.local_hf_multimodal_entailment import LocalHFMultimodalEntailmentScorer, MultimodalLabelMapping
from evaluation.semantic_support import ModelIdentity
from scientific.local_chart_adapters import ChartDecodeConfig, LocalHFChartToStructureAdapter
from scientific.local_document_model_adapters import (
    LayoutAdapterConfig,
    LocalHFFormulaOCRAdapter,
    LocalHFLayoutAdapter,
    LocalHFObjectDetector,
    LocalHFTableStructureAdapter,
    OCRProvider,
    TableStructureConfig,
)


def build_admitted_object_detector(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    device: str = "auto",
    threshold: float = 0.5,
) -> LocalHFObjectDetector:
    """Construct a layout/table detector only after supply-chain admission re-verification."""

    binding = require_admitted_local_binding(admitted_binding)
    return LocalHFObjectDetector(binding, device=device, threshold=threshold)


def build_admitted_layout_adapter(
    admitted_binding: AdmittedLocalArtifactBinding,
    config: LayoutAdapterConfig,
    *,
    ocr: OCRProvider | None = None,
    device: str = "auto",
    threshold: float = 0.5,
) -> LocalHFLayoutAdapter:
    """Construct an admitted layout adapter around an admitted object detector."""

    detector = build_admitted_object_detector(admitted_binding, device=device, threshold=threshold)
    return LocalHFLayoutAdapter(detector, config, ocr=ocr)


def build_admitted_table_structure_adapter(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    ocr: OCRProvider | None = None,
    config: TableStructureConfig = TableStructureConfig(),
    device: str = "auto",
    threshold: float = 0.5,
) -> LocalHFTableStructureAdapter:
    """Construct an admitted table-structure adapter around an admitted detector."""

    detector = build_admitted_object_detector(admitted_binding, device=device, threshold=threshold)
    return LocalHFTableStructureAdapter(detector, ocr=ocr, config=config)


def build_admitted_formula_ocr_adapter(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    device: str = "auto",
    max_new_tokens: int = 256,
) -> LocalHFFormulaOCRAdapter:
    """Construct an admitted vision-encoder-decoder formula OCR adapter."""

    binding = require_admitted_local_binding(admitted_binding)
    return LocalHFFormulaOCRAdapter(binding, device=device, max_new_tokens=max_new_tokens)


def build_admitted_chart_adapter(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    config: ChartDecodeConfig = ChartDecodeConfig(),
    device: str = "auto",
) -> LocalHFChartToStructureAdapter:
    """Construct an admitted chart-to-structure vision adapter."""

    binding = require_admitted_local_binding(admitted_binding)
    return LocalHFChartToStructureAdapter(binding, config=config, device=device)


def build_admitted_multimodal_entailment_scorer(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    model_identity: ModelIdentity,
    label_mapping: MultimodalLabelMapping,
    device: str = "auto",
    max_text_tokens: int = 512,
    max_pixels: int = 40_000_000,
) -> LocalHFMultimodalEntailmentScorer:
    """Construct an admitted image+text entailment scorer.

    The scorer itself additionally requires ``model_identity.artifact_sha256`` to equal the
    admitted model tree, preserving the semantic model identity used by evaluation output.
    """

    binding = require_admitted_local_binding(admitted_binding)
    if not isinstance(model_identity, ModelIdentity):
        raise ValueError("model_identity must be ModelIdentity")
    if model_identity.artifact_sha256 != binding.model_tree_sha256:
        raise ValueError("multimodal model identity must equal the admitted model tree digest")
    return LocalHFMultimodalEntailmentScorer(
        binding=binding,
        model_identity=model_identity,
        label_mapping=label_mapping,
        device=device,
        max_text_tokens=max_text_tokens,
        max_pixels=max_pixels,
    )


__all__ = [
    "build_admitted_chart_adapter",
    "build_admitted_formula_ocr_adapter",
    "build_admitted_layout_adapter",
    "build_admitted_multimodal_entailment_scorer",
    "build_admitted_object_detector",
    "build_admitted_table_structure_adapter",
]
