"""Compatibility shim over the scientific-integrity implementation.

The legacy implementation remains byte-for-byte preserved in ``integrity_legacy``.
This module normalizes visual budgets, bounds direct comparison iterables, and
replaces retained-PDF access so verification and rendering consume one immutable
byte snapshot.
"""

from __future__ import annotations

import base64
import itertools
import math
import re
import sys
from typing import Any, Iterable, List, Optional, Tuple

import fitz

from tools.config import bounded_int_env
from tools.security import normalize_owner_id

for _name, _default, _minimum, _maximum in (
    ("VISUAL_MAX_PDF_PAGES", 500, 1, 5000),
    ("VISUAL_MAX_RENDER_PIXELS", 2_000_000, 1_000_000, 100_000_000),
    ("VISUAL_MAX_ENCODED_BYTES", 10_000_000, 100_000, 100_000_000),
):
    bounded_int_env(
        _name,
        _default,
        minimum=_minimum,
        maximum=_maximum,
        write_back=True,
    )

from tools import integrity_legacy as _implementation
from tools.security import DEFAULT_MAX_UPLOAD_BYTES

_original_compare_papers = _implementation.compare_papers
_original_generate_comparison_matrix = _implementation.generate_comparison_matrix


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _bounded_text(
    value: Any,
    label: str,
    *,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    bounded = value.strip()
    if (
        not bounded
        or len(bounded) > max_length
        or _contains_ascii_control(bounded)
    ):
        raise ValueError(
            f"{label} must contain between 1 and {max_length:,} valid characters."
        )
    return bounded


def _bounded_values(
    values: Iterable[Any],
    label: str,
    *,
    max_items: int,
    max_length: int,
) -> List[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an array, not a string.")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable.") from exc
    raw_values = list(itertools.islice(iterator, max_items + 1))
    if len(raw_values) > max_items:
        raise ValueError(f"{label} supports at most {max_items} items.")
    bounded: List[str] = []
    for raw in raw_values:
        value = _bounded_text(raw, f"Each {label} item", max_length=max_length)
        if value not in bounded:
            bounded.append(value)
    return bounded


def _extract_figure_region(pdf_bytes: bytes, figure_id: str) -> Tuple[str, int, str]:
    """Render a bounded caption-adjacent region from immutable PDF bytes."""

    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("Visual entailment requires an immutable PDF byte snapshot.")
    payload = bytes(pdf_bytes)
    if not payload or len(payload) > DEFAULT_MAX_UPLOAD_BYTES:
        raise ValueError("The retained PDF source bytes are missing or oversized.")
    needle = _bounded_text(figure_id, "figure_id", max_length=200)
    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as exc:
        raise ValueError("The retained PDF could not be opened safely.") from exc
    try:
        if document.needs_pass:
            raise ValueError("Encrypted PDFs are not supported.")
        if not 1 <= int(document.page_count) <= _implementation._VISUAL_MAX_PDF_PAGES:
            raise ValueError("The retained PDF exceeds the visual page-count limit.")
        candidates = [needle]
        compact = re.sub(r"\s+", " ", needle.replace(".", "")).strip()
        if compact and compact not in candidates:
            candidates.append(compact)
        for page_index, page in enumerate(document):
            rectangles = []
            for candidate in candidates:
                rectangles = page.search_for(candidate)
                if rectangles:
                    break
            if not rectangles:
                continue
            caption = rectangles[0]
            page_rect = page.rect
            height = min(max(page_rect.height * 0.48, 220), 520)
            clip = fitz.Rect(
                page_rect.x0,
                max(page_rect.y0, caption.y0 - height),
                page_rect.x1,
                min(page_rect.y1, caption.y1 + 45),
            )
            render_width = math.ceil(float(clip.width) * 2.0)
            render_height = math.ceil(float(clip.height) * 2.0)
            if (
                render_width <= 0
                or render_height <= 0
                or render_width * render_height
                > _implementation._VISUAL_MAX_RENDER_PIXELS
            ):
                raise ValueError(
                    "The figure region exceeds the visual render-pixel limit."
                )
            try:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    clip=clip,
                    alpha=False,
                )
                render_pixels = int(pixmap.width) * int(pixmap.height)
                if render_pixels > _implementation._VISUAL_MAX_RENDER_PIXELS:
                    raise ValueError(
                        "The figure region exceeds the visual render-pixel limit."
                    )
                png_bytes = pixmap.tobytes("png")
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError("The figure region could not be rendered safely.") from exc
            encoded_bytes = base64.b64encode(png_bytes)
            if len(encoded_bytes) > _implementation._VISUAL_MAX_ENCODED_BYTES:
                raise ValueError(
                    "The encoded figure region exceeds the visual payload-byte limit."
                )
            caption_text = page.get_textbox(
                fitz.Rect(
                    page_rect.x0,
                    max(page_rect.y0, caption.y0 - 10),
                    page_rect.x1,
                    min(page_rect.y1, caption.y1 + 120),
                )
            ).strip()
            return encoded_bytes.decode("ascii"), page_index + 1, caption_text[:2000]
        raise ValueError(
            "The figure label was not found as selectable text. Provide the exact "
            "caption label or enable OCR before ingestion."
        )
    finally:
        document.close()


def check_visual_entailment(
    claim_text: str,
    figure_id: str,
    doc_id: str,
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    """Check one figure using the exact bytes verified by the private registry."""

    claim = _bounded_text(claim_text, "claim_text", max_length=10_000)
    figure = _bounded_text(figure_id, "figure_id", max_length=200)
    document_id = _bounded_text(doc_id, "doc_id", max_length=200)
    owner = normalize_owner_id(owner_id)
    model_name = _bounded_text(model, "model", max_length=200)

    metadata = _implementation._document_metadata(document_id, owner)
    source_bytes = _implementation.get_document_store().source_bytes(
        owner_id=owner,
        doc_id=document_id,
    )
    if source_bytes is None:
        return _implementation._json(
            _implementation.VisualEntailmentResult(
                claim_text=claim,
                figure_id=figure,
                verdict=_implementation.EntailmentVerdict.INSUFFICIENT,
                rationale=(
                    "No retained owner-scoped PDF source is available. Re-ingest with "
                    "RETAIN_SOURCE_FILES=true to use visual entailment."
                ),
                confidence=1.0,
                evidence_note="No image could be extracted.",
            ).model_dump()
        )
    try:
        image_b64, page_number, caption_text = _implementation._extract_figure_region(
            source_bytes,
            figure,
        )
    except Exception as exc:
        return _implementation._json(
            _implementation.VisualEntailmentResult(
                claim_text=claim,
                figure_id=figure,
                verdict=_implementation.EntailmentVerdict.INSUFFICIENT,
                rationale=str(exc),
                confidence=1.0,
                evidence_note="Visual evidence was not available.",
            ).model_dump()
        )
    citation = _implementation._document_citation(
        metadata,
        doc_id=document_id,
        snippet=caption_text or f"Figure region for {figure}",
        page_number=page_number,
        source_id=f"{document_id}:page:{page_number}:{figure}",
    )
    if client is None:
        result = _implementation.VisualEntailmentResult(
            claim_text=claim,
            figure_id=figure,
            verdict=_implementation.EntailmentVerdict.INSUFFICIENT,
            rationale="A vision-capable model is not configured.",
            confidence=1.0,
            page_number=page_number,
            evidence_note=caption_text or None,
        )
    else:
        prompt = (
            "Evaluate only whether the supplied figure region supports the claim. "
            "Return JSON with claim_text, figure_id, verdict "
            "(supports|contradicts|insufficient|uncertain), rationale, confidence. "
            "Do not infer details that are not visible. Caption text: "
            f"{caption_text[:2000]}"
        )
        user_content = [
            {
                "type": "text",
                "text": f"Claim: {claim}\nFigure label: {figure}\n{prompt}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
        try:
            raw = _implementation._completion(
                client,
                model=model_name,
                system="You are a conservative scientific figure reviewer.",
                user=user_content,
                max_tokens=700,
                json_mode=True,
            )
            parsed = _implementation._parse_json_object(raw)
            parsed.update(
                {
                    "claim_text": claim,
                    "figure_id": figure,
                    "page_number": page_number,
                    "evidence_note": caption_text or None,
                }
            )
            result = _implementation.VisualEntailmentResult(**parsed)
        except Exception as exc:
            result = _implementation.VisualEntailmentResult(
                claim_text=claim,
                figure_id=figure,
                verdict=_implementation.EntailmentVerdict.UNCERTAIN,
                rationale=f"Vision analysis failed: {type(exc).__name__}.",
                confidence=0.0,
                page_number=page_number,
                evidence_note=caption_text or None,
            )
    response = result.model_dump()
    response["citations"] = [citation.model_dump(exclude_none=True)]
    return _implementation._json(response)


def compare_papers(
    doc_ids: Iterable[Any],
    query: str,
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    documents = _bounded_values(
        doc_ids,
        "doc_ids",
        max_items=10,
        max_length=200,
    )
    question = _bounded_text(query, "query", max_length=10_000)
    owner = normalize_owner_id(owner_id)
    model_name = _bounded_text(model, "model", max_length=200)
    return _original_compare_papers(
        documents,
        question,
        owner_id=owner,
        client=client,
        model=model_name,
    )


def generate_comparison_matrix(
    doc_ids: Iterable[Any],
    metrics: Iterable[Any],
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    documents = _bounded_values(
        doc_ids,
        "doc_ids",
        max_items=10,
        max_length=200,
    )
    bounded_metrics = _bounded_values(
        metrics,
        "metrics",
        max_items=12,
        max_length=500,
    )
    owner = normalize_owner_id(owner_id)
    model_name = _bounded_text(model, "model", max_length=200)
    return _original_generate_comparison_matrix(
        documents,
        bounded_metrics,
        owner_id=owner,
        client=client,
        model=model_name,
    )


_implementation._extract_figure_region = _extract_figure_region
_implementation.check_visual_entailment = check_visual_entailment
_implementation.compare_papers = compare_papers
_implementation.generate_comparison_matrix = generate_comparison_matrix
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
