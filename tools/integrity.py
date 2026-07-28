"""Compatibility shim over the scientific-integrity implementation.

The legacy implementation remains byte-for-byte preserved in ``integrity_legacy``.
This module replaces only retained-PDF visual access so verification and rendering
consume the same immutable descriptor-anchored byte snapshot.
"""

from __future__ import annotations

import base64
import math
import re
import sys
from typing import Any, Optional, Tuple

import fitz

from tools import integrity_legacy as _implementation
from tools.security import DEFAULT_MAX_UPLOAD_BYTES


def _extract_figure_region(pdf_bytes: bytes, figure_id: str) -> Tuple[str, int, str]:
    """Render a bounded caption-adjacent region from immutable PDF bytes."""

    if not isinstance(pdf_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("Visual entailment requires an immutable PDF byte snapshot.")
    payload = bytes(pdf_bytes)
    if not payload or len(payload) > DEFAULT_MAX_UPLOAD_BYTES:
        raise ValueError("The retained PDF source bytes are missing or oversized.")
    needle = (figure_id or "").strip()
    if not needle or len(needle) > 200:
        raise ValueError("figure_id must contain between 1 and 200 characters.")
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

    metadata = _implementation._document_metadata(doc_id, owner_id)
    source_bytes = _implementation.get_document_store().source_bytes(
        owner_id=owner_id,
        doc_id=doc_id,
    )
    if source_bytes is None:
        return _implementation._json(
            _implementation.VisualEntailmentResult(
                claim_text=claim_text,
                figure_id=figure_id,
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
            figure_id,
        )
    except Exception as exc:
        return _implementation._json(
            _implementation.VisualEntailmentResult(
                claim_text=claim_text,
                figure_id=figure_id,
                verdict=_implementation.EntailmentVerdict.INSUFFICIENT,
                rationale=str(exc),
                confidence=1.0,
                evidence_note="Visual evidence was not available.",
            ).model_dump()
        )
    citation = _implementation._document_citation(
        metadata,
        doc_id=doc_id,
        snippet=caption_text or f"Figure region for {figure_id}",
        page_number=page_number,
        source_id=f"{doc_id}:page:{page_number}:{figure_id}",
    )
    if client is None:
        result = _implementation.VisualEntailmentResult(
            claim_text=claim_text,
            figure_id=figure_id,
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
                "text": f"Claim: {claim_text}\nFigure label: {figure_id}\n{prompt}",
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
                model=model,
                system="You are a conservative scientific figure reviewer.",
                user=user_content,
                max_tokens=700,
                json_mode=True,
            )
            parsed = _implementation._parse_json_object(raw)
            parsed.update(
                {
                    "claim_text": claim_text,
                    "figure_id": figure_id,
                    "page_number": page_number,
                    "evidence_note": caption_text or None,
                }
            )
            result = _implementation.VisualEntailmentResult(**parsed)
        except Exception as exc:
            result = _implementation.VisualEntailmentResult(
                claim_text=claim_text,
                figure_id=figure_id,
                verdict=_implementation.EntailmentVerdict.UNCERTAIN,
                rationale=f"Vision analysis failed: {type(exc).__name__}.",
                confidence=0.0,
                page_number=page_number,
                evidence_note=caption_text or None,
            )
    response = result.model_dump()
    response["citations"] = [citation.model_dump(exclude_none=True)]
    return _implementation._json(response)


# Override only the visual functions on the original module object, then expose that
# object under this module name. Existing monkeypatch paths keep targeting the globals
# used by all non-visual legacy functions.
_implementation._extract_figure_region = _extract_figure_region
_implementation.check_visual_entailment = check_visual_entailment
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
