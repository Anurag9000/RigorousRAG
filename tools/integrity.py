"""Evidence-aware scientific analysis tools.

These tools fail closed when source evidence is unavailable. They are analytical
assistants, not substitutes for expert review or experimental replication.
"""

from __future__ import annotations

import base64
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz
from pydantic import BaseModel, Field, ValidationError

from tools.models import Citation
from tools.rag import Chunk, get_rag_layer


class EntailmentVerdict(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    INSUFFICIENT = "insufficient"
    UNCERTAIN = "uncertain"


class VisualEntailmentResult(BaseModel):
    claim_text: str
    figure_id: str
    verdict: EntailmentVerdict
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    page_number: Optional[int] = Field(default=None, ge=1)
    evidence_note: Optional[str] = None


class ProtocolStep(BaseModel):
    description: str
    temperature: Optional[str] = None
    time: Optional[str] = None
    reagent: Optional[str] = None
    notes: Optional[str] = None


class Protocol(BaseModel):
    steps: List[ProtocolStep]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class DebateResult(BaseModel):
    verdict: str
    key_issues: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    recommended_followups: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None


class ComparisonResult(BaseModel):
    consistencies: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    trends: List[str] = Field(default_factory=list)
    summary: str
    evidence_gaps: List[str] = Field(default_factory=list)


VISUAL_ENTAILMENT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "check_visual_entailment",
        "description": (
            "Check whether a specific figure in an uploaded PDF supports a claim. "
            "Requires the exact document ID and figure label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string"},
                "figure_id": {"type": "string"},
                "doc_id": {"type": "string"},
            },
            "required": ["claim_text", "figure_id", "doc_id"],
            "additionalProperties": False,
        },
    },
}

PROTOCOL_EXTRACTION_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "extract_protocol",
        "description": "Extract an explicit procedure from supplied methods text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "doc_id": {"type": "string"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

DEBATE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_scientific_debate",
        "description": (
            "Produce an advocate, skeptic, and judge analysis grounded only in "
            "the supplied evidence context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["claim", "context"],
            "additionalProperties": False,
        },
    },
}

COMPARISON_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "compare_papers",
        "description": "Compare owner-scoped uploaded papers using retrieved evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "query": {"type": "string"},
            },
            "required": ["doc_ids", "query"],
            "additionalProperties": False,
        },
    },
}

MATRIX_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "generate_comparison_matrix",
        "description": "Build one evidence-grounded comparison table across uploaded documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "metrics": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
            },
            "required": ["doc_ids", "metrics"],
            "additionalProperties": False,
        },
    },
}

CONFLICT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "detect_conflicts",
        "description": "Identify direct contradictions in supplied evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["topic", "context"],
            "additionalProperties": False,
        },
    },
}

LIMITATIONS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "extract_limitations",
        "description": "Extract explicit limitations and scope constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
}


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _completion(
    client: Any,
    *,
    model: str,
    system: str,
    user: Any,
    max_tokens: int = 1200,
    json_mode: bool = False,
) -> str:
    if client is None:
        raise RuntimeError("No compatible language-model client is configured.")
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        if not json_mode:
            raise
        kwargs.pop("response_format", None)
        response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _parse_json_object(raw: str) -> Dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object.")
    return value


def _document_metadata(doc_id: str, owner_id: str) -> Dict[str, Any]:
    rag = get_rag_layer()
    results = rag.collection.get(
        where={
            "$and": [
                {"owner_id": {"$eq": owner_id}},
                {"doc_id": {"$eq": doc_id}},
            ]
        },
        include=["metadatas"],
        limit=1,
    )
    metadatas = results.get("metadatas") or []
    if not metadatas:
        raise ValueError("The requested document was not found for this owner.")
    return dict(metadatas[0] or {})


def _document_citation(
    metadata: Dict[str, Any],
    *,
    doc_id: str,
    snippet: str,
    page_number: Optional[int] = None,
    source_id: Optional[str] = None,
) -> Citation:
    return Citation(
        label="[1]",
        title=str(metadata.get("filename") or "Uploaded document"),
        url=f"local://{doc_id}",
        source_type="uploaded_document",
        snippet=snippet,
        quote=snippet,
        source_id=source_id or doc_id,
        doc_id=doc_id,
        page_number=page_number,
    )


def _extract_figure_region(pdf_path: str, figure_id: str) -> Tuple[str, int, str]:
    """Render a caption-adjacent region rather than selecting an arbitrary image."""

    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("Visual entailment currently supports PDF documents only.")
    document = fitz.open(path)
    try:
        if document.needs_pass:
            raise ValueError("Encrypted PDFs are not supported.")
        needle = (figure_id or "").strip()
        if not needle:
            raise ValueError("figure_id is required.")
        for page_index, page in enumerate(document):
            rectangles = page.search_for(needle)
            if not rectangles:
                alternate = needle.replace(".", "")
                if alternate != needle:
                    rectangles = page.search_for(alternate)
            if not rectangles:
                continue
            caption = rectangles[0]
            page_rect = page.rect
            height = min(max(page_rect.height * 0.48, 220), 520)
            top = max(page_rect.y0, caption.y0 - height)
            clip = fitz.Rect(
                page_rect.x0,
                top,
                page_rect.x1,
                min(page_rect.y1, caption.y1 + 45),
            )
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            caption_text = page.get_textbox(
                fitz.Rect(
                    page_rect.x0,
                    max(page_rect.y0, caption.y0 - 10),
                    page_rect.x1,
                    min(page_rect.y1, caption.y1 + 120),
                )
            ).strip()
            return encoded, page_index + 1, caption_text[:2000]
        raise ValueError(
            "The figure label was not found as selectable text. "
            "Provide the exact caption label or an OCR-enabled PDF."
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
    metadata = _document_metadata(doc_id, owner_id)
    storage_path = str(metadata.get("storage_path") or "")
    if not storage_path:
        return _json(
            VisualEntailmentResult(
                claim_text=claim_text,
                figure_id=figure_id,
                verdict=EntailmentVerdict.INSUFFICIENT,
                rationale="The indexed document has no retained owner-scoped PDF path.",
                confidence=1.0,
                evidence_note="No image could be extracted.",
            ).model_dump()
        )
    try:
        image_b64, page_number, caption_text = _extract_figure_region(storage_path, figure_id)
    except Exception as exc:
        return _json(
            VisualEntailmentResult(
                claim_text=claim_text,
                figure_id=figure_id,
                verdict=EntailmentVerdict.INSUFFICIENT,
                rationale=str(exc),
                confidence=1.0,
                evidence_note="Visual evidence was not available.",
            ).model_dump()
        )
    citation = _document_citation(
        metadata,
        doc_id=doc_id,
        snippet=caption_text or f"Figure region for {figure_id}",
        page_number=page_number,
        source_id=f"{doc_id}:page:{page_number}:{figure_id}",
    )
    if client is None:
        result = VisualEntailmentResult(
            claim_text=claim_text,
            figure_id=figure_id,
            verdict=EntailmentVerdict.INSUFFICIENT,
            rationale="A vision-capable model is not configured.",
            confidence=1.0,
            page_number=page_number,
            evidence_note=caption_text or None,
        )
        payload = result.model_dump()
        payload["citations"] = [citation.model_dump(exclude_none=True)]
        return _json(payload)
    prompt = (
        "Evaluate only whether the supplied figure region supports the claim. "
        "Return JSON with claim_text, figure_id, verdict "
        "(supports|contradicts|insufficient|uncertain), rationale, confidence. "
        "Do not infer details that are not visible. Caption text: "
        f"{caption_text[:2000]}"
    )
    user_content = [
        {"type": "text", "text": f"Claim: {claim_text}\nFigure label: {figure_id}\n{prompt}"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
        },
    ]
    try:
        raw = _completion(
            client,
            model=model,
            system="You are a conservative scientific figure reviewer.",
            user=user_content,
            max_tokens=700,
            json_mode=True,
        )
        parsed = _parse_json_object(raw)
        parsed.update({
            "claim_text": claim_text,
            "figure_id": figure_id,
            "page_number": page_number,
            "evidence_note": caption_text or None,
        })
        result = VisualEntailmentResult(**parsed)
    except Exception as exc:
        result = VisualEntailmentResult(
            claim_text=claim_text,
            figure_id=figure_id,
            verdict=EntailmentVerdict.UNCERTAIN,
            rationale=f"Vision analysis failed: {type(exc).__name__}.",
            confidence=0.0,
            page_number=page_number,
            evidence_note=caption_text or None,
        )
    payload = result.model_dump()
    payload["citations"] = [citation.model_dump(exclude_none=True)]
    return _json(payload)


def _fallback_protocol(text: str, doc_id: str) -> Protocol:
    steps: List[ProtocolStep] = []
    for sentence in re.split(r"(?<=[.;])\s+|\n+", text or ""):
        sentence = sentence.strip(" -\t")
        if len(sentence) < 8:
            continue
        has_action = bool(re.search(
            r"\b(add|mix|incubat|wash|centrifug|heat|cool|transfer|measure|"
            r"dilut|prepare|collect|filter|dry|stir|pipett|resuspend)\w*\b",
            sentence,
            flags=re.IGNORECASE,
        ))
        if not has_action:
            continue
        temperature = next(
            iter(re.findall(r"-?\d+(?:\.\d+)?\s*°?\s*[CFK]\b", sentence, re.I)),
            None,
        )
        duration = next(
            iter(re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:s|sec|seconds?|min|minutes?|h|hours?)\b",
                sentence,
                re.I,
            )),
            None,
        )
        steps.append(ProtocolStep(description=sentence[:1000], temperature=temperature, time=duration))
        if len(steps) >= 100:
            break
    warnings = []
    if not steps:
        warnings.append(
            "No explicit procedural steps were detected; prose was not converted into invented steps."
        )
    return Protocol(
        steps=steps,
        metadata={"source_doc": doc_id, "extraction_method": "conservative_regex"},
        warnings=warnings,
    )


def extract_protocol(
    text: str,
    doc_id: str = "",
    *,
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    text = (text or "").strip()
    if not text:
        return _json(
            Protocol(
                steps=[],
                metadata={"source_doc": doc_id, "extraction_method": "none"},
                warnings=["No methods text was supplied."],
            ).model_dump()
        )
    if client is None:
        return _json(_fallback_protocol(text, doc_id).model_dump())
    try:
        raw = _completion(
            client,
            model=model,
            system=(
                "Extract only explicit procedural steps from methods text. Return JSON "
                "with steps (description, temperature, time, reagent, notes), metadata, "
                "and warnings. Never fill missing details from general knowledge."
            ),
            user=text[:30_000],
            max_tokens=1800,
            json_mode=True,
        )
        parsed = _parse_json_object(raw)
        parsed["metadata"] = {
            **dict(parsed.get("metadata") or {}),
            "source_doc": doc_id,
            "extraction_method": "llm_explicit_only",
        }
        return _json(Protocol(**parsed).model_dump())
    except Exception:
        return _json(_fallback_protocol(text, doc_id).model_dump())


def run_scientific_debate(
    claim: str,
    context: str,
    *,
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    claim = (claim or "").strip()
    context = (context or "").strip()
    if not context:
        return _json(
            DebateResult(
                verdict="insufficient evidence",
                key_issues=["No evidence context was supplied."],
                recommended_followups=["Retrieve primary sources before debating the claim."],
                uncertainty="No evidence basis.",
            ).model_dump()
        )
    if client is None:
        return _json(
            DebateResult(
                verdict="model unavailable",
                key_issues=["A language-model client is required for the structured debate."],
                supporting_evidence=[context[:1500]],
                recommended_followups=["Review the supplied context manually."],
                uncertainty="No automated synthesis was performed.",
            ).model_dump()
        )
    advocate = _completion(
        client,
        model=model,
        system=(
            "Act as an advocate. Use only the supplied evidence. Identify the "
            "strongest support, explicitly noting missing evidence."
        ),
        user=f"Claim: {claim}\n\nEvidence:\n{context[:24_000]}",
        max_tokens=900,
    )
    skeptic = _completion(
        client,
        model=model,
        system=(
            "Act as a skeptical reviewer. Use only the original evidence and the "
            "advocate argument. Identify alternative explanations, bias, and gaps."
        ),
        user=(
            f"Claim: {claim}\n\nOriginal evidence:\n{context[:18_000]}"
            f"\n\nAdvocate:\n{advocate[:6000]}"
        ),
        max_tokens=900,
    )
    try:
        judge_raw = _completion(
            client,
            model=model,
            system=(
                "Act as an impartial scientific judge. Return JSON with verdict, "
                "key_issues, supporting_evidence, recommended_followups, uncertainty. "
                "The original evidence is authoritative; generated arguments are not evidence."
            ),
            user=(
                f"Claim: {claim}\n\nOriginal evidence:\n{context[:18_000]}"
                f"\n\nAdvocate:\n{advocate[:5000]}\n\nSkeptic:\n{skeptic[:5000]}"
            ),
            max_tokens=1200,
            json_mode=True,
        )
        result = DebateResult(**_parse_json_object(judge_raw))
    except Exception as exc:
        result = DebateResult(
            verdict="uncertain",
            key_issues=[advocate[:1000], skeptic[:1000]],
            recommended_followups=["Manually inspect the original evidence."],
            uncertainty=f"Judge synthesis failed: {type(exc).__name__}.",
        )
    payload = result.model_dump()
    payload["advocate"] = advocate
    payload["skeptic"] = skeptic
    return _json(payload)


def _retrieve_document_evidence(
    doc_id: str,
    query: str,
    *,
    owner_id: str,
    n_results: int = 4,
) -> Tuple[List[Chunk], List[Citation]]:
    rag = get_rag_layer()
    chunks = rag.query(query, n_results=n_results, owner_id=owner_id, doc_id=doc_id)
    citations: List[Citation] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        page_number = metadata.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            page_number = None
        citations.append(
            Citation(
                label=f"[{index + 1}]",
                title=str(metadata.get("filename") or doc_id),
                url=f"local://{doc_id}",
                source_type="uploaded_document",
                snippet=str(metadata.get("parent_text") or chunk.text),
                quote=chunk.text,
                source_id=chunk.id,
                doc_id=doc_id,
                chunk_id=chunk.id,
                page_number=page_number,
            )
        )
    return chunks, citations


def compare_papers(
    doc_ids: Sequence[str],
    query: str,
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    unique_ids = list(dict.fromkeys(str(value) for value in doc_ids if str(value).strip()))
    if not 2 <= len(unique_ids) <= 10:
        raise ValueError("compare_papers requires between 2 and 10 unique document IDs.")
    all_citations: List[Citation] = []
    contexts: List[str] = []
    gaps: List[str] = []
    label_counter = 1
    for doc_id in unique_ids:
        chunks, citations = _retrieve_document_evidence(doc_id, query, owner_id=owner_id, n_results=4)
        if not chunks:
            gaps.append(doc_id)
            continue
        relabelled = []
        for citation in citations:
            citation.label = f"[{label_counter}]"
            label_counter += 1
            all_citations.append(citation)
            relabelled.append(citation)
        contexts.append(
            f"DOCUMENT {doc_id}\n"
            + "\n\n".join(
                f"{citation.label} {citation.quote or citation.snippet or ''}"
                for citation in relabelled
            )
        )
    if gaps:
        payload = ComparisonResult(
            summary="Comparison was not generated because evidence was missing for one or more documents.",
            evidence_gaps=gaps,
        ).model_dump()
        payload["citations"] = [item.model_dump(exclude_none=True) for item in all_citations]
        return _json(payload)
    if client is None:
        payload = ComparisonResult(
            summary="Evidence was retrieved, but no model is configured for narrative synthesis.",
            evidence_gaps=[],
        ).model_dump()
        payload["evidence"] = contexts
        payload["citations"] = [item.model_dump(exclude_none=True) for item in all_citations]
        return _json(payload)
    raw = _completion(
        client,
        model=model,
        system=(
            "Compare the supplied documents using only labelled evidence. Return JSON "
            "with consistencies, conflicts, trends, summary, evidence_gaps. Cite labels "
            "inside every substantive item. Do not compare missing facts."
        ),
        user=f"Comparison question: {query}\n\n" + "\n\n".join(contexts)[:45_000],
        max_tokens=1800,
        json_mode=True,
    )
    try:
        result = ComparisonResult(**_parse_json_object(raw))
    except (ValidationError, ValueError, json.JSONDecodeError):
        result = ComparisonResult(
            summary="The comparison model returned an invalid structured response.",
            evidence_gaps=[],
        )
    payload = result.model_dump()
    payload["citations"] = [item.model_dump(exclude_none=True) for item in all_citations]
    return _json(payload)


def generate_comparison_matrix(
    doc_ids: Sequence[str],
    metrics: Sequence[str],
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    unique_docs = list(dict.fromkeys(str(value) for value in doc_ids if str(value).strip()))
    unique_metrics = list(dict.fromkeys(str(value) for value in metrics if str(value).strip()))
    if not 1 <= len(unique_docs) <= 10:
        raise ValueError("The matrix supports 1-10 documents.")
    if not 1 <= len(unique_metrics) <= 12:
        raise ValueError("The matrix supports 1-12 metrics.")
    all_citations: List[Citation] = []
    evidence_blocks: List[str] = []
    gaps: List[str] = []
    label_counter = 1
    query = " ; ".join(unique_metrics)
    for doc_id in unique_docs:
        chunks, citations = _retrieve_document_evidence(
            doc_id,
            query,
            owner_id=owner_id,
            n_results=min(8, len(unique_metrics) + 2),
        )
        if not chunks:
            gaps.append(doc_id)
            continue
        labelled_lines = []
        for citation in citations:
            citation.label = f"[{label_counter}]"
            label_counter += 1
            all_citations.append(citation)
            labelled_lines.append(f"{citation.label} {citation.quote or citation.snippet or ''}")
        evidence_blocks.append(f"DOCUMENT {doc_id}\n" + "\n".join(labelled_lines))
    if gaps:
        return _json({
            "markdown": "",
            "evidence_gaps": gaps,
            "error": "Matrix generation stopped because one or more documents had no evidence.",
            "citations": [item.model_dump(exclude_none=True) for item in all_citations],
        })
    if client is None:
        return _json({
            "markdown": "",
            "evidence_gaps": [],
            "error": "Evidence was retrieved, but no model is configured to extract matrix values.",
            "evidence": evidence_blocks,
            "citations": [item.model_dump(exclude_none=True) for item in all_citations],
        })
    raw = _completion(
        client,
        model=model,
        system=(
            "Build one Markdown comparison table from labelled evidence. Rows are the "
            "requested metrics and columns are document IDs. Every non-missing value "
            "must include at least one [n] citation. Use 'Not reported' when absent. "
            "Return JSON with markdown and evidence_gaps."
        ),
        user=(
            f"Documents: {unique_docs}\nMetrics: {unique_metrics}\n\n"
            + "\n\n".join(evidence_blocks)[:45_000]
        ),
        max_tokens=2000,
        json_mode=True,
    )
    try:
        parsed = _parse_json_object(raw)
    except Exception:
        parsed = {"markdown": "", "evidence_gaps": [], "error": "The matrix model returned invalid JSON."}
    parsed["citations"] = [item.model_dump(exclude_none=True) for item in all_citations]
    return _json(parsed)


def detect_conflicts(
    topic: str,
    context: str,
    *,
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    context = (context or "").strip()
    if not context:
        return _json({
            "topic": topic,
            "conflicts": [],
            "synthesis": "No evidence context was supplied.",
            "evidence_gaps": ["context"],
        })
    if client is None:
        return _json({
            "topic": topic,
            "conflicts": [],
            "synthesis": "No model is configured; no automated conflict inference was performed.",
            "evidence_excerpt": context[:3000],
        })
    raw = _completion(
        client,
        model=model,
        system=(
            "Identify only direct, evidence-visible contradictions. Return JSON with "
            "topic, conflicts (claim_a, claim_b, source_a, source_b, conflict_type), "
            "synthesis, and evidence_gaps. Do not label different populations or "
            "conditions as contradictions without explaining the distinction."
        ),
        user=f"Topic: {topic}\n\nEvidence:\n{context[:35_000]}",
        max_tokens=1600,
        json_mode=True,
    )
    try:
        return _json(_parse_json_object(raw))
    except Exception:
        return _json({
            "topic": topic,
            "conflicts": [],
            "synthesis": "The conflict-analysis model returned invalid structured output.",
        })


def extract_limitations(
    doc_id: str,
    text: str = "",
    *,
    owner_id: str = "default_user",
    client: Optional[Any] = None,
    model: str = "gpt-4o",
) -> str:
    citations: List[Citation] = []
    source_text = (text or "").strip()
    if not source_text and doc_id:
        chunks, citations = _retrieve_document_evidence(
            doc_id,
            "limitations caveats threats to validity scope constraints future work",
            owner_id=owner_id,
            n_results=8,
        )
        source_text = "\n\n".join(chunk.text for chunk in chunks)
    if not source_text:
        return _json({
            "doc_id": doc_id,
            "limitations": [],
            "recommendation": "No limitation evidence was supplied or retrieved.",
            "citations": [],
        })
    if client is None:
        explicit = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", source_text)
            if re.search(
                r"\b(limit(?:ation|ed)?|caveat|bias|uncertain|future work|"
                r"cannot|could not|may not|threat to validity)\b",
                sentence,
                flags=re.IGNORECASE,
            )
        ][:30]
        return _json({
            "doc_id": doc_id,
            "limitations": explicit,
            "recommendation": "Review the cited passages manually.",
            "extraction_method": "explicit_phrase_filter",
            "citations": [item.model_dump(exclude_none=True) for item in citations],
        })
    raw = _completion(
        client,
        model=model,
        system=(
            "Extract only explicit limitations, caveats, exclusions, scope constraints, "
            "and threats to validity. Return JSON with doc_id, limitations, recommendation. "
            "Do not infer generic limitations absent from the text."
        ),
        user=source_text[:35_000],
        max_tokens=1400,
        json_mode=True,
    )
    try:
        parsed = _parse_json_object(raw)
    except Exception:
        parsed = {
            "doc_id": doc_id,
            "limitations": [],
            "recommendation": "The limitations model returned invalid structured output.",
        }
    parsed["doc_id"] = doc_id
    parsed["citations"] = [item.model_dump(exclude_none=True) for item in citations]
    return _json(parsed)
