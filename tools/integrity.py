"""Scientific integrity tools: visual entailment, protocol extraction, debate, comparison, conflict detection."""

import json
import os
import re
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from tools.rag import get_rag_layer

# Initialize global client if possible
_client = None
if OpenAI is not None:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        _client = OpenAI(api_key=_api_key)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EntailmentVerdict(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    UNCERTAIN = "uncertain"
    INSUFFICIENT = "insufficient"


class VisualEntailmentResult(BaseModel):
    claim_text: str
    figure_id: str
    verdict: EntailmentVerdict
    rationale: str
    confidence: float


class ProtocolStep(BaseModel):
    description: str
    temperature: Optional[str] = None
    time: Optional[str] = None
    reagent: Optional[str] = None
    notes: Optional[str] = None


class Protocol(BaseModel):
    steps: List[ProtocolStep]
    metadata: Dict[str, Any]


class DebateResult(BaseModel):
    verdict: str
    key_issues: List[str]
    supporting_evidence: List[str]
    recommended_followups: List[str]


class ComparisonResult(BaseModel):
    consistencies: List[str]
    conflicts: List[str]
    trends: List[str]
    summary: str


# ---------------------------------------------------------------------------
# Private helpers for visual entailment
# ---------------------------------------------------------------------------


def _get_file_path_for_doc(doc_id: str) -> Optional[str]:
    """
    Retrieve the original file_path for an ingested document by querying the
    ChromaDB metadata store.  Returns None if the document is not found.
    """
    try:
        rag = get_rag_layer()
        results = rag.collection.get(
            where={"doc_id": doc_id},
            limit=1,
            include=["metadatas"],
        )
        metadatas = results.get("metadatas") or []
        if metadatas:
            return metadatas[0].get("file_path")
    except Exception:
        pass
    return None


def _extract_figure_image_b64(file_path: str, figure_id: str) -> Optional[str]:
    """
    Locate a specific figure in a PDF by its label (e.g. "Fig 1", "Figure 2",
    "Fig. 3a") and return the image as a base64-encoded JPEG string.

    Three extraction strategies (tried in order for the matching page):
      1. Extract the first embedded raster image on the page.
      2. Render a clipped pixmap around the figure's caption text bounding box.
      3. Render the full page as a last resort.

    Returns None when the PDF cannot be opened, or the figure label is not found.
    """
    import base64

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None

    if not os.path.isfile(file_path):
        return None

    # Build a flexible regex: "Fig 1" → matches "Fig 1", "Fig. 1", "Figure 1", "FIG1" etc.
    label_core = re.sub(r"\s+", r"\\s*", figure_id.strip())
    label_core = label_core.replace(".", "\\.?")
    pattern = re.compile(label_core, re.IGNORECASE)

    try:
        doc = fitz.open(file_path)
    except Exception:
        return None

    try:
        for page in doc:
            page_text = page.get_text()
            if not pattern.search(page_text):
                continue

            page_w = page.rect.width
            page_h = page.rect.height

            # Strategy 1: embedded raster image on this page
            images = page.get_images(full=True)
            if images:
                xref = images[0][0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    return base64.b64encode(img_bytes).decode("utf-8")
                except Exception:
                    pass

            # Strategy 2: rendered pixmap around caption text
            hits = page.search_for(figure_id)
            if not hits:
                # Try just first word of figure_id (e.g., "Fig")
                hits = page.search_for(figure_id.split()[0])

            if hits:
                rect = hits[0]
                # Capture 300 pts above (figure body) and 80 pts below (caption)
                capture = fitz.Rect(
                    0,
                    max(0.0, rect.y0 - 300.0),
                    page_w,
                    min(page_h, rect.y1 + 80.0),
                )
                try:
                    pix = page.get_pixmap(clip=capture, dpi=150)
                    return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
                except Exception:
                    pass

            # Strategy 3: full page render (last resort)
            try:
                pix = page.get_pixmap(dpi=100)
                return base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
            except Exception:
                pass
    finally:
        doc.close()

    return None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def check_visual_entailment(claim_text: str, figure_id: str, doc_id: str) -> str:
    """
    Verify whether a scientific claim is supported by a specific figure.

    Extracts the figure from the ingested PDF using PyMuPDF and passes the
    real image to GPT-4o Vision for evaluation.  If the figure cannot be
    located, returns an INSUFFICIENT verdict rather than hallucinating.
    """
    if not _client:
        return json.dumps({"error": "Vision client not available."})

    # --- Step 1: Locate and extract the figure image ---
    file_path = _get_file_path_for_doc(doc_id)
    img_b64: Optional[str] = None
    if file_path:
        img_b64 = _extract_figure_image_b64(file_path, figure_id)

    if not img_b64:
        result = VisualEntailmentResult(
            claim_text=claim_text,
            figure_id=figure_id,
            verdict=EntailmentVerdict.INSUFFICIENT,
            rationale=(
                f"Figure '{figure_id}' could not be located or extracted from "
                f"document '{doc_id}'.  The document may not be a PDF, the "
                "figure label was not found on any page, or image extraction "
                "failed.  Please verify the document has been ingested and the "
                "figure label matches exactly (e.g., 'Fig 1', 'Figure 2')."
            ),
            confidence=0.0,
        )
        return result.model_dump_json()

    # --- Step 2: GPT-4o Vision evaluation ---
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"You are evaluating a scientific figure for a specific claim.\n\n"
                f"Claim: {claim_text}\n\n"
                f"Figure reference: {figure_id}\n\n"
                "Based solely on the image provided, determine whether the claim is:\n"
                "- 'support': The figure clearly supports the claim.\n"
                "- 'contradict': The figure clearly contradicts the claim.\n"
                "- 'uncertain': The figure is related but the evidence is ambiguous.\n"
                "- 'insufficient': The figure does not contain relevant data for the claim.\n\n"
                "Output MUST be valid JSON with keys: claim_text (str), figure_id (str), "
                "verdict (one of support|contradict|uncertain|insufficient), "
                "rationale (str, cite specific visual features), confidence (float 0.0–1.0)."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high",
            },
        },
    ]

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous scientific image analyst.  Evaluate "
                        "figures objectively and produce structured JSON verdicts."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.1,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        result = VisualEntailmentResult(
            claim_text=data.get("claim_text", claim_text),
            figure_id=data.get("figure_id", figure_id),
            verdict=EntailmentVerdict(data.get("verdict", "uncertain")),
            rationale=data.get("rationale", "No rationale provided."),
            confidence=float(data.get("confidence", 0.5)),
        )
        return result.model_dump_json()
    except Exception as e:
        return json.dumps(
            {"error": str(e), "doc_id": doc_id, "figure_id": figure_id}
        )


def extract_protocol(text: str, doc_id: str = "") -> str:
    """
    Extract a structured wet-lab protocol from a methods section.

    Uses GPT-4o with a strict JSON schema for accurate step-by-step extraction.
    Falls back to a robust regex-based extractor when no LLM client is available.
    """

    def _regex_fallback(text: str, doc_id: str) -> str:
        steps: List[ProtocolStep] = []

        # Strategy 1: split on numbered-list delimiters (both newline and inline)
        # Handles: "1. Step\n2. Step" and "1. Step 2. Step 3. Step"
        raw_parts = re.split(r"(?:^|\s)(?=\d+[.)]\s)", text.strip())
        numbered_parts = [
            re.sub(r"^\d+[.)]\s+", "", p).strip()
            for p in raw_parts
            if re.match(r"\d+[.)]\s", p.strip())
        ]
        if len(numbered_parts) > 1:
            for desc_raw in numbered_parts:
                desc = desc_raw.replace("\n", " ")[:500]
                if len(desc) < 10:
                    continue
                temp_m = re.search(r"(\d+\s*°?\s*[Cc])", desc)
                time_m = re.search(
                    r"(\d+\s*(?:min(?:utes?)?|hours?|secs?|seconds?|h\b))",
                    desc,
                    re.IGNORECASE,
                )
                steps.append(
                    ProtocolStep(
                        description=desc,
                        temperature=temp_m.group(1) if temp_m else None,
                        time=time_m.group(1) if time_m else None,
                    )
                )

        # Strategy 2: bulleted list "- ...", "• ..."
        if not steps:
            bulleted = re.findall(
                r"(?:^|\n)\s*[•\-*]\s+(.+?)(?=\n\s*[•\-*]|\Z)",
                text,
                re.DOTALL,
            )
            for raw in bulleted:
                desc = raw.strip().replace("\n", " ")[:500]
                if len(desc) >= 10:
                    steps.append(ProtocolStep(description=desc))

        # Strategy 3: sentence-level fallback
        if not steps:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            steps = [
                ProtocolStep(description=s.strip()[:500])
                for s in sentences[:15]
                if len(s.strip()) > 10
            ]

        if not steps:
            steps = [ProtocolStep(description=text[:600])]

        return Protocol(
            steps=steps,
            metadata={
                "source_doc": doc_id,
                "step_count": len(steps),
                "extraction_method": "regex_fallback",
            },
        ).model_dump_json()


    # LLM path
    if not _client:
        return _regex_fallback(text, doc_id)

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific protocol extractor. "
                        "Given a methods section, extract every procedural step. "
                        "For each step capture: description (required), "
                        "temperature (e.g. '37°C' or null), "
                        "time (e.g. '10 minutes' or null), "
                        "reagent (primary chemical/biological reagent or null), "
                        "notes (warnings, tips, alternatives or null). "
                        "Output MUST be valid JSON:\n"
                        '{"steps": [{"description":"...","temperature":"...","time":"...",'
                        '"reagent":"...","notes":"..."}], '
                        '"metadata": {"source_doc":"...","step_count":N,"extraction_method":"llm"}}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract the complete structured protocol from this methods section:\n\n"
                        f"{text[:8000]}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)

        raw_steps = data.get("steps", [])
        steps: List[ProtocolStep] = []
        for s in raw_steps:
            if not isinstance(s, dict):
                continue
            desc = str(s.get("description", "")).strip()
            if not desc:
                continue
            steps.append(
                ProtocolStep(
                    description=desc,
                    temperature=s.get("temperature") or None,
                    time=s.get("time") or None,
                    reagent=s.get("reagent") or None,
                    notes=s.get("notes") or None,
                )
            )

        if not steps:
            return _regex_fallback(text, doc_id)

        meta: Dict[str, Any] = data.get("metadata", {})
        meta["source_doc"] = doc_id
        meta["step_count"] = len(steps)
        meta.setdefault("extraction_method", "llm")

        return Protocol(steps=steps, metadata=meta).model_dump_json()

    except Exception:
        return _regex_fallback(text, doc_id)


def run_scientific_debate(claim: str, context: str) -> str:
    """
    Runs a 3-agent adversarial debate (Advocate → Skeptic → Judge) to evaluate
    the scientific merit of a claim against the provided evidence context.
    """
    if not _client:
        return json.dumps({"error": "OpenAI client not initialized for debate."})

    model = "gpt-4o"

    adv_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific Advocate.  Find every piece of evidence "
                    "in the context that supports the claim.  Be persuasive but grounded."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nClaim: {claim}"},
        ],
    )
    advocate_argument = adv_resp.choices[0].message.content or ""

    skep_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific Skeptic.  Identify weaknesses, missing "
                    "controls, or contradictory data in the context regarding the "
                    "claim.  Challenge the Advocate's points rigorously."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Claim: {claim}\n\n"
                    f"Advocate said:\n{advocate_argument}"
                ),
            },
        ],
    )
    skeptic_argument = skep_resp.choices[0].message.content or ""

    judge_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Scientific Judge.  Review the Advocate and Skeptic "
                    "arguments and produce a final, balanced verdict.  Output MUST "
                    "be valid JSON: {verdict: str, key_issues: list[str], "
                    "supporting_evidence: list[str], recommended_followups: list[str]}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\n"
                    f"Advocate:\n{advocate_argument}\n\n"
                    f"Skeptic:\n{skeptic_argument}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    return judge_resp.choices[0].message.content or "{}"


def compare_papers(doc_ids: List[str], query: str) -> str:
    """
    Compares multiple ingested papers on a specific query via targeted RAG
    retrieval and GPT-4o synthesis.
    """
    if not _client:
        return json.dumps({"error": "OpenAI client not initialized for comparison."})

    rag = get_rag_layer()
    contexts: List[str] = []
    for doc_id in doc_ids:
        chunks = rag.query(query, n_results=3, where={"doc_id": doc_id})
        doc_text = "\n".join(c.text for c in chunks)
        contexts.append(f"Document {doc_id}:\n{doc_text}")

    context_str = "\n\n---\n\n".join(contexts)

    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert researcher comparing multiple documents. "
                    "Identify consistencies, direct conflicts, emerging trends and "
                    "provide a high-level synthesis.  Output MUST be valid JSON: "
                    "{consistencies: list[str], conflicts: list[str], "
                    "trends: list[str], summary: str}"
                ),
            },
            {"role": "user", "content": f"Query: {query}\n\n{context_str}"},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def generate_comparison_matrix(doc_ids: List[str], metrics: List[str]) -> str:
    """
    Generates a Markdown table comparing specific metrics across multiple papers
    using targeted RAG extraction + GPT-4o-mini value extraction per cell.
    """
    rag = get_rag_layer()
    header = "| Metric | " + " | ".join(doc_ids) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(doc_ids)) + " |"

    rows: List[str] = []
    for metric in metrics:
        row_vals: List[str] = []
        for doc_id in doc_ids:
            chunks = rag.query(metric, n_results=1, where={"doc_id": doc_id})
            if chunks:
                if _client:
                    ext = _client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Extract the specific metric value as concisely as possible "
                                    "(e.g. '150 samples', '98% Accuracy'). "
                                    "Return 'N/A' if not found in the text."
                                ),
                            },
                            {
                                "role": "user",
                                "content": f"Metric: {metric}\n\nText: {chunks[0].text}",
                            },
                        ],
                        max_tokens=50,
                    )
                    row_vals.append(
                        (ext.choices[0].message.content or "N/A").strip()
                    )
                else:
                    row_vals.append(chunks[0].text[:30] + "…")
            else:
                row_vals.append("N/A")

        rows.append("| " + metric + " | " + " | ".join(row_vals) + " |")

    return "\n".join([header, divider] + rows)


def detect_conflicts(topic: str, context: str) -> str:
    """
    Identifies contradictory claims within the provided context about a topic.
    """
    if not _client:
        return json.dumps({"topic": topic, "error": "LLM client not available."})

    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific conflict detector.  Identify every direct "
                    "contradiction or inconsistency in the provided context for the "
                    "given topic. Name sources when multiple are cited.  Output MUST "
                    "be valid JSON: {topic: str, conflicts: list[{claim_a: str, "
                    "claim_b: str, source_a: str, source_b: str}], synthesis: str}"
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nContext:\n{context}"},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def extract_limitations(doc_id: str, text: str) -> str:
    """
    Extracts explicit limitations, disclaimers, and scope constraints from a paper.
    """
    if not _client:
        return json.dumps({"doc_id": doc_id, "error": "LLM client not available."})

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract explicit limitations, disclaimers, or scope constraints "
                    "from the provided research text.  Be thorough.  Output MUST be "
                    "valid JSON: {doc_id: str, limitations: list[str], recommendation: str}"
                ),
            },
            {"role": "user", "content": f"Text:\n{text}"},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    data["doc_id"] = doc_id  # Ensure correctness
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Tool definition schemas for OpenAI function calling
# ---------------------------------------------------------------------------

VISUAL_ENTAILMENT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "check_visual_entailment",
        "description": (
            "Check if a paper's claim is supported by a specific figure. "
            "Extracts the actual figure from the PDF and evaluates it with GPT-4o Vision."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string", "description": "The specific claim from the text."},
                "figure_id": {"type": "string", "description": "The figure label, e.g. 'Fig 1' or 'Figure 3b'."},
                "doc_id": {"type": "string", "description": "The document ID containing the figure."},
            },
            "required": ["claim_text", "figure_id", "doc_id"],
        },
    },
}

PROTOCOL_EXTRACTION_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "extract_protocol",
        "description": "Extract a structured wet-lab protocol from a methods section.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The raw text of the Methods section."},
                "doc_id": {"type": "string", "description": "Source document ID."},
            },
            "required": ["text"],
        },
    },
}

DEBATE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_scientific_debate",
        "description": "Run an adversarial 3-agent debate (Advocate, Skeptic, Judge) to evaluate a scientific claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The main claim to debate."},
                "context": {"type": "string", "description": "Summary of evidence or full text to analyze."},
            },
            "required": ["claim", "context"],
        },
    },
}

COMPARISON_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "compare_papers",
        "description": "Compare results and methodologies across multiple papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "List of document IDs to compare."},
                "query": {"type": "string", "description": "Specific topic to compare across papers."},
            },
            "required": ["doc_ids", "query"],
        },
    },
}

MATRIX_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "generate_comparison_matrix",
        "description": "Generate a Markdown table comparing papers across specific metrics (e.g. Sample Size, Accuracy).",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["doc_ids", "metrics"],
        },
    },
}

CONFLICT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "detect_conflicts",
        "description": "Hunt for contradictory claims between sources on a specific topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["topic", "context"],
        },
    },
}

LIMITATIONS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "extract_limitations",
        "description": "Extract the Limitations and Disclaimers from a paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["doc_id", "text"],
        },
    },
}
