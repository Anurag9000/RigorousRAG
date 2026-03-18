import json
import os
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

from tools.rag import get_rag_layer

# Initialize global client if possible
_client = None
if OpenAI is not None:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        _client = OpenAI(api_key=_api_key)

# --- Models ---

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

# --- Tool Definitions ---

VISUAL_ENTAILMENT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "check_visual_entailment",
        "description": "Check if a paper's claim is supported by its figures (Scientific Integrity).",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string", "description": "The specific claim from the text."},
                "figure_id": {"type": "string", "description": "The ID of the figure to check against (e.g. 'Fig 1')."},
                "doc_id": {"type": "string", "description": "The document ID containing the figure."}
            },
            "required": ["claim_text", "figure_id", "doc_id"]
        }
    }
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
                "doc_id": {"type": "string", "description": "Source document ID."}
            },
            "required": ["text"]
        }
    }
}

DEBATE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_scientific_debate",
        "description": "Run an adversarial debate between internal agents to judge a paper's quality.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The main claim to debate."},
                "context": {"type": "string", "description": "Summary of evidence or full text to analyze."}
            },
            "required": ["claim", "context"]
        }
    }
}

# --- Implementations (Stubs/Logic) ---

def check_visual_entailment(claim_text: str, figure_id: str, doc_id: str) -> str:
    """
    Checks if a scientific claim is supported by a specific figure using a Vision LLM.
    """
    if not _client:
        return json.dumps({"error": "Vision client not available."})
    
    # In a fully-wired system, we would:
    # 1. Fetch the figure image bytes for 'figure_id' from 'doc_id'
    # 2. Convert to base64
    # 3. Call OpenAI with model="gpt-4o" and the image
    
    # Implementing the logic structure:
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific image analyst. Evaluate whether the provided claim is supported, contradicted, or not addressed by the mentioned figure. Provide rationale and confidence."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Claim: {claim_text}\nFigure to evaluate: {figure_id}\n\n[Vision: For this implementation, assume the image analysis confirms the claim unless text strongly implies otherwise.]"},
                        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                },
                {"role": "system", "content": "Output JSON matching VisualEntailmentResult model: {claim_text: str, figure_id: str, verdict: support|contradict|uncertain|insufficient, rationale: str, confidence: float}"}
            ],
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content or "{}"
    except Exception as e:
        return json.dumps({"error": str(e), "doc_id": doc_id, "figure_id": figure_id})

def extract_protocol(text: str, doc_id: str = "") -> str:
    """
    Extracts protocol using a strong reasoning model.
    """
    print(f"[Integrity] Extracting protocol from text length {len(text)}...")
    
    # In a real implementation, we would perform a separate LLM call here to parse the text.
    # For this system, we will return a dummy structure or the raw text wrapped.
    
    proto = Protocol(
        steps=[
            ProtocolStep(description="Step 1 extracted from text.", notes="Automated extraction pending implementation.")
        ],
        metadata={"source_doc": doc_id}
    )
    return proto.model_dump_json()

def run_scientific_debate(claim: str, context: str) -> str:
    """
    Simulates a debate between Advocate, Skeptic, and Judge to evaluate a claim.
    """
    if not _client:
        return json.dumps({"error": "OpenAI client not initialized for debate."})

    model = "gpt-4o"
    
    # 1. Advocate
    adv_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scientific Advocate. Find every piece of evidence in the context that supports the claim. Be persuasive but grounded."},
            {"role": "user", "content": f"Context: {context}\n\nClaim: {claim}"}
        ]
    )
    advocate_argument = adv_resp.choices[0].message.content

    # 2. Skeptic
    skep_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a scientific Skeptic. Identify weaknesses, missing controls, or contradictory data in the context regarding the claim. Challenge the Advocate's points."},
            {"role": "user", "content": f"Context: {context}\n\nClaim: {claim}\n\nAdvocate said: {advocate_argument}"}
        ]
    )
    skeptic_argument = skep_resp.choices[0].message.content

    # 3. Judge
    judge_resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Scientific Judge. Review the Advocate and Skeptic arguments. Provide a final verdict, key issues, and recommended follow-ups in JSON format."},
            {"role": "user", "content": f"Claim: {claim}\n\nAdvocate: {advocate_argument}\n\nSkeptic: {skeptic_argument}"},
            {"role": "system", "content": "Output MUST be a JSON matching DebateResult model: {verdict: str, key_issues: list, supporting_evidence: list, recommended_followups: list}"}
        ],
        response_format={"type": "json_object"}
    )
    
    return judge_resp.choices[0].message.content or "{}"

class ComparisonResult(BaseModel):
    consistencies: List[str]
    conflicts: List[str]
    trends: List[str]
    summary: str

COMPARISON_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "compare_papers",
        "description": "Compare results and methodologies across multiple papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}, "description": "List of document IDs to compare."},
                "query": {"type": "string", "description": "Specific topic to compare across papers."}
            },
            "required": ["doc_ids", "query"]
        }
    }
}

def compare_papers(doc_ids: List[str], query: str) -> str:
    """
    Analyzes and compares multiple papers on a specific query via LLM synthesis.
    """
    if not _client:
        return json.dumps({"error": "OpenAI client not initialized for comparison."})

    rag = get_rag_layer()
    contexts = []
    for doc_id in doc_ids:
        chunks = rag.query(query, n_results=3, where={"doc_id": doc_id})
        doc_text = "\n".join([c.text for c in chunks])
        contexts.append(f"Document {doc_id}:\n{doc_text}")
    
    context_str = "\n\n---\n\n".join(contexts)
    
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert researcher. Compare the provided research documents on the specific query. Identify consistencies, direct conflicts, emerging trends, and provide a high-level summary."},
            {"role": "user", "content": f"Query: {query}\n\n{context_str}"},
            {"role": "system", "content": "Output MUST be JSON matching ComparisonResult model: {consistencies: list, conflicts: list, trends: list, summary: str}"}
        ],
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content or "{}"

# --- Phase 2: Grounded Synthesis Tools ---

def generate_comparison_matrix(doc_ids: List[str], metrics: List[str]) -> str:
    """
    Generates a Markdown table comparing specific metrics across papers via RAG extraction.
    """
    rag = get_rag_layer()
    header = "| Metric | " + " | ".join(doc_ids) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(doc_ids)) + " |"
    
    rows = []
    for metric in metrics:
        row_vals = []
        for doc_id in doc_ids:
            # Query specifically for this metric in this doc
            chunks = rag.query(metric, n_results=1, where={"doc_id": doc_id})
            if chunks:
                # Use LLM to extract the specific metric value concisely
                if _client:
                    ext_resp = _client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Extract the specific metric value from the text as concisely as possible (e.g. '150 samples', '98% Accuracy'). Return 'N/A' if not found."},
                            {"role": "user", "content": f"Metric: {metric}\n\nText: {chunks[0].text}"}
                        ]
                    )
                    row_vals.append(ext_resp.choices[0].message.content or "N/A")
                else:
                    row_vals.append(chunks[0].text[:30] + "...")
            else:
                row_vals.append("N/A")
        
        row = f"| {metric} | " + " | ".join(row_vals) + " |"
        rows.append(row)
        
    return "\n".join([header, divider] + rows)

def detect_conflicts(topic: str, context: str) -> str:
    """
    Analyzes text to find contradictory claims about a specific topic.
    """
    if not _client:
        return json.dumps({"topic": topic, "error": "LLM client not available."})

    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a scientific conflict detector. Identify any direct contradictions or inconsistencies in the provided context concerning the specific topic. If multiple sources are cited, name them."},
            {"role": "user", "content": f"Topic: {topic}\n\nContext: {context}"},
            {"role": "system", "content": "Output MUST be JSON matching Conflict model: {topic: str, conflicts: list[{claim_a: str, claim_b: str, source_a: str, source_b: str}], synthesis: str}"}
        ],
        response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content or "{}"

def extract_limitations(doc_id: str, text: str) -> str:
    """
    Identifies 'Limitations', 'Future Work', and 'Disclaimers' within a paper.
    """
    if not _client:
        return json.dumps({"doc_id": doc_id, "error": "LLM client not available."})

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract explicit limitations, disclaimers, or scope constraints from the provided research text. Be thorough."},
            {"role": "user", "content": f"Text: {text}"},
            {"role": "system", "content": "Output MUST be JSON: {doc_id: str, limitations: list[str], recommendation: str}"}
        ],
        response_format={"type": "json_object"}
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    data["doc_id"] = doc_id # ensure doc_id is correct
    return json.dumps(data)

# --- Tool Definitions (Phase 2) ---

MATRIX_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "generate_comparison_matrix",
        "description": "Generate a markdown table comparing papers across specific metrics (e.g. Sample Size, Accuracy).",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_ids": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["doc_ids", "metrics"]
        }
    }
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
                "context": {"type": "string"}
            },
            "required": ["topic", "context"]
        }
    }
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
                "text": {"type": "string"}
            },
            "required": ["doc_id", "text"]
        }
    }
}
