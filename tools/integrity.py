import json
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

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
    Stub for Visual Entailment. In a real system, this would call a Vision-LLM (GPT-4V).
    """
    # Logic: Fetch document -> Extract Figure Image -> Send Query to Vision Model
    print(f"[Integrity] Checking visual entailment for '{figure_id}' against claim: {claim_text[:50]}...")
    
    # Placeholder response
    result = VisualEntailmentResult(
        claim_text=claim_text,
        figure_id=figure_id,
        verdict=EntailmentVerdict.UNCERTAIN,
        rationale="Visual analysis module requires GPT-4V integration. Verify manually.",
        confidence=0.5
    )
    return result.model_dump_json()

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
    Simulates a debate between Advocate, Skeptic, and Judge.
    """
    print(f"[Integrity] Starting debate on claim: {claim}")
    
    # In a real implementation, this would chain 3 separate LLM calls with different system prompts.
    
    result = DebateResult(
        verdict="Caution",
        key_issues=["Sample size justification missing", "Potential p-hacking"],
        supporting_evidence=["Strong effect size reported"],
        recommended_followups=["Request raw data", "Check reproduction studies"]
    )
    return result.model_dump_json()

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
    Stub for cross-paper analysis (Goal 13.4).
    """
    print(f"[Integrity] Comparing {len(doc_ids)} papers on: {query}")
    
    result = ComparisonResult(
        consistencies=["Consensus found on baseline metrics."],
        conflicts=["Discrepancy in methodology for step 3."],
        trends=["Increasing focus on efficiency over time."],
        summary="Across the selected documents, there is agreement on the core claims but methodology varies."
    )
    return result.model_dump_json()

# --- Phase 2: Grounded Synthesis Tools ---

def generate_comparison_matrix(doc_ids: List[str], metrics: List[str]) -> str:
    """
    Generates a Markdown table comparing specific metrics across papers.
    """
    header = "| Metric | " + " | ".join(doc_ids) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(doc_ids)) + " |"
    
    rows = []
    for m in metrics:
        # Placeholder: Real logic would query RAG for each metric/doc combo
        row = f"| {m} | " + " | ".join(["[Extracted Data]"] * len(doc_ids)) + " |"
        rows.append(row)
        
    return "\n".join([header, divider] + rows)

def detect_conflicts(topic: str, context: str) -> str:
    """
    Analyzes text to find contradictory claims about a specific topic.
    """
    # In a real implementation, this would be an LLM call with a "Debate" prompt.
    return json.dumps({
        "topic": topic,
        "conflicts": [
            {"claim_a": "Sample A increases X", "claim_b": "Sample B decreases X", "source_a": "Doc 1", "source_b": "Doc 2"}
        ],
        "synthesis": "There is a direct contradiction in results regarding X."
    })

def extract_limitations(doc_id: str, text: str) -> str:
    """
    Specifically targets 'Limitations', 'Future Work', and 'Disclaimers'.
    """
    # Heuristic for demo purposes
    limitations = []
    if "limitations" in text.lower():
        # extract snippet around 'limitations'
        limitations.append("Found explicit 'Limitations' section.")
        
    return json.dumps({
        "doc_id": doc_id,
        "limitations": limitations,
        "recommendation": "Treat primary claims with caution if mentioned limitations are structural."
    })

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
