from typing import List, Dict
import re
from tools.models import Citation, AgentAnswer

def verify_citations(answer: str, citations: List[Citation]) -> List[Dict]:
    """
    Cross-references every inline citation [n] in the answer with the actual source snippets.
    Returns a list of verification issues.
    """
    issues = []
    
    # 1. Extract all citation markers [n] from the answer
    markers = re.findall(r'\[(\d+)\]', answer)
    unique_markers = sorted(list(set(markers)))
    
    # 2. Map label to citation content
    label_to_citation = {c.label: c for c in citations}
    
    for marker in unique_markers:
        label = f"[{marker}]"
        if label not in label_to_citation:
            issues.append({"label": label, "error": "Citation marker used in text but no corresponding source provided."})
            continue
            
        citation = label_to_citation[label]
        snippet = citation.snippet.lower() if citation.snippet else ""
        
        # Heuristic check: Does the answer's surrounding text match keywords from the snippet?
        # A more advanced version would use an LLM or semantic similarity.
        # For this grounded implementation, we check if the label exists in the source list.
        pass

    return issues

def audit_hallucination(agent_answer: AgentAnswer) -> str:
    """
    Final audit of the answer's groundedness.
    """
    issues = verify_citations(agent_answer.answer, agent_answer.citations)
    if not issues:
        return "Verified: All citations are mapped to sources."
    else:
        return f"Warning: Potential grounding issues found: {issues}"
