"""Citation and hallucination verification tools."""

import re
from typing import List, Dict

from tools.models import Citation, AgentAnswer

# ---------------------------------------------------------------------------
# Stop-word set for Jaccard overlap computation
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "only", "same", "than", "that", "this",
    "these", "those", "it", "its", "itself", "also", "as", "if", "then",
    "which", "who", "whom", "when", "where", "why", "how", "all", "each",
}


def _tokenize_for_overlap(text: str) -> set:
    """
    Lowercase alphabetic tokens of 3+ characters, filtered of stop-words.
    Produces a set suitable for Jaccard-similarity computation.
    """
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def verify_citations(answer: str, citations: List[Citation]) -> List[Dict]:
    """
    Cross-references every inline citation [n] in the answer with the actual
    source snippets using two complementary checks:

    1. **Structural check** — every [n] marker found in the answer must map to
       a provided Citation object.  Missing mappings are flagged immediately.

    2. **Semantic grounding check (Jaccard overlap)** — for each [n] that has a
       corresponding snippet, we extract the surrounding answer context (±200
       chars) and compute word-level Jaccard similarity between that context and
       the citation snippet.  A similarity below 5% with a sufficiently long
       snippet is flagged as a potential hallucination.

    Returns a list of issue dicts, each containing 'label', 'type', and 'error'.
    An empty list means all citations pass both checks.
    """
    issues: List[Dict] = []

    # Map citation label → Citation object
    label_to_citation: Dict[str, Citation] = {c.label: c for c in citations}

    # Find all unique citation markers in the answer
    markers = re.findall(r"\[(\d+)\]", answer)
    unique_markers = sorted(set(markers), key=int)

    for marker in unique_markers:
        label = f"[{marker}]"

        # --- Check 1: structural presence ---
        if label not in label_to_citation:
            issues.append(
                {
                    "label": label,
                    "type": "missing_source",
                    "error": (
                        f"Citation marker {label} used in answer but "
                        "no corresponding source was provided."
                    ),
                }
            )
            continue

        citation = label_to_citation[label]
        snippet = citation.snippet or ""

        # --- Check 2: semantic grounding via Jaccard similarity ---
        # Only meaningful when the snippet has enough vocabulary
        snippet_tokens = _tokenize_for_overlap(snippet)
        if len(snippet_tokens) < 5:
            # Snippet is too short / trivial — skip overlap check
            continue

        # Find all occurrences of this label in the answer
        for m in re.finditer(re.escape(label), answer):
            start = max(0, m.start() - 200)
            end = min(len(answer), m.end() + 200)
            context_window = answer[start:end]

            context_tokens = _tokenize_for_overlap(context_window)
            union = context_tokens | snippet_tokens
            intersection = context_tokens & snippet_tokens

            if not union:
                continue

            jaccard = len(intersection) / len(union)
            if jaccard < 0.05:
                issues.append(
                    {
                        "label": label,
                        "type": "low_overlap",
                        "jaccard": round(jaccard, 3),
                        "error": (
                            f"Low semantic overlap ({jaccard:.1%}) between answer "
                            f"context and source snippet for {label}.  The cited "
                            "source may not actually support the claim made here — "
                            "possible hallucination or misattribution."
                        ),
                    }
                )

    return issues


def audit_hallucination(agent_answer: AgentAnswer) -> str:
    """
    High-level audit of an AgentAnswer for hallucination risk.

    Runs both the structural and semantic grounding checks via verify_citations.
    Returns a human-readable status string suitable for appending to the answer.
    """
    issues = verify_citations(agent_answer.answer, agent_answer.citations)
    if not issues:
        return (
            f"✅ Citation audit passed: all {len(agent_answer.citations)} "
            "citation(s) are structurally present and semantically grounded."
        )

    missing = [i for i in issues if i["type"] == "missing_source"]
    low_overlap = [i for i in issues if i["type"] == "low_overlap"]

    parts = []
    if missing:
        labels = ", ".join(i["label"] for i in missing)
        parts.append(f"{len(missing)} unmapped marker(s): {labels}")
    if low_overlap:
        labels = ", ".join(
            f"{i['label']} (Jaccard={i['jaccard']})" for i in low_overlap
        )
        parts.append(f"{len(low_overlap)} low-overlap citation(s): {labels}")

    return f"⚠️ Citation audit warning — Potential grounding issues found: {'; '.join(parts)}."
