"""Deterministic citation-structure checks.

This module intentionally does not claim to prove factual entailment. It checks
that answer markers map to server-selected evidence and reports weak lexical
alignment only as a diagnostic signal.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from tools.models import AgentAnswer, Citation

_MARKER_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "from", "and", "but", "or", "not", "that",
    "this", "these", "those", "it", "its", "as", "if", "then", "which",
}


def _tokenize_for_overlap(text: str) -> set[str]:
    tokens = re.findall(r"[\w-]{3,}", (text or "").lower(), flags=re.UNICODE)
    return {token for token in tokens if token not in _STOP_WORDS}


def _answer_markers(answer: str) -> List[str]:
    return [f"[{value}]" for value in _MARKER_RE.findall(answer or "")]


def verify_citations(answer: str, citations: List[Citation]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    labels = [citation.label for citation in citations]
    label_map = {citation.label: citation for citation in citations}
    if len(labels) != len(set(labels)):
        issues.append({
            "type": "duplicate_labels",
            "label": "*",
            "error": "The citation list contains duplicate labels.",
        })

    markers = _answer_markers(answer)
    for label in sorted(set(markers)):
        citation = label_map.get(label)
        if citation is None:
            issues.append({
                "type": "missing_source",
                "label": label,
                "error": f"Citation marker {label} has no corresponding source.",
            })
            continue
        evidence = citation.quote or citation.snippet or ""
        evidence_tokens = _tokenize_for_overlap(evidence)
        if len(evidence_tokens) < 8:
            issues.append({
                "type": "weak_evidence_text",
                "label": label,
                "evidence_token_count": len(evidence_tokens),
                "error": (
                    f"The evidence text for {label} contains too few meaningful tokens "
                    "for even a lexical-alignment diagnostic. Inspect the source manually."
                ),
            })
            continue
        positions = [match.start() for match in re.finditer(re.escape(label), answer)]
        for position in positions:
            context = answer[max(0, position - 300): min(len(answer), position + 300)]
            context_tokens = _tokenize_for_overlap(context)
            if not context_tokens:
                continue
            overlap = len(context_tokens & evidence_tokens) / max(
                1, len(context_tokens | evidence_tokens)
            )
            if overlap < 0.02:
                issues.append({
                    "type": "low_lexical_alignment",
                    "label": label,
                    "overlap": round(overlap, 4),
                    "error": (
                        f"The nearby answer text has very low lexical alignment with "
                        f"the evidence for {label}. This is a diagnostic, not a "
                        "semantic entailment verdict."
                    ),
                })

    used = set(markers)
    for label in labels:
        if label not in used:
            issues.append({
                "type": "unused_source",
                "label": label,
                "error": f"Source {label} was returned but not cited in the answer.",
            })
    if citations and not markers:
        issues.append({
            "type": "missing_markers",
            "label": "*",
            "error": "Evidence was retrieved, but the answer contains no citation markers.",
        })
    return issues


def audit_hallucination(agent_answer: AgentAnswer) -> str:
    issues = verify_citations(agent_answer.answer, agent_answer.citations)
    serious = [
        issue for issue in issues
        if issue["type"] in {"missing_source", "duplicate_labels", "missing_markers"}
    ]
    if serious:
        details = "; ".join(issue["error"] for issue in serious)
        return f"⚠️ Citation-structure warning: {details}"
    weak = [
        issue for issue in issues if issue["type"] == "weak_evidence_text"
    ]
    if weak:
        labels = ", ".join(sorted({issue["label"] for issue in weak}))
        return (
            "⚠️ Citation diagnostic: evidence text is too short for lexical checking "
            f"for {labels}; manual source inspection is required."
        )
    diagnostics = [
        issue for issue in issues if issue["type"] == "low_lexical_alignment"
    ]
    if diagnostics:
        labels = ", ".join(sorted({issue["label"] for issue in diagnostics}))
        return (
            "⚠️ Citation diagnostic: low lexical alignment was detected for "
            f"{labels}; manual source inspection is recommended."
        )
    return (
        f"✅ Citation-structure check passed for "
        f"{len(agent_answer.citations)} source(s)."
    )
