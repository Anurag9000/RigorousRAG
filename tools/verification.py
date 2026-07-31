"""Deterministic citation-structure checks.

This module intentionally does not claim to prove factual entailment. It checks
that answer markers map to server-selected evidence and reports weak lexical
alignment only as a diagnostic signal.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Dict, List, Tuple

from tools.models import AgentAnswer, Citation

_MARKER_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "from", "and", "but", "or", "not", "that",
    "this", "these", "those", "it", "its", "as", "if", "then", "which",
}
_MAX_ANSWER_CHARS = 100_000
_MAX_CITATIONS = 100
_MAX_MARKERS = 1000
_MAX_ISSUES = 500
_MAX_EVIDENCE_CHARS = 4000


def _bounded_answer(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("answer must be a string.")
    if len(value) > _MAX_ANSWER_CHARS:
        raise ValueError(
            f"answer may contain at most {_MAX_ANSWER_CHARS:,} characters."
        )
    return value


def _bounded_citations(values: Any) -> List[Citation]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("citations must be a list of Citation objects.")
    try:
        selected = list(itertools.islice(iter(values), _MAX_CITATIONS + 1))
    except Exception as exc:
        raise ValueError("citations must be iterable.") from exc
    if len(selected) > _MAX_CITATIONS:
        raise ValueError(f"citations may contain at most {_MAX_CITATIONS} items.")
    if any(not isinstance(citation, Citation) for citation in selected):
        raise ValueError("citations must contain only Citation objects.")
    return selected


def _tokenize_for_overlap(text: Any) -> set[str]:
    if not isinstance(text, str):
        return set()
    tokens = re.findall(
        r"[\w-]{3,}",
        text[:_MAX_EVIDENCE_CHARS].lower(),
        flags=re.UNICODE,
    )
    return {token for token in tokens if token not in _STOP_WORDS}


def _answer_markers(answer: str) -> Tuple[List[Tuple[str, int]], bool]:
    matches = list(
        itertools.islice(_MARKER_RE.finditer(answer), _MAX_MARKERS + 1)
    )
    truncated = len(matches) > _MAX_MARKERS
    return [
        (f"[{match.group(1)}]", match.start())
        for match in matches[:_MAX_MARKERS]
    ], truncated


def verify_citations(answer: str, citations: List[Citation]) -> List[Dict[str, Any]]:
    bounded_answer = _bounded_answer(answer)
    bounded_citations = _bounded_citations(citations)
    issues: List[Dict[str, Any]] = []
    issue_overflow = False

    def append_issue(issue: Dict[str, Any]) -> bool:
        nonlocal issue_overflow
        if len(issues) >= _MAX_ISSUES:
            issue_overflow = True
            return False
        issues.append(issue)
        return True

    labels = [citation.label for citation in bounded_citations]
    label_map: Dict[str, Citation] = {}
    for citation in bounded_citations:
        label_map.setdefault(citation.label, citation)
    if len(labels) != len(set(labels)):
        append_issue(
            {
                "type": "duplicate_labels",
                "label": "*",
                "error": "The citation list contains duplicate labels.",
            }
        )

    marker_positions, marker_overflow = _answer_markers(bounded_answer)
    if marker_overflow:
        append_issue(
            {
                "type": "too_many_markers",
                "label": "*",
                "error": (
                    f"The answer contains more than {_MAX_MARKERS} citation markers; "
                    "verification was truncated."
                ),
            }
        )

    positions_by_label: Dict[str, List[int]] = {}
    for label, position in marker_positions:
        positions_by_label.setdefault(label, []).append(position)

    for label in positions_by_label:
        citation = label_map.get(label)
        if citation is None:
            if not append_issue(
                {
                    "type": "missing_source",
                    "label": label,
                    "error": f"Citation marker {label} has no corresponding source.",
                }
            ):
                break
            continue
        evidence = citation.quote or citation.snippet or ""
        evidence_tokens = _tokenize_for_overlap(evidence)
        if len(evidence_tokens) < 8:
            if not append_issue(
                {
                    "type": "weak_evidence_text",
                    "label": label,
                    "evidence_token_count": len(evidence_tokens),
                    "error": (
                        f"The evidence text for {label} contains too few meaningful tokens "
                        "for even a lexical-alignment diagnostic. Inspect the source manually."
                    ),
                }
            ):
                break
            continue
        for position in positions_by_label[label]:
            context = bounded_answer[
                max(0, position - 300): min(len(bounded_answer), position + 300)
            ]
            context_tokens = _tokenize_for_overlap(context)
            if not context_tokens:
                continue
            overlap = len(context_tokens & evidence_tokens) / max(
                1,
                len(context_tokens | evidence_tokens),
            )
            if overlap < 0.02:
                if not append_issue(
                    {
                        "type": "low_lexical_alignment",
                        "label": label,
                        "overlap": round(overlap, 4),
                        "error": (
                            f"The nearby answer text has very low lexical alignment with "
                            f"the evidence for {label}. This is a diagnostic, not a "
                            "semantic entailment verdict."
                        ),
                    }
                ):
                    break

    used = set(positions_by_label)
    for label in labels:
        if label not in used:
            if not append_issue(
                {
                    "type": "unused_source",
                    "label": label,
                    "error": f"Source {label} was returned but not cited in the answer.",
                }
            ):
                break
    if bounded_citations and not marker_positions:
        append_issue(
            {
                "type": "missing_markers",
                "label": "*",
                "error": "Evidence was retrieved, but the answer contains no citation markers.",
            }
        )
    if not bounded_citations and not marker_positions:
        append_issue(
            {
                "type": "no_evidence",
                "label": "*",
                "error": (
                    "No citations or citation markers were supplied; citation structure "
                    "could not be evaluated."
                ),
            }
        )
    if issue_overflow:
        issues[-1] = {
            "type": "issue_limit_reached",
            "label": "*",
            "error": (
                f"Citation verification reached the {_MAX_ISSUES}-issue limit and was truncated."
            ),
        }
    return issues


def audit_hallucination(agent_answer: AgentAnswer) -> str:
    if not isinstance(agent_answer, AgentAnswer):
        raise ValueError("agent_answer must be an AgentAnswer.")
    issues = verify_citations(agent_answer.answer, agent_answer.citations)
    serious_types = {
        "missing_source",
        "duplicate_labels",
        "missing_markers",
        "too_many_markers",
        "issue_limit_reached",
    }
    serious = [issue for issue in issues if issue["type"] in serious_types]
    if serious:
        details = "; ".join(issue["error"] for issue in serious[:20])
        return f"⚠️ Citation-structure warning: {details}"
    no_evidence = [issue for issue in issues if issue["type"] == "no_evidence"]
    if no_evidence:
        return (
            "⚠️ Citation-structure warning: no evidence sources or markers were "
            "available for structural checking."
        )
    weak = [issue for issue in issues if issue["type"] == "weak_evidence_text"]
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
