"""Deterministic generation metrics for RAG evaluation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple


_TOKEN = re.compile(r"\w+", re.UNICODE)
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(text or ""))


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 over word tokens."""

    a = _tokens(candidate)
    b = _tokens(reference)
    if not a or not b:
        return 1.0 if a == b else 0.0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(a)
    recall = lcs / len(b)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def chrf(candidate: str, reference: str, *, max_order: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score inspired by chrF, dependency-free.

    Orders that cannot exist in either string are excluded from the effective-order
    average. This preserves the expected identity property for short exact matches
    (for example, ``chrf("Paris", "Paris") == 1``) instead of penalizing the pair
    for unavailable six-character n-grams.
    """

    if max_order <= 0 or beta <= 0:
        raise ValueError("max_order and beta must be positive.")
    candidate = candidate or ""
    reference = reference or ""
    if not candidate and not reference:
        return 1.0
    scores = []
    beta2 = beta * beta
    for order in range(1, max_order + 1):
        cand = Counter(candidate[i:i + order] for i in range(max(len(candidate) - order + 1, 0)))
        ref = Counter(reference[i:i + order] for i in range(max(len(reference) - order + 1, 0)))
        cand_total = sum(cand.values())
        ref_total = sum(ref.values())
        if not cand_total and not ref_total:
            continue
        overlap = sum((cand & ref).values())
        precision = overlap / cand_total if cand_total else 0.0
        recall = overlap / ref_total if ref_total else 0.0
        denom = beta2 * precision + recall
        scores.append(((1 + beta2) * precision * recall / denom) if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def split_claims(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in _CLAIM_SPLIT.split((text or "").strip()) if part.strip())


@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    supported: bool
    score: float


def unsupported_claim_rate(
    answer: str,
    evidence: Sequence[str],
    *,
    scorer: Optional[Callable[[str, Sequence[str]], float]] = None,
    threshold: float = 0.5,
) -> float:
    """Fraction of sentence-like claims not supported by supplied evidence.

    The default lexical scorer is conservative and deterministic. Production
    evaluation can inject an NLI/factuality scorer through ``scorer``.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    claims = split_claims(answer)
    if not claims:
        return 0.0

    def lexical(claim: str, passages: Sequence[str]) -> float:
        claim_tokens = set(_tokens(claim))
        if not claim_tokens:
            return 1.0
        evidence_tokens = set()
        for passage in evidence:
            evidence_tokens.update(_tokens(passage))
        return len(claim_tokens & evidence_tokens) / len(claim_tokens)

    selected = scorer or lexical
    unsupported = 0
    for claim in claims:
        score = float(selected(claim, evidence))
        if score < threshold:
            unsupported += 1
    return unsupported / len(claims)


def best_reference_score(
    candidate: str,
    references: Iterable[str],
    metric: Callable[[str, str], float] = rouge_l,
) -> float:
    values = [float(metric(candidate, reference)) for reference in references]
    return max(values, default=0.0)