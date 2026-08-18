from __future__ import annotations

import hashlib

import pytest

from tools.evidence_context_packing import ContextEvidenceCandidate, ContextPackingPolicy, EvidenceSimilarity, pack_evidence_context
from tools.evidence_context_materialization import ContextContentBinding, materialize_context


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def candidate(name: str, *, tokens=10, relevance=0.5, support=0.0, contradiction=0.0, authority=1.0, mandatory=False, doc=None, source=None):
    return ContextEvidenceCandidate(
        evidence_id=name,
        evidence_sha256=sha(f"evidence:{name}"),
        document_id=doc or f"doc:{name}",
        source_id=source or f"source:{name}",
        generation_id="gen",
        token_count=tokens,
        relevance=relevance,
        support=support,
        contradiction=contradiction,
        authority=authority,
        mandatory=mandatory,
    )


def policy(**overrides):
    values = dict(max_context_tokens=100, max_items=10, max_per_document=3, max_per_source=3, relevance_weight=1.0, support_weight=0.5, contradiction_weight=0.5, authority_weight=0.2, redundancy_penalty=0.8, min_counterevidence_items=0)
    values.update(overrides)
    return ContextPackingPolicy(**values)


def test_mandatory_evidence_is_selected_first_and_budget_overflow_fails() -> None:
    required = candidate("required", tokens=20, mandatory=True)
    optional = candidate("optional", relevance=1.0)
    packed = pack_evidence_context((optional, required), policy=policy(max_context_tokens=30))
    assert packed.selected[0].evidence_sha256 == required.evidence_sha256
    assert packed.selected[0].selection_reason == "mandatory"

    with pytest.raises(ValueError, match="mandatory evidence exceeds"):
        pack_evidence_context((candidate("too-big", tokens=101, mandatory=True),), policy=policy(max_context_tokens=100))


def test_mmr_similarity_penalty_prefers_diverse_evidence() -> None:
    first = candidate("first", relevance=1.0)
    duplicate = candidate("duplicate", relevance=0.99)
    diverse = candidate("diverse", relevance=0.8)
    similarity = EvidenceSimilarity(first.evidence_sha256, duplicate.evidence_sha256, 1.0)
    packed = pack_evidence_context((first, duplicate, diverse), policy=policy(max_items=2), similarities=(similarity,))
    selected = {row.evidence_sha256 for row in packed.selected}
    assert first.evidence_sha256 in selected
    assert diverse.evidence_sha256 in selected
    assert duplicate.evidence_sha256 not in selected


def test_counterevidence_quota_preserves_contradictory_evidence() -> None:
    support = candidate("support", relevance=1.0, support=1.0)
    counter = candidate("counter", relevance=0.1, contradiction=0.9)
    packed = pack_evidence_context(
        (support, counter),
        policy=policy(max_items=2, min_counterevidence_items=1, counterevidence_threshold=0.5),
    )
    selected = {row.evidence_sha256: row.selection_reason for row in packed.selected}
    assert selected[counter.evidence_sha256] == "counterevidence_quota"
    assert packed.counterevidence_count >= 1


def test_document_and_source_caps_prevent_context_monopoly() -> None:
    rows = (
        candidate("a", relevance=1.0, doc="same-doc", source="same-source"),
        candidate("b", relevance=0.9, doc="same-doc", source="same-source"),
        candidate("c", relevance=0.8, doc="other-doc", source="other-source"),
    )
    packed = pack_evidence_context(rows, policy=policy(max_per_document=1, max_per_source=1))
    ids = {row.evidence_sha256 for row in packed.selected}
    assert sha("evidence:c") in ids
    assert len(ids & {sha("evidence:a"), sha("evidence:b")}) == 1


def test_packing_is_deterministically_content_addressed() -> None:
    rows = (candidate("a", relevance=0.7), candidate("b", relevance=0.6))
    first = pack_evidence_context(rows, policy=policy())
    second = pack_evidence_context(tuple(reversed(rows)), policy=policy())
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.candidate_pool_sha256 == second.candidate_pool_sha256


class Provider:
    def __init__(self, mapping):
        self.mapping = mapping

    def fetch_text(self, *, evidence_sha256):
        return self.mapping[evidence_sha256]


class Counter:
    def __init__(self, tokenizer_sha256):
        self._sha = tokenizer_sha256

    @property
    def tokenizer_sha256(self):
        return self._sha

    def count_tokens(self, text):
        return len(text.split())


def test_materialization_rehashes_and_retokenizes_every_selected_item() -> None:
    row = candidate("a", tokens=2)
    packed = pack_evidence_context((row,), policy=policy(max_context_tokens=10))
    text = "hello world"
    tokenizer = sha("tokenizer")
    binding = ContextContentBinding(row.evidence_sha256, sha(text), tokenizer, 2)
    materialized = materialize_context(packed, bindings=(binding,), provider=Provider({row.evidence_sha256: text}), token_counter=Counter(tokenizer))
    assert materialized.prompt_text == text
    assert materialized.total_tokens == 2
    assert len(materialized.context_sha256) == 64


def test_materialization_rejects_storage_tamper_tokenizer_or_token_count_drift() -> None:
    row = candidate("a", tokens=2)
    packed = pack_evidence_context((row,), policy=policy(max_context_tokens=10))
    tokenizer = sha("tokenizer")
    binding = ContextContentBinding(row.evidence_sha256, sha("hello world"), tokenizer, 2)
    with pytest.raises(RuntimeError, match="text digest"):
        materialize_context(packed, bindings=(binding,), provider=Provider({row.evidence_sha256: "mutated text"}), token_counter=Counter(tokenizer))
    with pytest.raises(ValueError, match="runtime tokenizer"):
        materialize_context(packed, bindings=(binding,), provider=Provider({row.evidence_sha256: "hello world"}), token_counter=Counter(sha("other-tokenizer")))

    class WrongCounter(Counter):
        def count_tokens(self, text):
            return 3

    with pytest.raises(RuntimeError, match="token count"):
        materialize_context(packed, bindings=(binding,), provider=Provider({row.evidence_sha256: "hello world"}), token_counter=WrongCounter(tokenizer))
