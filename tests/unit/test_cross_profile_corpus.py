import hashlib

import pytest

import tools.cross_profile_corpus as cross
from tools.corpus_hybrid_retrieval import CorpusEvidence
from tools.retrieval_architectures import ScoreCalibration


def fp(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Rag:
    def __init__(self, name):
        self.name = name

    def query(self, *_args, **_kwargs):
        return []


def evidence(profile: str, score: float, *, text="shared evidence", source_kind="dense_chunk"):
    return CorpusEvidence(
        evidence_id=f"e:{profile[:8]}:{source_kind}",
        doc_id="doc-1",
        text=text,
        score=score,
        dense_score=score if source_kind == "dense_chunk" else 0.0,
        sparse_score=score if source_kind == "sparse_field" else 0.0,
        generation_sequence=7,
        profile_fingerprint=profile,
        source_kind=source_kind,
        page_number=2,
        section="Results",
    )


def test_cross_profile_fanout_calibrates_and_deduplicates(monkeypatch):
    a = fp("profile-a")
    b = fp("profile-b")
    calls = []

    def fake_retrieve(query, **kwargs):
        calls.append((kwargs["mode"], getattr(kwargs["rag"], "name", "")))
        if kwargs["mode"] == "sparse":
            return (evidence(a, 0.6, source_kind="sparse_field"),)
        if kwargs["rag"].name == "a":
            return (evidence(a, 0.8),)
        return (evidence(b, 0.4),)

    monkeypatch.setattr(cross, "retrieve_corpus_evidence", fake_retrieve)
    result = cross.retrieve_cross_profile_evidence(
        "query",
        owner_id="alice",
        backends=(
            cross.ProfileCorpusBackend(a, Rag("a"), weight=2.0),
            cross.ProfileCorpusBackend(
                b,
                Rag("b"),
                weight=1.0,
                calibration=ScoreCalibration(temperature=1.0, bias=0.2),
            ),
        ),
        sparse=object(),
        generations=object(),
        top_k=5,
        per_profile_top_k=5,
        sparse_weight=1.0,
    )
    assert len(result) == 1
    row = result[0]
    assert row.contributing_profiles == tuple(sorted((a, b)))
    assert set(row.profile_scores) == {a, b}
    assert row.sparse_score is not None
    assert 0.0 <= row.score <= 1.0
    assert calls.count(("sparse", "a")) == 1
    assert len([item for item in calls if item[0] == "dense"]) == 2


def test_profile_mismatch_is_not_allowed_to_cross_backend_boundary(monkeypatch):
    a = fp("profile-a")
    b = fp("profile-b")

    def fake_retrieve(_query, **kwargs):
        if kwargs["mode"] == "sparse":
            return ()
        return (evidence(b, 0.99),)

    monkeypatch.setattr(cross, "retrieve_corpus_evidence", fake_retrieve)
    result = cross.retrieve_cross_profile_evidence(
        "query",
        owner_id="alice",
        backends=(cross.ProfileCorpusBackend(a, Rag("a")),),
        sparse=object(),
        generations=object(),
        include_sparse=False,
    )
    assert result == ()


def test_optional_profile_failure_falls_back_but_required_failure_is_bounded(monkeypatch):
    a = fp("profile-a")
    b = fp("profile-b")

    def fake_retrieve(_query, **kwargs):
        if kwargs["rag"].name == "a":
            raise RuntimeError("private backend failure")
        return (evidence(b, 0.7),)

    monkeypatch.setattr(cross, "retrieve_corpus_evidence", fake_retrieve)
    fallback = cross.retrieve_cross_profile_evidence(
        "query",
        owner_id="alice",
        backends=(
            cross.ProfileCorpusBackend(a, Rag("a"), required=False),
            cross.ProfileCorpusBackend(b, Rag("b"), required=True),
        ),
        sparse=object(),
        generations=object(),
        include_sparse=False,
    )
    assert len(fallback) == 1
    assert fallback[0].contributing_profiles == (b,)

    with pytest.raises(RuntimeError, match="required profile retrieval failed"):
        cross.retrieve_cross_profile_evidence(
            "query",
            owner_id="alice",
            backends=(cross.ProfileCorpusBackend(a, Rag("a"), required=True),),
            sparse=object(),
            generations=object(),
            include_sparse=False,
        )


def test_profile_count_and_duplicate_fingerprints_are_bounded():
    fingerprint = fp("same")
    backend = cross.ProfileCorpusBackend(fingerprint, Rag("a"))
    with pytest.raises(ValueError, match="unique"):
        cross.retrieve_cross_profile_evidence(
            "query",
            owner_id="alice",
            backends=(backend, backend),
            sparse=object(),
            generations=object(),
        )
    with pytest.raises(ValueError, match="between 1 and 8"):
        cross.retrieve_cross_profile_evidence(
            "query",
            owner_id="alice",
            backends=(),
            sparse=object(),
            generations=object(),
        )
