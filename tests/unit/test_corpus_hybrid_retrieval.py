from dataclasses import dataclass

from tools.corpus_hybrid_retrieval import retrieve_corpus_evidence
from tools.generation_store import GenerationRecord
from tools.sparse_types import (
    SparseDocumentSnapshot,
    SparseFieldSnapshot,
    SparseMatch,
    SparseSearchHit,
)


@dataclass
class Chunk:
    id: str
    text: str
    score: float
    metadata: dict


class Rag:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.chunks


class Sparse:
    def __init__(self, hits, snapshots):
        self.hits = hits
        self.snapshots = snapshots
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.hits

    def snapshot_document(self, *, owner_id, doc_id):
        return self.snapshots.get((owner_id, doc_id))


class Generations:
    def __init__(self, records):
        self.records = records

    def current(self, *, owner_id, doc_id):
        return self.records.get((owner_id, doc_id))


def record(doc_id, *, sequence=1, sparse_generation=1, content="a", profile="b"):
    return GenerationRecord(
        "alice",
        doc_id,
        sequence,
        "active",
        content * 64,
        profile * 64,
        2,
        sparse_generation,
        1.0,
        {},
    )


def hit(doc_id, *, score=1.0, generation=1, profile="b", field_id="body"):
    return SparseSearchHit(
        doc_id=doc_id,
        score=score,
        generation=generation,
        profile_fingerprint=profile * 64,
        metadata={},
        matches=(
            SparseMatch(
                field_id=field_id,
                field_type="body",
                field_position=0,
                page_number=2,
                section="Results",
                term_frequencies={"rare": 1},
                positions={"rare": (3,)},
                metadata={},
            ),
        ),
    )


def snapshot(doc_id, *, generation=1, profile="b", field_id="body"):
    return SparseDocumentSnapshot(
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        profile_fingerprint=profile * 64,
        metadata={},
        fields=(
            SparseFieldSnapshot(
                field_id=field_id,
                field_type="body",
                text=f"rare lexical evidence for {doc_id}",
                position=0,
                token_count=6,
                page_number=2,
                section="Results",
                metadata={},
            ),
        ),
    )


def dense(doc_id, *, score=0.8, owner="alice", profile="b", content="a"):
    return Chunk(
        id=f"{doc_id}:chunk",
        text=f"dense evidence for {doc_id}",
        score=score,
        metadata={
            "owner_id": owner,
            "doc_id": doc_id,
            "content_sha256": content * 64,
            "embedding_profile_fingerprint": profile * 64,
            "page_number": 1,
            "section_title": "Introduction",
        },
    )


def test_hybrid_can_recover_sparse_only_document():
    rag = Rag([dense("dense-doc", score=0.7)])
    sparse = Sparse(
        [hit("sparse-doc", score=1.0)],
        {("alice", "sparse-doc"): snapshot("sparse-doc")},
    )
    generations = Generations(
        {
            ("alice", "dense-doc"): record("dense-doc"),
            ("alice", "sparse-doc"): record("sparse-doc"),
        }
    )

    results = retrieve_corpus_evidence(
        "rare term",
        owner_id="alice",
        rag=rag,
        sparse=sparse,
        generations=generations,
        mode="hybrid",
        top_k=3,
        dense_weight=0.3,
        sparse_weight=0.7,
    )

    assert any(item.doc_id == "sparse-doc" for item in results)
    sparse_result = next(item for item in results if item.doc_id == "sparse-doc")
    assert sparse_result.source_kind == "sparse_field"
    assert sparse_result.page_number == 2
    assert sparse_result.metadata["positions"] == {"rare": (3,)}


def test_stale_sparse_generation_is_rejected():
    results = retrieve_corpus_evidence(
        "rare",
        owner_id="alice",
        rag=Rag([]),
        sparse=Sparse(
            [hit("doc-1", generation=1)],
            {("alice", "doc-1"): snapshot("doc-1", generation=1)},
        ),
        generations=Generations(
            {("alice", "doc-1"): record("doc-1", sparse_generation=2)}
        ),
        mode="sparse",
    )
    assert results == ()


def test_dense_profile_hash_and_owner_must_match_manifest():
    rag = Rag(
        [
            dense("owner-leak", owner="bob"),
            dense("wrong-profile", profile="c"),
            dense("wrong-content", content="c"),
            dense("valid"),
        ]
    )
    generations = Generations(
        {
            ("alice", "owner-leak"): record("owner-leak"),
            ("alice", "wrong-profile"): record("wrong-profile"),
            ("alice", "wrong-content"): record("wrong-content"),
            ("alice", "valid"): record("valid"),
        }
    )
    results = retrieve_corpus_evidence(
        "evidence",
        owner_id="alice",
        rag=rag,
        sparse=Sparse([], {}),
        generations=generations,
        mode="dense",
        top_k=10,
    )
    assert [item.doc_id for item in results] == ["valid"]


def test_document_filter_is_forwarded_to_both_retrievers():
    rag = Rag([dense("doc-1")])
    sparse = Sparse(
        [hit("doc-1")],
        {("alice", "doc-1"): snapshot("doc-1")},
    )
    results = retrieve_corpus_evidence(
        "rare",
        owner_id="alice",
        doc_id="doc-1",
        rag=rag,
        sparse=sparse,
        generations=Generations({("alice", "doc-1"): record("doc-1")}),
        mode="hybrid",
    )
    assert results
    assert rag.calls[0][1]["doc_id"] == "doc-1"
    assert sparse.calls[0][1]["doc_id"] == "doc-1"


def test_deleted_or_missing_manifest_is_never_returned():
    deleted = GenerationRecord(
        "alice",
        "doc-1",
        2,
        "deleted",
        "a" * 64,
        "b" * 64,
        0,
        0,
        2.0,
        {},
    )
    results = retrieve_corpus_evidence(
        "evidence",
        owner_id="alice",
        rag=Rag([dense("doc-1"), dense("doc-2")]),
        sparse=Sparse([], {}),
        generations=Generations({("alice", "doc-1"): deleted}),
        mode="dense",
        top_k=10,
    )
    assert results == ()
