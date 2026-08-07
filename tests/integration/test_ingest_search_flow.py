from types import SimpleNamespace

from tools.document_service import ingest_and_index


class FakeRag:
    def __init__(self):
        self.calls = []

    def add_document(self, **kwargs):
        self.calls.append(kwargs)
        return len(kwargs["sections"])


def install_commit(monkeypatch, rag):
    def commit(document, **kwargs):
        count = rag.add_document(
            doc_id=document.id,
            text=document.text,
            sections=document.sections,
            metadata=kwargs["metadata"],
            chunk_size=1_000,
            overlap=120,
            replace=True,
        )
        return SimpleNamespace(vector_rows=count)

    monkeypatch.setattr(
        "tools.document_service.commit_finalized_document",
        commit,
    )


def test_ingestion_service_indexes_redacted_semantic_sections(tmp_path, monkeypatch):
    path = tmp_path / "research.txt"
    path.write_text(
        "Title: Project Alpha\n\nContact alice@example.com.\n\nMethods: measure accuracy.",
        encoding="utf-8",
    )
    rag = FakeRag()
    install_commit(monkeypatch, rag)
    indexed = ingest_and_index(
        str(path),
        owner_id="alice",
        rag=rag,
        client=None,
    )
    assert indexed.chunk_count >= 1
    assert len(rag.calls) == 1
    call = rag.calls[0]
    assert call["metadata"]["owner_id"] == "alice"
    assert call["replace"] is True
    assert call["sections"]
    assert all("alice@example.com" not in section.content for section in call["sections"])
    assert call["metadata"]["llm_summary"]


def test_repeated_ingestion_produces_same_document_identity(tmp_path, monkeypatch):
    path = tmp_path / "repeat.md"
    path.write_text("# Repeatable\n\nSame contents.", encoding="utf-8")
    first_rag = FakeRag()
    second_rag = FakeRag()
    install_commit(monkeypatch, first_rag)
    first = ingest_and_index(
        str(path), owner_id="alice", rag=first_rag, client=None,
    )
    install_commit(monkeypatch, second_rag)
    second = ingest_and_index(
        str(path), owner_id="alice", rag=second_rag, client=None,
    )
    assert first.document.id == second.document.id
