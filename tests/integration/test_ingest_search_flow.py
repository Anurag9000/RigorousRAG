from tools.document_service import ingest_and_index


class FakeRag:
    def __init__(self):
        self.calls = []

    def add_document(self, **kwargs):
        self.calls.append(kwargs)
        return len(kwargs["sections"])


def test_ingestion_service_indexes_redacted_semantic_sections(tmp_path):
    path = tmp_path / "research.txt"
    path.write_text(
        "Title: Project Alpha\n\nContact alice@example.com.\n\nMethods: measure accuracy.",
        encoding="utf-8",
    )
    rag = FakeRag()
    indexed = ingest_and_index(str(path), owner_id="alice", rag=rag, client=None)
    assert indexed.chunk_count >= 1
    assert len(rag.calls) == 1
    call = rag.calls[0]
    assert call["metadata"]["owner_id"] == "alice"
    assert call["replace"] is True
    assert call["sections"]
    assert all("alice@example.com" not in section.content for section in call["sections"])
    assert call["metadata"]["llm_summary"]


def test_repeated_ingestion_produces_same_document_identity(tmp_path):
    path = tmp_path / "repeat.md"
    path.write_text("# Repeatable\n\nSame contents.", encoding="utf-8")
    first = ingest_and_index(str(path), owner_id="alice", rag=FakeRag(), client=None)
    second = ingest_and_index(str(path), owner_id="alice", rag=FakeRag(), client=None)
    assert first.document.id == second.document.id
