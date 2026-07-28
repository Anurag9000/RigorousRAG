from tools.job_store import JobStore


def test_failed_transition_clears_finalizing_document_id(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.pdf",
    )
    assert store.claim("job-1", "alice", max_attempts=1) is True
    store.update(
        "job-1",
        "alice",
        status="finalizing",
        filename="paper.pdf",
        doc_id="doc-not-committed",
    )
    assert store.get("job-1", "alice")["doc_id"] == "doc-not-committed"

    store.update(
        "job-1",
        "alice",
        status="failed",
        filename="paper.pdf",
        message="Finalization failed.",
    )

    public = store.get("job-1", "alice")
    internal = store.get_internal("job-1", "alice")
    assert public and public["status"] == "failed"
    assert public["doc_id"] is None
    assert internal and internal["doc_id"] is None
