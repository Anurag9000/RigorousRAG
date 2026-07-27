from tools.job_store import JobStore
from tools.rate_limit import SlidingWindowRateLimiter


def test_job_store_is_owner_scoped_persistent_and_prunable(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    first = JobStore(path, ttl_seconds=60)
    first.update("job-1", "alice", status="processing", filename="paper.pdf")
    second = JobStore(path, ttl_seconds=60)
    assert second.get("job-1", "alice")["filename"] == "paper.pdf"
    assert second.get("job-1", "bob") is None
    second.update("job-1", "alice", status="success", filename="paper.pdf", doc_id="doc-1")
    assert first.get("job-1", "alice")["doc_id"] == "doc-1"
    assert second.prune(now=10_000_000) == 1
    assert second.get("job-1", "alice") is None


def test_sliding_window_rate_limiter_reports_retry_time():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)
    assert limiter.retry_after("alice", now=0.0) == 0.0
    assert limiter.retry_after("alice", now=1.0) == 0.0
    retry = limiter.retry_after("alice", now=2.0)
    assert retry > 0
    assert limiter.retry_after("bob", now=2.0) == 0.0
    assert limiter.retry_after("alice", now=61.0) == 0.0
