from tools.job_store import JobStore
from tools.operator_repair import list_corrupt_jobs


def test_job_store_normalizes_public_text_controls_before_persistence(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")

    store.update(
        "job-public-text",
        "operator-owner",
        status="queued",
        filename="paper\nname\t.pdf",
        message="queued\r\nwith\tcontrols\x7f",
    )

    record = store.get_internal("job-public-text", "operator-owner")
    assert record is not None
    assert record["filename"] == "paper name .pdf"
    assert record["message"] == "queued  with controls"
    assert not any(
        ord(character) < 32 or ord(character) == 127
        for value in (record["filename"], record["message"])
        for character in value
    )
    assert list_corrupt_jobs(store) == []


def test_job_store_public_text_remains_bounded_after_control_normalization(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")

    store.update(
        "job-bounds",
        "operator-owner",
        status="queued",
        filename=("a\n" * 600),
        message=("b\t" * 3000),
    )

    record = store.get_internal("job-bounds", "operator-owner")
    assert record is not None
    assert len(record["filename"]) == 500
    assert len(record["message"]) == 2000
    assert "\n" not in record["filename"]
    assert "\t" not in record["message"]
