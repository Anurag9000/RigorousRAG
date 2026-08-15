import hashlib
import json

import pytest

from evaluation.dataset_manifest import DatasetAcquisitionManifest, DatasetFileManifest
from evaluation.dataset_registry import get_dataset_spec
from evaluation.verified_benchmark_dataset import load_verified_benchmark_dataset


def _manifest(path, dataset_name, *, version="v1", records=None):
    payload = path.read_bytes()
    return DatasetAcquisitionManifest(
        dataset_name=dataset_name,
        version=version,
        revision="fixture-revision",
        source_uri="fixture://local",
        license_id="TEST-ONLY",
        license_sha256=hashlib.sha256(b"fixture license").hexdigest(),
        files=(
            DatasetFileManifest(
                path=path.name,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                records=records,
            ),
        ),
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_qasper_and_miracl_are_in_governed_registry():
    qasper = get_dataset_spec("QASPER")
    miracl = get_dataset_spec("miracl")

    assert qasper.domain == "scientific"
    assert qasper.multihop is True
    assert miracl.domain == "multilingual"
    assert miracl.format == "multilingual-json"


def test_verified_hotpot_jsonl_binds_record_provenance_and_is_deterministic(tmp_path):
    path = tmp_path / "hotpot.jsonl"
    _write_jsonl(
        path,
        [
            {
                "_id": "q1",
                "question": "Which source supports the answer?",
                "answer": "Alpha",
                "context": [["Doc A", ["Alpha is supported."]]],
                "supporting_facts": [["Doc A", 0]],
            },
            {
                "_id": "q2",
                "question": "What follows?",
                "answer": "Beta",
                "context": [["Doc B", ["Beta follows."]]],
                "supporting_facts": [["Doc B", 0]],
            },
        ],
    )
    manifest = _manifest(path, "hotpotqa", version="fixture-1", records=2)

    first = load_verified_benchmark_dataset(
        tmp_path,
        manifest,
        expected_dataset="hotpotqa",
        expected_version="fixture-1",
    )
    second = load_verified_benchmark_dataset(tmp_path, manifest)

    assert first.verification.verified is True
    assert first.execution_digest == second.execution_digest
    assert [example.example_id for example in first.examples] == ["q1", "q2"]
    assert first.examples[0].relevant_ids == ("Doc A",)
    metadata = first.examples[0].metadata
    assert metadata["dataset_version"] == "fixture-1"
    assert metadata["dataset_revision"] == "fixture-revision"
    assert metadata["dataset_manifest_sha256"] == manifest.manifest_digest
    assert metadata["source_path"] == "hotpot.jsonl"
    assert metadata["source_record"] == 1
    assert len(metadata["source_record_sha256"]) == 64


def test_verified_miracl_preserves_language_metadata(tmp_path):
    path = tmp_path / "miracl.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "hi-1",
                "query": "प्रश्न",
                "lang": "hi",
                "positive_passages": [{"docid": "d1", "text": "उत्तर संदर्भ"}],
            }
        ],
    )
    manifest = _manifest(path, "miracl", records=1)

    dataset = load_verified_benchmark_dataset(tmp_path, manifest)

    assert dataset.examples[0].metadata["language"] == "hi"
    assert dataset.examples[0].relevant_ids == ("d1",)
    assert dataset.examples[0].contexts == ("उत्तर संदर्भ",)


def test_manifest_tampering_fails_before_benchmark_adaptation(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"_id": "q1", "question": "safe", "answer": "a"}])
    manifest = _manifest(path, "hotpotqa", records=1)
    original = path.stat()
    tampered = json.dumps({"_id": "q1", "question": "evil", "answer": "a"}) + "\n"
    path.write_text(tampered, encoding="utf-8")
    # Preserve the original timestamp to ensure content identity, not mtime, is decisive.
    path.touch()
    if len(path.read_bytes()) == original.st_size:
        pass

    with pytest.raises(RuntimeError, match="digest does not match"):
        load_verified_benchmark_dataset(tmp_path, manifest)


def test_duplicate_example_ids_are_rejected_across_records(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(
        path,
        [
            {"_id": "same", "question": "first", "answer": "a"},
            {"_id": "same", "question": "second", "answer": "b"},
        ],
    )
    manifest = _manifest(path, "hotpotqa", records=2)

    with pytest.raises(ValueError, match="duplicate benchmark example id"):
        load_verified_benchmark_dataset(tmp_path, manifest)


def test_wrong_expected_dataset_or_version_fails_closed(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"_id": "q1", "question": "question", "answer": "answer"}])
    manifest = _manifest(path, "hotpotqa", version="v7", records=1)

    with pytest.raises(RuntimeError, match="dataset name"):
        load_verified_benchmark_dataset(tmp_path, manifest, expected_dataset="musique")
    with pytest.raises(RuntimeError, match="dataset version"):
        load_verified_benchmark_dataset(tmp_path, manifest, expected_version="v8")


def test_manifest_record_count_is_enforced_at_parse_time(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"_id": "q1", "question": "question", "answer": "answer"}])
    manifest = _manifest(path, "hotpotqa", records=2)

    with pytest.raises(RuntimeError, match="record count does not match"):
        load_verified_benchmark_dataset(tmp_path, manifest)
