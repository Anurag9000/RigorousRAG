import os
from pathlib import Path

import pytest

from evaluation import (
    EvaluationQuery,
    RetrievalResult,
    citation_metrics,
    evaluate_query,
    load_beir_dataset,
)
from experiments import ExperimentStore, build_manifest


def _write_beir(root: Path):
    root.mkdir(parents=True)
    (root / "qrels").mkdir()
    (root / "corpus.jsonl").write_text(
        '{"_id":"d1","title":"Alpha","text":"target evidence"}\n'
        '{"_id":"d2","title":"Beta","text":"other"}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text('{"_id":"q1","text":"target"}\n', encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8")


def test_beir_loader_and_path_safety(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_beir(dataset_root)
    dataset = load_beir_dataset(dataset_root)
    assert dataset.documents["d1"].title == "Alpha"
    assert dataset.queries[0].relevant == {"d1": 2.0}
    link = tmp_path / "linked-dataset"
    try:
        link.symlink_to(dataset_root, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links unavailable")
    with pytest.raises(ValueError, match="links|reparse"):
        load_beir_dataset(link)


def test_metrics_and_citation_metrics():
    query = EvaluationQuery("q", "question", {"d1": 2.0, "d2": 1.0})
    results = [RetrievalResult("d1", 1.0, 1), RetrievalResult("d3", 0.5, 2), RetrievalResult("d2", 0.2, 3)]
    metrics = evaluate_query(query, results, ks=(1, 3))
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["ndcg@3"] > 0.0
    citation = citation_metrics(cited_source_ids=["a", "bad"], supported_source_ids=["a"], required_source_ids=["a", "b"])
    assert citation["citation_precision"] == 0.5
    assert citation["unsupported_citation_rate"] == 0.5


def test_manifest_determinism_and_store_immutability(tmp_path):
    first = build_manifest({"mode": ["dense", "hybrid"], "k": [5, 10]}, prefix="retrieval")
    second = build_manifest({"k": [5, 10], "mode": ["dense", "hybrid"]}, prefix="retrieval")
    assert first == second
    assert len({run.run_id for run in first}) == 4
    store = ExperimentStore(tmp_path / "results.sqlite3")
    assert store.put(first[0].run_id, parameters=first[0].parameters, metrics={"mrr": 1.0}) is True
    assert store.put(first[0].run_id, parameters=first[0].parameters, metrics={"mrr": 0.0}) is False
    assert store.get(first[0].run_id)["metrics"] == {"mrr": 1.0}
