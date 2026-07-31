#!/usr/bin/env python3
"""Run deterministic offline BM25 retrieval baselines over BEIR-style datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import RetrievalResult, load_beir_dataset, run_retrieval_evaluation
from experiments import ExperimentStore, build_manifest
from tools.hybrid_retrieval import RetrievalCandidate, bm25_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="BEIR-format dataset directory")
    parser.add_argument("--output", default="data/retrieval_experiments.sqlite3")
    parser.add_argument("--top-k", type=int, nargs="+", default=[10])
    parser.add_argument("--k1", type=float, nargs="+", default=[1.2])
    parser.add_argument("--b", type=float, nargs="+", default=[0.75])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_beir_dataset(args.dataset)
    store = ExperimentStore(args.output)
    manifest = build_manifest({"b": args.b, "k1": args.k1, "top_k": args.top_k}, prefix="bm25")
    candidates = [RetrievalCandidate(document.document_id, document.text, document.document_id) for document in dataset.documents.values()]
    for run in manifest:
        if run.run_id in store.completed():
            continue
        k1, b, top_k = float(run.parameters["k1"]), float(run.parameters["b"]), int(run.parameters["top_k"])

        def retrieve(query: str, limit: int):
            scores = bm25_scores(query, candidates, k1=k1, b=b)
            ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:limit]
            return [RetrievalResult(identifier, score, rank) for rank, (identifier, score) in enumerate(ordered, start=1)]

        report = run_retrieval_evaluation(dataset, retrieve, top_k=top_k)
        store.put(run.run_id, parameters=run.parameters, metrics=report["metrics"], metadata={"dataset": dataset.name})
        print(json.dumps({"run_id": run.run_id, "metrics": report["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
