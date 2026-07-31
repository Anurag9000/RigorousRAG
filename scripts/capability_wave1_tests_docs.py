from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Correct core self-audit findings after the core generator writes its modules.
replace_once(
    "tools/hybrid_retrieval.py",
    '''                score += inverse_document_frequency * frequency * (self.k1 + 1.0) / normalizer\n                score *= 1.0 + math.log1p(query_frequency - 1)\n''',
    '''                score += (\n                    inverse_document_frequency\n                    * frequency\n                    * (self.k1 + 1.0)\n                    / normalizer\n                    * (1.0 + math.log1p(query_frequency - 1))\n                )\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    retrieval_mode: str = "hybrid",\n    reranker: str = "heuristic",\n    candidate_pool: int = 20,\n    diversity_lambda: float = 0.82,\n''',
    '''    retrieval_mode: str = "dense",\n    reranker: str = "none",\n    candidate_pool: Optional[int] = None,\n    diversity_lambda: float = 1.0,\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''    pool = _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS)\n    pool = max(requested, pool)\n''',
    '''    pool = (\n        requested\n        if candidate_pool is None\n        else _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS)\n    )\n    pool = max(requested, pool)\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''        source_id = metadata.get("doc_id")\n        if not isinstance(source_id, str):\n            continue\n''',
    '''        metadata_owner = metadata.get("owner_id")\n        source_id = metadata.get("doc_id")\n        if metadata_owner != owner or not isinstance(source_id, str):\n            continue\n        if document_id is not None and source_id != document_id:\n            continue\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''        per_source_limit=max(1, min(requested, 3)),\n''',
    '''        per_source_limit=(requested if document_id is not None else max(1, min(requested, 3))),\n''',
)
replace_once(
    "tools/rag_tool.py",
    '''                    "relevance": round(\n                        ranking_scores.get(chunk_id).score\n                        if chunk_id in ranking_scores\n                        else _finite_score(_safe_attr(chunk, "score", 0.0)),\n                        6,\n                    ),\n                    "retrieval_mode": mode,\n''',
    '''                    "relevance": round(\n                        _finite_score(_safe_attr(chunk, "score", 0.0)),\n                        6,\n                    ),\n                    "ranking_score": round(\n                        ranking_scores.get(chunk_id).score\n                        if chunk_id in ranking_scores\n                        else _finite_score(_safe_attr(chunk, "score", 0.0)),\n                        6,\n                    ),\n                    "retrieval_mode": mode,\n''',
)

Path("tests/unit/test_hybrid_retrieval.py").write_text(r'''from types import SimpleNamespace

import pytest

from tools.hybrid_retrieval import (
    BM25Index,
    RetrievalCandidate,
    SparseDocument,
    bm25_scores,
    rank_candidates,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from tools.reranking import CrossEncoderReranker, HeuristicReranker


def _candidate(candidate_id, text, score, source="doc"):
    return RetrievalCandidate(
        candidate_id=candidate_id,
        text=text,
        source_id=source,
        dense_score=score,
        metadata={"section_title": "Methods"},
    )


def test_bm25_and_hybrid_fusion_recover_lexically_relevant_evidence():
    candidates = [
        _candidate("dense", "unrelated neural architecture", 0.95, "doc-a"),
        _candidate("lexical", "randomized controlled trial mortality outcome", 0.40, "doc-b"),
        _candidate("mixed", "controlled trial design", 0.65, "doc-c"),
    ]
    lexical = bm25_scores("controlled trial mortality", candidates)
    assert max(lexical, key=lexical.get) == "lexical"

    ranked = rank_candidates(
        "controlled trial mortality",
        candidates,
        mode="hybrid",
        reranker=HeuristicReranker(),
        limit=3,
        diversity_lambda=1.0,
    )
    assert ranked[0].candidate.candidate_id == "lexical"
    assert all(0.0 <= item.score <= 1.0 for item in ranked)
    assert set(ranked[0].components) == {"dense", "lexical", "rrf", "reranker"}


def test_rrf_weighted_fusion_and_mmr_are_deterministic_and_source_bounded():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]]) == {
        "a": pytest.approx(1 / 61 + 1 / 62),
        "b": pytest.approx(1 / 62 + 1 / 61),
    }
    fused = weighted_score_fusion([{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 0.0}])
    assert fused == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}

    candidates = [
        _candidate("a1", "alpha beta", 1.0, "a"),
        _candidate("a2", "alpha beta duplicate", 0.9, "a"),
        _candidate("b1", "alpha gamma", 0.8, "b"),
    ]
    ranked = rank_candidates(
        "alpha",
        candidates,
        mode="dense",
        limit=3,
        diversity_lambda=0.5,
        per_source_limit=1,
    )
    assert [item.candidate.source_id for item in ranked] == ["a", "b"]


def test_exact_numeric_and_candidate_boundaries_fail_closed():
    candidate = _candidate("a", "evidence", 1.0)
    for value in (True, 1.0):
        with pytest.raises(ValueError):
            rank_candidates("query", [candidate], limit=value)
    with pytest.raises(ValueError):
        rank_candidates("query", [candidate], mode="unknown")
    with pytest.raises(ValueError):
        RetrievalCandidate(candidate_id="bad\n", text="x", source_id="doc")


def test_sparse_index_is_deterministic_and_uses_query_frequency_correctly():
    index = BM25Index(
        [
            SparseDocument("d1", "alpha beta alpha"),
            SparseDocument("d2", "beta gamma"),
            SparseDocument("d3", "delta"),
        ]
    )
    first = index.search("alpha alpha", top_k=2)
    second = index.search("alpha alpha", top_k=2)
    assert first == second
    assert first[0][0] == "d1"


def test_cross_encoder_is_lazy_normalized_and_falls_back_safely():
    calls = []

    class FakeModel:
        def predict(self, pairs, **kwargs):
            calls.append((pairs, kwargs))
            return [10.0, 5.0]

    reranker = CrossEncoderReranker(
        "local/model",
        model_factory=lambda _name: FakeModel(),
    )
    candidates = [_candidate("a", "alpha", 0.1), _candidate("b", "beta", 0.2)]
    assert reranker("query", candidates) == {"a": 1.0, "b": 0.0}
    assert len(calls) == 1

    failing = CrossEncoderReranker(
        "local/model",
        model_factory=lambda _name: SimpleNamespace(
            predict=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private"))
        ),
    )
    fallback = failing("alpha", candidates)
    assert set(fallback) == {"a", "b"}
''', encoding="utf-8")

Path("tests/unit/test_evaluation_foundation.py").write_text(r'''import json
from pathlib import Path

import pytest

from evaluation.datasets import load_beir_dataset
from evaluation.metrics import (
    aggregate_metrics,
    average_precision,
    citation_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
)
from evaluation.runner import run_benchmark
from experiments.manifest import (
    ExperimentResult,
    ResultStore,
    build_matrix,
    make_experiment_spec,
    write_manifest,
)


def _write_beir(root: Path):
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "d1", "title": "Alpha", "text": "alpha evidence"}),
                json.dumps({"_id": "d2", "title": "Beta", "text": "beta evidence"}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\n",
        encoding="utf-8",
    )


def test_beir_loader_normalizes_documents_queries_and_qrels(tmp_path):
    _write_beir(tmp_path)
    dataset = load_beir_dataset(tmp_path)
    assert dataset.documents["d1"].title == "Alpha"
    assert dataset.queries[0].relevant == {"d1": 2.0}

    link = tmp_path / "linked"
    try:
        link.symlink_to(tmp_path / "corpus.jsonl")
    except OSError:
        pytest.skip("Symbolic links unavailable")
    with pytest.raises(ValueError, match="links"):
        load_beir_dataset(link.parent / "linked")


def test_retrieval_and_citation_metrics_have_expected_values():
    ranked = ["d2", "d1", "d3"]
    qrels = {"d1": 2.0, "d3": 1.0}
    assert precision_at_k(ranked, qrels, 2) == 0.5
    assert recall_at_k(ranked, qrels, 2) == 0.5
    assert reciprocal_rank(ranked, qrels) == 0.5
    assert average_precision(ranked, qrels) == pytest.approx((0.5 + 2 / 3) / 2)
    assert 0 < ndcg_at_k(ranked, qrels, 3) <= 1
    metrics = retrieval_metrics(ranked, qrels, cutoffs=(1, 3))
    assert set(metrics) == {
        "mrr", "map", "hit_rate", "precision@1", "recall@1", "ndcg@1",
        "precision@3", "recall@3", "ndcg@3",
    }
    assert citation_metrics(["a", "b"], ["b", "c"]) == {
        "citation_precision": 0.5,
        "citation_recall": 0.5,
        "citation_f1": 0.5,
    }
    assert aggregate_metrics([{"x": 1.0}, {"x": 3.0}]) == {"x": 2.0}


def test_experiment_matrix_manifest_and_result_store_are_resumable(tmp_path):
    specs = build_matrix(
        {
            "retrieval_mode": ["dense", "hybrid"],
            "reranker": ["none", "heuristic"],
            "top_k": [5],
        }
    )
    assert len(specs) == 4
    assert len({spec.experiment_id for spec in specs}) == 4
    assert specs[0] == make_experiment_spec(specs[0].parameters)

    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, specs)
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 4

    store = ResultStore(tmp_path / "results")
    result = ExperimentResult(specs[0].experiment_id, {"mrr": 0.5})
    assert store.write(result) is True
    assert store.write(result) is False
    assert len(store.pending(specs)) == 3


def test_generic_benchmark_runner_skips_completed_specs(tmp_path):
    _write_beir(tmp_path / "dataset")
    dataset = load_beir_dataset(tmp_path / "dataset")
    specs = build_matrix({"retrieval_mode": ["lexical"], "top_k": [1, 2]})
    store = ResultStore(tmp_path / "results")

    def retrieve(query, parameters):
        return ["d1", "d2"][: int(parameters["top_k"])]

    first = run_benchmark(dataset, specs, retrieve, store, cutoffs=(1,))
    second = run_benchmark(dataset, specs, retrieve, store, cutoffs=(1,))
    assert len(first) == 2
    assert second == ()
''', encoding="utf-8")

Path("tests/unit/test_rag_tool_hybrid.py").write_text(r'''from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.rag_tool as rag_tool


def _chunk(chunk_id, text, score, doc_id, owner="alice"):
    return SimpleNamespace(
        id=chunk_id,
        text=text,
        score=score,
        metadata={
            "owner_id": owner,
            "doc_id": doc_id,
            "filename": f"{doc_id}.pdf",
            "parent_text": text,
            "section_title": "Methods",
            "page_number": 1,
        },
    )


def test_hybrid_mode_can_promote_lexically_specific_evidence(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = [
        _chunk("dense", "unrelated embedding result", 0.95, "doc-a"),
        _chunk("lexical", "randomized controlled trial mortality", 0.35, "doc-b"),
    ]
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    citations = rag_tool.search_uploaded_docs(
        "controlled trial mortality",
        owner_id="alice",
        n_results=2,
        retrieval_mode="hybrid",
        reranker="heuristic",
        candidate_pool=10,
        diversity_lambda=1.0,
    )
    assert [citation.chunk_id for citation in citations] == ["lexical", "dense"]
    assert citations[0].metadata["retrieval_mode"] == "hybrid"
    assert citations[0].metadata["reranker"] == "heuristic"
    assert "ranking_score" in citations[0].metadata
    assert set(citations[0].metadata["rank_components"]) == {
        "dense", "lexical", "rrf", "reranker"
    }
    assert rag.query.call_args.kwargs["n_results"] == 10


def test_cross_owner_candidates_never_influence_fusion_or_result_count(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = [
        _chunk("attacker", "exact query exact query", 1.0, "secret", owner="bob"),
        _chunk("safe", "query evidence", 0.5, "public"),
    ]
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)
    citations = rag_tool.search_uploaded_docs(
        "exact query",
        owner_id="alice",
        n_results=2,
        retrieval_mode="hybrid",
        reranker="heuristic",
    )
    assert [citation.chunk_id for citation in citations] == ["safe"]


def test_new_retrieval_arguments_fail_before_vector_initialization(monkeypatch):
    initializer = MagicMock(side_effect=AssertionError("vector layer should not initialize"))
    monkeypatch.setattr(rag_tool, "get_rag_layer", initializer)
    for arguments in (
        {"retrieval_mode": "bad"},
        {"reranker": "cross_encoder"},
        {"candidate_pool": True},
        {"candidate_pool": 51},
        {"diversity_lambda": True},
        {"diversity_lambda": -0.1},
        {"diversity_lambda": float("nan")},
    ):
        with pytest.raises(ValueError):
            rag_tool.search_uploaded_docs("query", owner_id="alice", **arguments)
    initializer.assert_not_called()
''', encoding="utf-8")

Path("docs/CAPABILITY_EXPANSION_ROADMAP.md").write_text(r'''# RigorousRAG capability expansion roadmap

This roadmap covers valuable models, architectures, pipelines, datasets, experiments,
features, and research tasks beyond the original remediation. Items are ordered by
foundational dependency, not novelty. Every implementation must preserve owner isolation,
bounded execution, authoritative server-side provenance, exact-head verification, and the
project's explicit scientific non-claims.

## Wave 1 — retrieval and evaluation foundation

Implemented in the first capability wave:

- canonical typed retrieval candidates and component score traces;
- BM25 lexical scoring over bounded candidate pools;
- deterministic reciprocal-rank fusion and normalized weighted fusion;
- MMR diversity with per-source caps;
- dependency-free scientific heuristic reranking;
- optional lazily loaded cross-encoder reranking with safe fallback;
- hybrid uploaded-document retrieval controls and score provenance;
- normalized BEIR dataset loading;
- precision, recall, MRR, MAP, NDCG, hit-rate, citation precision/recall/F1, and aggregate metrics;
- deterministic Cartesian experiment manifests and immutable per-run result files;
- resumable benchmark execution and an offline BM25 baseline CLI.

## Wave 2 — retrieval model families and indexing

Planned implementations:

### Sparse retrieval

- corpus-level BM25/BM25+ index for uploaded documents rather than candidate-pool-only lexical reranking;
- fielded BM25 for title, abstract, headings, captions, tables, references, and body text;
- scientific identifier-aware exact matching for DOI, arXiv ID, PMID, trial ID, gene/protein names, equations, and units;
- optional learned sparse adapters for SPLADE-family models;
- incremental sparse-index generations aligned transactionally with vector generations.

### Dense and multi-vector retrieval

- model registry and compatibility profiles for E5, BGE, GTE, SPECTER2, Instructor, and domain-specific biomedical/scientific encoders;
- embedding-dimension/schema migration and side-by-side index generations;
- late-interaction ColBERT-style token-vector retrieval;
- document, section, paragraph, sentence, caption, table, and claim-level multi-vector representations;
- query/document instruction templates and language/domain routing;
- embedding drift and nearest-neighbour stability diagnostics.

### Rerankers and fusion

- BGE/Cohere/Jina/cross-encoder model adapters behind explicit operator configuration;
- listwise and pairwise reranking experiments;
- learned fusion/calibration over dense, sparse, PageRank, venue, recency, citation-network, and source-quality features;
- dynamic candidate-pool sizing and early exit;
- source, document, section, and near-duplicate diversity constraints;
- confidence calibration and no-answer thresholds.

## Wave 3 — adaptive and corrective RAG

- deterministic query classification: lookup, synthesis, comparison, causal, methodological, contradiction, temporal, and multi-hop;
- step-back prompting, query rewriting, HyDE, multi-query, self-ask, and decomposition policies;
- CRAG-style retrieval grading and corrective web/document fallback;
- Self-RAG-style retrieve/continue/critique decisions without allowing the model to author provenance;
- FLARE-style uncertainty-triggered retrieval during generation;
- context compression, sentence selection, and evidence-budget allocation;
- answer planning separated from evidence collection and prose generation;
- explicit abstention, insufficient-evidence, and conflicting-evidence policies;
- deterministic fallback routes for provider outage, retrieval outage, and partial tool failure.

## Wave 4 — graph and multi-hop research

- owner-scoped evidence graph with documents, sections, claims, entities, methods, datasets, metrics, results, citations, and contradiction edges;
- citation-network ingestion and PageRank/HITS/community features;
- GraphRAG community summaries with server-controlled source membership;
- multi-hop path search, bridge-entity discovery, and path-level provenance;
- claim decomposition and claim-to-evidence bipartite graphs;
- temporal/version graphs for preprint, revision, conference, journal, correction, retraction, and replication relationships;
- RAPTOR-style hierarchical clusters and summaries with exact child-source lineage;
- graph-aware diversity and contradiction retrieval.

## Wave 5 — multimodal, layout, table, and formula evidence

- layout-aware PDF blocks with reading-order confidence;
- OCR word/line coordinates and scanned-caption localization;
- table detection, structure recovery, header propagation, cell provenance, and table QA;
- formula/equation extraction with page/bounding-box provenance;
- figure-panel segmentation, legend association, axis/label OCR, chart data extraction, and panel-level citations;
- document image embeddings and text-image late fusion;
- multimodal rerankers for page/figure relevance;
- DocVQA, ChartQA, TableQA, QASPER, and scientific-figure benchmark adapters;
- explicit uncertainty and “not visually recoverable” outcomes.

## Wave 6 — scientific quality and evidence intelligence

- source-type and publication-status classification;
- venue/peer-review/preprint/retraction/correction metadata;
- study-design and evidence-level extraction;
- sample, population, intervention, comparator, outcome, metric, uncertainty, and limitation schemas;
- statistical consistency checks for sample sizes, confidence intervals, p-values, units, and effect directions;
- citation-context classification: support, contrast, background, use, extension, and critique;
- replication and contradiction clustering;
- source quality/risk features that remain transparent rather than an opaque truth score;
- evidence tables, systematic-review exports, PRISMA-style screening logs, and living-review updates.

## Wave 7 — datasets and benchmark registry

Adapters and reproducible download/registration contracts for:

- BEIR: SciFact, NFCorpus, TREC-COVID, ArguAna, FiQA, Climate-FEVER, DBPedia, FEVER, HotpotQA, NQ, Quora, SCIDOCS, Touché, Signal-1M, BioASQ;
- multi-hop QA: HotpotQA, 2WikiMultiHopQA, MuSiQue, IIRC;
- scientific QA: QASPER, PubMedQA, BioASQ, SciQA, PaperQA-style corpora;
- fact verification: SciFact, FEVER, Climate-FEVER, PubHealth;
- citation/claim datasets: SciCite, CitationIntent, CORWA, CiteWorth;
- document/multimodal: DocVQA, InfographicVQA, ChartQA, PlotQA, TabFact, FinQA, TAT-QA;
- safety/robustness: prompt-injection corpora, poisoned-document sets, PII/secret fixtures, malformed-provider and adversarial-file suites;
- private user-authored benchmark packs with owner-scoped storage and no raw-query telemetry.

The registry will track license, source URL, checksum, version, split, language, domain,
task, required preprocessing, and redistribution restrictions.

## Wave 8 — evaluation and experiment system

- retrieval metrics at document, section, chunk, sentence, table, figure, and claim level;
- graded qrels, multi-hop path recall, evidence coverage, redundancy, source diversity, and calibration;
- answer correctness, semantic similarity, factuality, faithfulness, completeness, citation correctness, citation completeness, and citation placement;
- LLM-as-judge only as an optional recorded metric, never the sole gate;
- latency, throughput, memory, model-download, token, provider-cost, and energy proxies;
- robustness to paraphrase, typo, language, distractor, contradiction, stale version, prompt injection, and poisoned evidence;
- deterministic seeds, five-repeat support, confidence intervals, bootstrap significance, paired tests, and effect sizes;
- chunking, embedding, fusion, reranker, expansion, planner, compression, model, and context-budget ablations;
- resumable manifests, failed-run diagnosis, immutable results, comparison reports, and Pareto frontiers;
- regression thresholds for quality, safety, latency, and resource use.

## Wave 9 — agents, workflows, and user-facing research tasks

- planner/executor/reviewer state machine with bounded transitions;
- parallel independent evidence collection with deterministic merge;
- systematic literature review workflow;
- scoping review, rapid review, evidence map, and research-gap analysis;
- paper comparison, method/dataset/metric extraction, experiment reproduction planning, and protocol auditing;
- claim checking, citation recommendation, related-work construction, and bibliography deduplication;
- research timeline, author/institution/topic maps, and living alerts;
- batch jobs, saved experiment profiles, export bundles, and provenance-rich reports;
- CLI, API, and browser controls for retrieval mode, reranker, planner, dataset, and experiment profile.

## Wave 10 — scale and deployment architecture

- PostgreSQL/shared durable jobs, distributed queue, distributed rate limits, and worker leases;
- transactional outbox/saga coordination across registry, sparse index, vector index, graph, and retained files;
- object storage with encryption, versioning, integrity metadata, and lifecycle policy;
- dedicated model-serving processes, batching, GPU scheduling, circuit breakers, and warm pools;
- parser/OCR sandbox workers and malware scanning integration points;
- OpenTelemetry traces, Prometheus metrics, SLOs, capacity models, and audit dashboards;
- Kubernetes/nomad deployment profiles while preserving single-host Compose;
- backup/restore, migrations, disaster recovery, and index rebuild verification.
''', encoding="utf-8")

Path("docs/TODO.md").write_text(r'''# RigorousRAG exhaustive implementation backlog

Status meanings: **done** is implemented and focused-tested; **in progress** is the next
active wave; **planned** has an accepted design but is not yet source-complete; **external**
requires deployment infrastructure, credentials, licensed data, or operator policy.

## Current baseline and unfinished prior work

- [x] Consolidate all historical branches and PR work into the only branch, `main`.
- [x] Complete remediation passes 1–15 and preserve their audit records.
- [x] Restore one authoritative 16-job exact-head workflow and freshness-bound reporter.
- [ ] Obtain a complete successful exact-current-`main` 16-job matrix after the newest capability work.
- [ ] Fix every matrix failure and rerun the whole unchanged head.
- [ ] Raise branch-coverage floor only after measuring stable module-specific baselines.
- [ ] Run Windows storage/registry/upload suites after each new filesystem contract.
- [ ] Validate final container/Compose behavior on the same certified SHA.
- [ ] Perform final repository-wide line-by-line regression audit after all capability waves.

## Capability wave 1 — retrieval/evaluation foundation

- [x] Typed retrieval candidates and score traces.
- [x] Candidate-pool BM25 scoring.
- [x] Reciprocal-rank and weighted score fusion.
- [x] MMR diversity and per-source caps.
- [x] Dependency-free heuristic reranker.
- [x] Optional lazy cross-encoder adapter with safe fallback.
- [x] Uploaded-document dense/lexical/hybrid controls.
- [x] Raw dense relevance plus separate fused ranking provenance.
- [x] BEIR normalized loader.
- [x] Retrieval, ranking, citation, and aggregate metrics.
- [x] Deterministic experiment matrices and immutable resumable result store.
- [x] Generic benchmark runner and offline BM25 CLI.
- [ ] Add wave-one modules to the complete exact-head workflow result.

## Capability wave 2 — sparse/dense/multi-vector retrieval

- [ ] Owner-scoped persistent BM25/BM25+ uploaded-document index.
- [ ] Fielded sparse index for title, headings, body, captions, tables, and references.
- [ ] Sparse/vector generation transaction and rollback contracts.
- [ ] Embedding-model registry with dimensions, instructions, language, domain, and license metadata.
- [ ] Side-by-side vector generations and migration/reindex tooling.
- [ ] E5, BGE, GTE, SPECTER2, Instructor, and biomedical profiles.
- [ ] ColBERT/late-interaction adapter and token-vector storage contract.
- [ ] Multi-vector document/section/sentence/caption/table/claim representation.
- [ ] Learned fusion and calibrated abstention thresholds.
- [ ] Operator-configured cross-encoder/BGE/Jina rerankers with admission and model-serving limits.

## Capability wave 3 — adaptive/corrective agents

- [ ] Query/task classifier.
- [ ] Deterministic rewrite, step-back, decomposition, and self-ask plans.
- [ ] CRAG retrieval grading and correction routes.
- [ ] Self-RAG retrieve/critique decisions with server provenance authority.
- [ ] FLARE uncertainty-triggered retrieval.
- [ ] Evidence-budget allocation and context compression.
- [ ] Planner/executor/reviewer state machine.
- [ ] Provider-outage, retrieval-outage, contradiction, and no-evidence routes.
- [ ] Agent trace model with bounded public/private fields.

## Capability wave 4 — graph and multi-hop

- [ ] Evidence graph schema and owner isolation.
- [ ] Entity, claim, method, dataset, metric, result, citation, contradiction, and version edges.
- [ ] Multi-hop bridge/path retrieval and path-level citations.
- [ ] Citation network analytics and community detection.
- [ ] GraphRAG community summaries with exact membership lineage.
- [ ] RAPTOR hierarchical retrieval and child-source lineage.
- [ ] Temporal preprint/revision/publication/correction/retraction graph.

## Capability wave 5 — multimodal/layout/table/formula

- [ ] Layout-aware page blocks and reading order.
- [ ] OCR coordinates and scanned-caption localization.
- [ ] Table structure, cells, headers, and table QA.
- [ ] Formula extraction and equation provenance.
- [ ] Figure-panel segmentation, legends, axes, chart OCR, and panel citations.
- [ ] Text-image embeddings and multimodal reranking.
- [ ] Multimodal benchmark adapters and explicit uncertainty outcomes.

## Capability wave 6 — scientific evidence intelligence

- [ ] Publication/source/status classification.
- [ ] Study design and evidence-level extraction.
- [ ] PICO/population/method/dataset/metric/result/uncertainty schemas.
- [ ] Statistical/unit/effect-direction consistency checks.
- [ ] Citation intent and support/contrast classification.
- [ ] Replication, contradiction, and limitation clustering.
- [ ] Transparent source-risk features.
- [ ] Systematic-review evidence tables and PRISMA-compatible screening logs.

## Capability wave 7 — datasets

- [ ] Dataset registry with license/checksum/version/split/task/domain metadata.
- [ ] BEIR collection adapters beyond generic loader.
- [ ] Multi-hop QA adapters.
- [ ] Scientific QA and fact-verification adapters.
- [ ] Citation-intent/claim datasets.
- [ ] Document, table, chart, and figure QA adapters.
- [ ] Prompt-injection, poisoning, PII, malformed-file, and provider-adversarial packs.
- [ ] Owner-scoped private benchmark packs.

## Capability wave 8 — experiments and metrics

- [ ] Document/section/chunk/sentence/table/figure/claim metrics.
- [ ] Multi-hop path and evidence-coverage metrics.
- [ ] Answer correctness, faithfulness, completeness, and citation-placement metrics.
- [ ] Calibration, abstention, contradiction, and uncertainty metrics.
- [ ] Latency/throughput/memory/token/cost/resource metrics.
- [ ] Robustness transformations and adversarial evaluation.
- [ ] Repeats, bootstrap confidence intervals, significance tests, and effect sizes.
- [ ] Quality/safety/latency regression thresholds.
- [ ] HTML/JSON/CSV experiment reports and Pareto frontiers.

## Capability wave 9 — workflows and interface

- [ ] Literature/scoping/rapid review workflows.
- [ ] Evidence maps, gap analysis, protocol auditing, and reproduction plans.
- [ ] Claim checking, citation recommendation, and related-work construction.
- [ ] Saved retrieval/planner/experiment profiles.
- [ ] API, CLI, and browser controls for new architectures.
- [ ] Batch jobs, exports, reports, and living-review alerts.

## Capability wave 10 — scale and external deployment

- [ ] Shared SQL jobs/registry and distributed queue/leases.
- [ ] Transactional outbox/saga across all stores.
- [ ] Encrypted/versioned object storage.
- [ ] Dedicated model servers, GPU scheduling, batching, and circuit breakers.
- [ ] Parser sandbox and malware scanning integration. **external**
- [ ] Egress firewall, deployment DNS policy, TLS ingress, and secret manager. **external**
- [ ] OpenTelemetry/Prometheus/SLO dashboards.
- [ ] Backup, restore, migration, disaster recovery, and rebuild verification.

## Permanent scientific and safety non-claims

These are not TODOs to falsely mark complete: citations do not prove entailment; heuristic
or model analyses do not establish scientific truth; regex masking is not certified
de-identification; parser limits are not malware sandboxing; filesystem identity is not
host isolation or encryption; process-local components are not distributed exactly-once
infrastructure; and scientific conclusions require source inspection, expert review, and
replication.
''', encoding="utf-8")

# Main-only workflow policy and current documentation links.
replace_once(
    ".github/workflows/release-locks.yml",
    '    branches: [main, "agent/**"]\n',
    "    branches: [main]\n",
)
replace_once(
    "tests/unit/test_deployment_parity.py",
    '''    assert "  merge_group:\\n" in workflow\n    assert "paths:" not in workflow.split("permissions:", 1)[0]\n''',
    '''    assert "  merge_group:\\n" in workflow\n    assert "    branches: [main]\\n" in workflow\n    assert "agent/**" not in workflow\n    assert "paths:" not in workflow.split("permissions:", 1)[0]\n''',
)
replace_once(
    "README.md",
    '''See [Goals and Architecture](docs/GOALS_AND_ARCHITECTURE.md), [Security Model](docs/SECURITY.md), [Remediation Status](docs/REMEDIATION_STATUS.md), and the continuation-audit records in `docs/`.\n''',
    '''See [Goals and Architecture](docs/GOALS_AND_ARCHITECTURE.md), [Security Model](docs/SECURITY.md), [Remediation Status](docs/REMEDIATION_STATUS.md), [Capability Expansion Roadmap](docs/CAPABILITY_EXPANSION_ROADMAP.md), [Exhaustive TODO](docs/TODO.md), and the continuation-audit records in `docs/`.\n''',
)
replace_once(
    "README.md",
    '''**Current PR verification warning:** the remediation environment cannot clone or download the branch because `github.com` DNS resolution fails. No exact-current-head GitHub Actions result has been observed through the available connector. The remediation PR must remain draft until every configured check executes against the final head and all failures are corrected.\n''',
    '''**Exact-head verification warning:** only `main` is active and changes are committed directly to it. A release claim requires the authoritative 16-job workflow to succeed for the exact current `main` SHA; a stale result from an older commit is not certification.\n''',
)

# Update the high-level status without erasing the detailed historical audits.
status_path = Path("docs/REMEDIATION_STATUS.md")
status = status_path.read_text(encoding="utf-8")
status = status.replace("continued through **nine** regression/audit passes", "continued through **fifteen** remediation passes and capability wave one")
status = status.replace("- Branch: `agent/exhaustive-remediation`\n- Draft pull request: #1\n- Pull-request state: open, mergeable, draft\n", "- Default and only branch: `main`\n- Open pull requests: none\n- Development policy: verified commits directly to `main`\n")
status = status.replace("PR #1 must remain draft until one final exact head completes:", "One unchanged final `main` head must complete:")
status = status.replace("before the PR leaves draft.", "before a release claim is made.")
status += "\n\n## Capability expansion\n\nCapability wave one adds hybrid fusion, BM25 candidate scoring, MMR diversity, optional reranking, normalized BEIR datasets, comprehensive retrieval/citation metrics, deterministic experiment manifests, resumable result storage, and an offline benchmark runner. The accepted exhaustive future program is tracked in `CAPABILITY_EXPANSION_ROADMAP.md` and `TODO.md`.\n"
status_path.write_text(status, encoding="utf-8")
''', encoding="utf-8")
