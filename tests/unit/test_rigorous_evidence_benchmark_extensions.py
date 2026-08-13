import pytest

from evaluation.benchmark_suite import run_benchmark_suite
from evaluation.promotion import MetricPromotionRule, evaluate_promotion
from evaluation.stability import PerturbationRun, rank_biased_overlap, stability_report
from tools.adaptive_compute import ComputeCaps, DifficultySignals, allocate_compute
from tools.benchmark_adapters import BenchmarkExample
from tools.counter_evidence import counter_evidence_coverage, find_counter_evidence
from tools.evidence_graph import EvidenceEdge, EvidenceGraph, EvidenceNode
from tools.evidence_quality import (
    EvidenceItem,
    evidence_quality_report,
    independent_source_ratio,
    minimal_evidence_cover,
    source_dependency_groups,
)
from tools.proof_carrying import ClaimProof, ProofCarryingAnswer, validate_proof


def test_executable_benchmark_suite_combines_retrieval_generation_and_latency():
    examples = (
        BenchmarkExample(
            "q1",
            "capital of France",
            answers=("Paris",),
            relevant_ids=("france",),
        ),
        BenchmarkExample(
            "q2",
            "capital of Japan",
            answers=("Tokyo",),
            relevant_ids=("japan",),
        ),
    )

    def retriever(example, top_k):
        relevant = example.relevant_ids[0]
        return ((relevant, 1.0), ("distractor", 0.1))[:top_k]

    def generator(example, retrieval):
        return example.answers[0]

    result = run_benchmark_suite(examples, retriever, generator=generator, top_k=2)
    assert len(result.rows) == 2
    assert result.aggregate["recall@1"] == 1.0
    assert result.aggregate["rouge_l"] == 1.0
    assert result.aggregate["chrf"] == 1.0
    assert result.aggregate["retrieval_latency_ms"] >= 0.0


def test_statistical_promotion_supports_quality_and_lower_is_better_metrics():
    baseline = {
        "quality": (0.60, 0.62, 0.61, 0.63, 0.60, 0.61),
        "latency": (100.0, 105.0, 98.0, 102.0, 101.0, 99.0),
    }
    candidate = {
        "quality": (0.70, 0.72, 0.71, 0.73, 0.70, 0.71),
        "latency": (90.0, 94.0, 88.0, 92.0, 91.0, 89.0),
    }
    decision = evaluate_promotion(
        baseline,
        candidate,
        (
            MetricPromotionRule("quality", require_significance=False),
            MetricPromotionRule("latency", higher_is_better=False, require_significance=False),
        ),
        resamples=200,
        permutations=200,
        seed=4,
    )
    assert decision.passed is True
    assert all(metric.mean_difference > 0.0 for metric in decision.metrics)


def test_evidence_independence_minimality_and_counter_evidence():
    evidence = (
        EvidenceItem("e1", "mirror-a", "root-a", frozenset({"c1"}), 0.9),
        EvidenceItem("e2", "mirror-b", "root-a", frozenset({"c1"}), 0.8),
        EvidenceItem("e3", "source-b", "root-b", frozenset({"c2"}), 0.85),
        EvidenceItem(
            "e4",
            "counter",
            "root-c",
            frozenset(),
            0.95,
            contradicts_claims=frozenset({"c1"}),
        ),
    )
    assert independent_source_ratio(evidence[:2]) == 0.5
    selected = minimal_evidence_cover(evidence, ("c1", "c2"))
    assert {item.evidence_id for item in selected} == {"e1", "e3"}
    report = evidence_quality_report(evidence, ("c1", "c2"))
    assert report.claim_coverage == 1.0
    assert report.contradiction_rate == 0.5
    assert source_dependency_groups(evidence)["root-a"] == ("e1", "e2")

    graph = EvidenceGraph()
    graph.add_node(EvidenceNode("c1", "claim"))
    graph.add_node(EvidenceNode("counter-node", "claim"))
    graph.add_edge(EvidenceEdge("c1", "counter-node", "contradicts"))
    counter = find_counter_evidence(graph, "c1", evidence)
    assert counter.graph_node_ids == ("counter-node",)
    assert counter.evidence_ids == ("e4",)
    assert counter_evidence_coverage(("c1", "c2"), (counter,)) == 0.5


def test_proof_carrying_answer_validates_evidence_and_path_adjacency():
    answer = ProofCarryingAnswer(
        answer="Paris is the capital of France.",
        claims=(
            ClaimProof(
                "c1",
                "Paris is the capital of France.",
                ("e1",),
                support_paths=(("c1", "chunk-1", "e1"),),
                confidence=0.95,
            ),
        ),
        evidence_catalog={"e1": {"source": "France reference"}},
    )
    valid = validate_proof(
        answer,
        known_edges=(("c1", "chunk-1"), ("chunk-1", "e1")),
    )
    assert valid.valid is True
    broken = validate_proof(answer, known_edges=(("c1", "wrong"),))
    assert broken.valid is False
    assert broken.path_resolution == 0.0


def test_adaptive_compute_is_bounded_and_monotonic():
    caps = ComputeCaps(
        max_retrieval_k=40,
        max_rerank_k=20,
        max_hops=5,
        max_query_expansions=4,
        max_generation_tokens=2000,
    )
    easy = allocate_compute(DifficultySignals(), caps=caps)
    hard = allocate_compute(
        DifficultySignals(1.0, 1.0, 1.0, 1.0, 1.0),
        caps=caps,
    )
    assert hard.difficulty == 1.0
    assert hard.retrieval_k >= easy.retrieval_k
    assert hard.rerank_k >= easy.rerank_k
    assert hard.max_hops == caps.max_hops
    assert hard.generation_tokens == caps.max_generation_tokens
    assert hard.rerank_k <= hard.retrieval_k


def test_index_perturbation_stability_reports_rank_and_answer_changes():
    assert rank_biased_overlap(("a", "b"), ("a", "b")) == pytest.approx(1.0)
    report = stability_report(
        (
            PerturbationRun("base", ("a", "b", "c"), "Answer one"),
            PerturbationRun("shuffle", ("a", "c", "b"), "Answer one"),
            PerturbationRun("drop", ("d", "a", "b"), "Different answer"),
        ),
        baseline="base",
    )
    assert 0.0 < report.retrieval_jaccard_mean <= 1.0
    assert 0.0 < report.retrieval_rbo_mean < 1.0
    assert report.exact_answer_stability == 0.5
