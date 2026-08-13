from pathlib import Path

import pytest

from evaluation.efficiency import estimate_cost, summarize_latencies, throughput
from evaluation.generation_metrics import chrf, rouge_l, unsupported_claim_rate
from evaluation.robustness import counterfactual_citation_report, metadata_poisoning_report
from tools.adapter_registry import AdapterRegistry, AdapterVersion
from tools.benchmark_adapters import adapt_record
from tools.calibration import (
    brier_score,
    confidence_from_signals,
    expected_calibration_error,
    optimize_threshold,
    reliability_bins,
    selective_curve,
    selective_decision,
)
from tools.evidence_graph import EvidenceEdge, EvidenceGraph, EvidenceNode
from tools.governance import ACL, AccessContext, LineageRegistry, filter_authorized, redact_pii
from tools.multimodal import Modality, MultiModalChunk, RankedChunk, reciprocal_rank_fusion
from tools.observability import TraceRecorder
from tools.scientific_rag import (
    canonical_section,
    extract_abbreviations,
    extract_equations,
    normalize_unit_spacing,
)


def test_calibration_metrics_and_selective_answering():
    confidences = [0.1, 0.4, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    bins = reliability_bins(confidences, labels, bins=2)
    assert sum(item.count for item in bins) == 4
    assert expected_calibration_error(confidences, labels, bins=2) == pytest.approx(0.2)
    assert brier_score(confidences, labels) == pytest.approx(0.055)

    curve = selective_curve(confidences, labels, thresholds=[0.0, 0.8])
    assert curve[0].coverage == 1.0
    assert curve[1].coverage == 0.5
    assert curve[1].accuracy == 1.0

    threshold = optimize_threshold(
        confidences,
        labels,
        false_positive_cost=4.0,
        false_negative_cost=1.0,
    )
    assert 0.0 <= threshold.threshold <= 1.0
    assert threshold.expected_cost >= 0.0

    confidence = confidence_from_signals(
        {"retrieval": 0.8, "citations": 1.0, "consistency": 0.6},
        weights={"retrieval": 2.0, "citations": 1.0, "consistency": 1.0},
    )
    assert confidence == pytest.approx(0.8)
    decision = selective_decision(
        confidence,
        threshold=0.75,
        minimum_citation_coverage=0.8,
        citation_coverage=0.9,
    )
    assert decision.answer is True


def test_observability_span_usage_failure_and_exports(tmp_path):
    recorder = TraceRecorder(max_events=4)
    with recorder.span("retrieve", trace_id="trace-1", input_items=1) as span:
        span.output_items = 3
        span.add_usage(prompt_tokens=10, completion_tokens=5, estimated_cost_usd=0.001)

    with pytest.raises(RuntimeError):
        with recorder.span("rerank", trace_id="trace-1"):
            raise RuntimeError("boom")

    snapshot = recorder.snapshot()
    assert snapshot.events == 2
    assert snapshot.failed_events == 1
    assert snapshot.prompt_tokens == 10
    assert "retrieve" in snapshot.by_stage

    path = tmp_path / "trace.jsonl"
    assert recorder.write_jsonl(path) == 2
    assert path.read_text(encoding="utf-8").count("\n") == 2
    metrics = recorder.prometheus_text()
    assert "rigorousrag_events_total 2" in metrics
    assert 'stage="retrieve"' in metrics


def test_governance_redaction_acl_and_revocation():
    result = redact_pii(
        "Email alice@example.com, call +1 202-555-0172, card 4111 1111 1111 1111."
    )
    assert "alice@example.com" not in result.text
    assert "4111 1111 1111 1111" not in result.text
    assert len(result.redactions) >= 2

    class Item:
        def __init__(self, name, acl):
            self.name = name
            self.acl = acl

    items = [
        Item("public", ACL(public=True)),
        Item("team", ACL(groups=frozenset({"research"}))),
        Item("secret", ACL(owners=frozenset({"bob"}))),
    ]
    allowed = filter_authorized(
        items,
        AccessContext("alice", groups=frozenset({"research"})),
    )
    assert [item.name for item in allowed] == ["public", "team"]

    lineage = LineageRegistry()
    lineage.link("source", "chunk")
    lineage.link("chunk", "embedding")
    lineage.link("embedding", "answer")
    affected = lineage.revoke("source", reason="source deleted", revoked_at=1.0)
    assert affected == {"source", "chunk", "embedding", "answer"}
    assert lineage.is_revoked("answer")


def test_evidence_graph_support_and_contradictions():
    graph = EvidenceGraph()
    for node in (
        EvidenceNode("claim", "claim", "A"),
        EvidenceNode("chunk", "chunk", "evidence"),
        EvidenceNode("source", "source", "paper"),
        EvidenceNode("counter", "claim", "not A"),
    ):
        graph.add_node(node)
    graph.add_edge(EvidenceEdge("claim", "chunk", "supports", 0.9))
    graph.add_edge(EvidenceEdge("chunk", "source", "derived_from", 1.0))
    graph.add_edge(EvidenceEdge("claim", "counter", "contradicts", 1.0))

    paths = graph.support_paths("claim")
    assert paths
    assert paths[0].nodes[-1] in {"chunk", "source"}
    assert graph.evidence_coverage(["claim"]) == 1.0
    assert "counter" in graph.contradictions("claim")


def test_benchmark_adapters_cover_major_formats():
    hotpot = adapt_record(
        "hotpotqa",
        {
            "_id": "h1",
            "question": "Who?",
            "answer": "Ada",
            "supporting_facts": [["Doc A", 0]],
            "context": [["Doc A", ["Ada wrote it."]], ["Doc B", ["Other."]]],
        },
    )
    assert hotpot.query == "Who?"
    assert hotpot.relevant_ids == ("Doc A",)

    scifact = adapt_record(
        "scifact",
        {"id": 7, "claim": "Claim", "label": "SUPPORT", "evidence": {"42": [[0]]}},
    )
    assert scifact.example_id == "7"
    assert scifact.relevant_ids == ("42",)

    miracl = adapt_record(
        "miracl",
        {
            "query_id": "m1",
            "query": "bonjour",
            "lang": "fr",
            "positive_passages": [{"docid": "d1", "text": "texte"}],
        },
    )
    assert miracl.metadata["language"] == "fr"


def test_adapter_registry_versioning_persistence_and_rollback(tmp_path):
    path = tmp_path / "registry.json"
    registry = AdapterRegistry(path)
    checksum1 = AdapterRegistry.checksum_bytes(b"one")
    checksum2 = AdapterRegistry.checksum_bytes(b"two")
    registry.register(
        AdapterVersion("legal-embed", "1.0.0", "embedding", "file://one", checksum1, tags=("legal",))
    )
    registry.register(
        AdapterVersion("legal-embed", "1.1.0", "embedding", "file://two", checksum2, tags=("legal",))
    )
    registry.promote("legal-embed", "1.1.0")
    assert registry.active("legal-embed").version == "1.1.0"
    assert registry.rollback("legal-embed").version == "1.0.0"

    restored = AdapterRegistry(path)
    assert restored.active("legal-embed").version == "1.0.0"
    assert restored.compatible(kind="embedding", tags=("legal",))


def test_generation_efficiency_and_robustness_metrics():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert 0.0 < chrf("colour", "color") < 1.0
    rate = unsupported_claim_rate(
        "Paris is in France. Mars has oceans.",
        ["Paris is the capital of France."],
        threshold=0.5,
    )
    assert rate == 0.5

    summary = summarize_latencies([10, 20, 30, 40])
    assert summary.count == 4
    assert summary.median_ms == 25
    usage = estimate_cost(
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        prompt_usd_per_million=2.0,
        completion_usd_per_million=4.0,
    )
    assert usage.cost_usd == pytest.approx(4.0)
    assert throughput(10, 2.0) == 5.0

    citation_report = counterfactual_citation_report(
        expected_source_ids=["good"],
        decoy_source_ids=["decoy"],
        cited_source_ids=["good"],
    )
    assert citation_report.expected_citation_fraction == 1.0
    ranking_report = metadata_poisoning_report(
        clean_ranking=["good", "other"],
        perturbed_ranking=["decoy", "other"],
        relevant_ids=["good"],
        k=2,
    )
    assert ranking_report.recall_drop == 1.0


def test_multimodal_fusion_preserves_provenance():
    text = MultiModalChunk("t1", Modality.TEXT, "text", "doc", page=1)
    table = MultiModalChunk("tab1", Modality.TABLE, "table", "doc", page=2)
    fused = reciprocal_rank_fusion(
        [
            [RankedChunk(text, 1.0, 1), RankedChunk(table, 0.8, 2)],
            [RankedChunk(table, 1.0, 1), RankedChunk(text, 0.7, 2)],
        ],
        modality_weights={Modality.TABLE: 2.0},
    )
    assert fused[0].chunk.chunk_id == "tab1"
    assert fused[0].chunk.citation_key().startswith("doc:p2")


def test_scientific_normalization_helpers():
    text = "Retrieval Augmented Generation (RAG) uses $p(x|q)$ at 10kg."
    assert extract_abbreviations(text)["RAG"] == "Retrieval Augmented Generation"
    equations = extract_equations(text)
    assert equations and equations[0].text == "$p(x|q)$"
    assert canonical_section("3.2 Materials and Methods") == "methods"
    assert "10 kg" in normalize_unit_spacing(text)
