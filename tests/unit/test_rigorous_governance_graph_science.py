import hashlib

import pytest

from evaluation.multiple_comparisons import (
    benjamini_hochberg,
    holm_adjust,
    noninferiority_gate,
)
from tools.dataset_governance import DatasetCard, DatasetRegistry, DatasetSplit
from tools.evidence_graph import EvidenceEdge, EvidenceGraph, EvidenceNode
from tools.graph_retrieval import (
    evidence_source_diversity,
    path_completeness,
    retrieve_supporting_evidence,
)
from tools.runtime_calibrator import (
    CalibrationPoint,
    CalibrationProfile,
    RuntimeCalibratorRegistry,
)
from tools.scientific_evidence import (
    EffectEstimate,
    EffectMeasure,
    EvidenceConflict,
    QuestionFramework,
    ResearchQuestion,
    RiskLevel,
    RiskOfBias,
    ScientificEvidenceRecord,
    normalize_ratio_to_log,
)
from tools.table_provenance import TableCell, table_from_cells


def test_multiple_comparison_controls_and_noninferiority():
    raw = {"quality": 0.01, "latency": 0.03, "cost": 0.20}
    holm = {row.name: row for row in holm_adjust(raw)}
    bh = {row.name: row for row in benjamini_hochberg(raw)}
    assert holm["quality"].adjusted_p_value == pytest.approx(0.03)
    assert bh["quality"].adjusted_p_value == pytest.approx(0.03)
    assert holm["cost"].rejected is False

    gate = noninferiority_gate(
        estimate=-0.002,
        confidence_low=-0.008,
        confidence_high=0.003,
        margin=0.01,
        higher_is_better=True,
    )
    assert gate.passed is True


def test_dataset_registry_pins_checksum_license_and_split(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_bytes(b'{"query":"hello"}\n')
    checksum = hashlib.sha256(data.read_bytes()).hexdigest()
    card = DatasetCard(
        dataset_id="demo-rag",
        version="2026.08",
        license_id="CC-BY-4.0",
        languages=("en",),
        tasks=("retrieval",),
        splits=(DatasetSplit("train", "train.jsonl", checksum, examples=1),),
    )
    registry_path = tmp_path / "registry.json"
    registry = DatasetRegistry(registry_path)
    registry.register(card)
    assert registry.get("demo-rag", "2026.08").fingerprint() == card.fingerprint()
    assert registry.verify_local_split(
        "demo-rag", "2026.08", "train", base_dir=tmp_path
    )
    registry.require_license("demo-rag", "2026.08", ("CC-BY-4.0",))
    with pytest.raises(PermissionError):
        registry.require_license("demo-rag", "2026.08", ("MIT",))

    restored = DatasetRegistry(registry_path)
    assert restored.get("demo-rag", "2026.08") == card


def test_runtime_calibrator_selection_is_versioned_and_profile_specific(tmp_path):
    registry = RuntimeCalibratorRegistry(tmp_path / "calibrators.json")
    profile = CalibrationProfile(
        calibrator_id="isotonic",
        version="1.0.0",
        corpus_profile="scientific",
        benchmark="scifact",
        points=(
            CalibrationPoint(0.0, 0.05),
            CalibrationPoint(0.5, 0.4),
            CalibrationPoint(1.0, 0.95),
        ),
        answer_threshold=0.7,
    )
    registry.register(profile)
    registry.activate(
        corpus_profile="scientific",
        benchmark="scifact",
        calibrator_id="isotonic",
        version="1.0.0",
    )
    assert registry.calibrate(
        0.75, corpus_profile="scientific", benchmark="scifact"
    ) == pytest.approx(0.675)

    restored = RuntimeCalibratorRegistry(tmp_path / "calibrators.json")
    assert restored.selected(
        corpus_profile="scientific", benchmark="scifact"
    ).fingerprint() == profile.fingerprint()
    with pytest.raises(KeyError):
        restored.selected(corpus_profile="legal", benchmark="scifact")


def test_path_aware_graph_retrieval_preserves_lineage_and_source_diversity():
    graph = EvidenceGraph()
    nodes = (
        EvidenceNode("claim", "claim", "target"),
        EvidenceNode("a", "chunk", "evidence a", source_id="doc-a"),
        EvidenceNode("a-source", "source", "document a", source_id="doc-a"),
        EvidenceNode("b", "chunk", "evidence b", source_id="doc-b"),
        EvidenceNode("b-source", "source", "document b", source_id="doc-b"),
    )
    for node in nodes:
        graph.add_node(node)
    graph.add_edge(EvidenceEdge("claim", "a", "supports", 0.95))
    graph.add_edge(EvidenceEdge("a", "a-source", "derived_from", 1.0))
    graph.add_edge(EvidenceEdge("claim", "b", "supports", 0.85))
    graph.add_edge(EvidenceEdge("b", "b-source", "derived_from", 1.0))

    result = retrieve_supporting_evidence(graph, "claim", limit=4, per_source_cap=1)
    assert result.unique_sources == 2
    assert len(result.evidence) == 2
    assert evidence_source_diversity(result) == 1.0
    observed = [item.support_path for item in result.evidence]
    assert path_completeness(observed, result) == 1.0


def test_table_cell_provenance_validates_structure_and_citations():
    table = table_from_cells(
        table_id="t1",
        source_id="paper",
        page=4,
        cells=(
            TableCell("h", 0, 0, "Outcome", column_span=2, header=True),
            TableCell("r1c1", 1, 0, "Mortality"),
            TableCell("r1c2", 1, 1, "0.82"),
        ),
    )
    assert table.shape == (2, 2)
    assert table.row_text(1) == "Mortality | 0.82"
    assert table.citation_key("r1c2") == "paper:p4:table=t1:cell=r1c2"
    with pytest.raises(ValueError):
        table_from_cells(
            table_id="bad",
            source_id="paper",
            cells=(
                TableCell("a", 0, 0, "a", column_span=2),
                TableCell("b", 0, 1, "b"),
            ),
        )


def test_scientific_evidence_schema_captures_question_effect_bias_and_conflict():
    question = ResearchQuestion(
        question_id="q1",
        framework=QuestionFramework.PICO,
        population="adults with condition X",
        intervention_or_exposure="treatment A",
        comparator="placebo",
        outcomes=("mortality",),
    )
    effect = EffectEstimate(
        EffectMeasure.RISK_RATIO,
        0.82,
        confidence_low=0.70,
        confidence_high=0.96,
    )
    bias = RiskOfBias(overall=RiskLevel.SOME_CONCERNS, randomization=RiskLevel.LOW)
    record = ScientificEvidenceRecord(
        evidence_id="e1",
        source_id="paper-1",
        question_id=question.question_id,
        population=question.population,
        intervention_or_exposure=question.intervention_or_exposure,
        comparator=question.comparator,
        outcome="mortality",
        result_text="Treatment A reduced mortality relative to placebo.",
        effect=effect,
        risk_of_bias=bias,
        limitations=("single center",),
        provenance={"page": "7", "table": "2"},
        reviewed=True,
    )
    assert record.effect is effect
    assert normalize_ratio_to_log(effect) < 0.0

    conflict = EvidenceConflict(
        outcome="mortality",
        supporting_ids=("e1",),
        contradicting_ids=("e2",),
        rationale="Direction of effect differs across studies.",
    )
    assert conflict.supporting_ids == ("e1",)
