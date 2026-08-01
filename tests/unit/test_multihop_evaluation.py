import pytest

from tools.multihop_evaluation import (
    MultiHopEvaluationExample,
    MultiHopEvaluationPrediction,
    SupportFact,
    aggregate_multihop_metrics,
    evaluate_multihop_example,
    normalize_answer,
    token_f1,
)


def example():
    return MultiHopEvaluationExample(
        example_id="ex-1",
        question="Which system won?",
        answers=("System A", "A"),
        support_facts=(
            SupportFact("doc-a", "page:1"),
            SupportFact("doc-b"),
        ),
        required_hops=2,
    )


def test_unicode_answer_normalization_and_alias_scoring():
    assert normalize_answer("  SYSTÉM—A  ") == "systém a"
    assert token_f1("System A wins", "System A") == pytest.approx(0.8)
    metrics = evaluate_multihop_example(
        example(),
        MultiHopEvaluationPrediction(answer="A", evidence=()),
    )
    assert metrics.answer_exact_match == 1.0
    assert metrics.answer_token_f1 == 1.0


def test_support_path_hop_and_lineage_metrics():
    prediction = MultiHopEvaluationPrediction(
        answer="System A",
        evidence=(
            {
                "doc_id": "doc-a",
                "page_number": 1,
                "hop_id": "q1",
                "source_id": "a-1",
            },
            {
                "doc_id": "doc-b",
                "hop_id": "q2",
                "source_id": "b-1",
            },
        ),
    )
    metrics = evaluate_multihop_example(example(), prediction)
    assert metrics.document_precision == 1.0
    assert metrics.document_recall == 1.0
    assert metrics.support_precision == 1.0
    assert metrics.support_recall == 1.0
    assert metrics.path_complete is True
    assert metrics.hop_coverage == 1.0
    assert metrics.citation_lineage_validity == 1.0
    assert metrics.answer_support_score == 1.0


def test_extra_and_locator_mismatched_evidence_reduce_scores():
    prediction = MultiHopEvaluationPrediction(
        answer="System B",
        evidence=(
            {
                "doc_id": "doc-a",
                "page_number": 2,
                "hop_id": "q1",
                "source_id": "wrong-page",
            },
            {"doc_id": "noise", "hop_id": "q2"},
        ),
    )
    metrics = evaluate_multihop_example(example(), prediction)
    assert metrics.document_precision == 0.5
    assert metrics.document_recall == 0.5
    assert metrics.support_recall == 0.0
    assert metrics.support_precision == 0.0
    assert metrics.path_complete is False
    assert metrics.citation_lineage_validity == 0.5


def test_empty_gold_and_prediction_are_defined_not_nan():
    empty = MultiHopEvaluationExample(
        example_id="empty",
        question="Unanswerable?",
        answers=("",),
        support_facts=(),
    )
    metrics = evaluate_multihop_example(
        empty,
        MultiHopEvaluationPrediction(answer="", evidence=(), abstained=True),
    )
    assert metrics.answer_exact_match == 1.0
    assert metrics.document_precision == 1.0
    assert metrics.document_recall == 1.0
    assert metrics.support_precision == 1.0
    assert metrics.support_recall == 1.0
    assert metrics.citation_lineage_validity == 1.0
    assert metrics.abstained is True


def test_aggregate_metrics_are_macro_averages():
    first = evaluate_multihop_example(
        example(),
        MultiHopEvaluationPrediction(
            answer="System A",
            evidence=(
                {
                    "doc_id": "doc-a",
                    "page_number": 1,
                    "hop_id": "q1",
                    "source_id": "a",
                },
                {"doc_id": "doc-b", "hop_id": "q2", "source_id": "b"},
            ),
        ),
    )
    second = evaluate_multihop_example(
        example(),
        MultiHopEvaluationPrediction(answer="", evidence=(), abstained=True),
    )
    report = aggregate_multihop_metrics((first, second))
    assert report.example_count == 2
    assert report.answer_exact_match == 0.5
    assert report.path_complete_rate == 0.5
    assert report.abstention_rate == 0.5
    assert report.hop_coverage == 0.5


def test_invalid_types_and_hostile_iterables_fail_closed():
    with pytest.raises(ValueError, match="required_hops"):
        MultiHopEvaluationExample("id", "q", ("a",), (), required_hops=True)
    with pytest.raises(ValueError, match="sequence"):
        MultiHopEvaluationPrediction("a", evidence="bad")

    class Hostile:
        def __iter__(self):
            yield object()
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        aggregate_multihop_metrics(Hostile())
