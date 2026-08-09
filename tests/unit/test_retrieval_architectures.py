from __future__ import annotations

import math

import pytest

from tools.retrieval_architectures import (
    ScoreCalibration,
    aggregate_multi_vector_scores,
    calibrate_score,
    calibrated_weighted_fusion,
    colbert_maxsim,
    cosine_similarity,
    select_matryoshka_dimensions,
    splade_sparse_similarity,
)


def test_temperature_calibration_is_monotonic_and_neutral_at_half():
    assert calibrate_score(0.5) == pytest.approx(0.5)
    assert calibrate_score(0.2) < calibrate_score(0.8)
    assert calibrate_score(0.8, temperature=2.0) < calibrate_score(0.8)
    assert ScoreCalibration(bias=1.0).apply(0.5) > 0.5


def test_calibrated_fusion_applies_component_specific_calibration():
    fused = calibrated_weighted_fusion(
        {
            "dense": {"a": 0.8, "b": 0.4},
            "sparse": {"a": 0.2, "b": 0.9},
        },
        weights={"dense": 2.0, "sparse": 1.0},
        calibrations={
            "dense": ScoreCalibration(temperature=2.0),
            "sparse": ScoreCalibration(),
        },
    )
    expected_a = (
        2.0 * calibrate_score(0.8, temperature=2.0) + calibrate_score(0.2)
    ) / 3.0
    assert fused["a"] == pytest.approx(expected_a)
    assert 0.0 <= fused["b"] <= 1.0


def test_splade_sparse_similarity_is_cosine_normalized_and_nonnegative():
    exact = splade_sparse_similarity(
        {"retrieval": 2.0, "rag": 1.0},
        {"retrieval": 2.0, "rag": 1.0},
    )
    partial = splade_sparse_similarity(
        {"retrieval": 2.0, "rag": 1.0},
        {"retrieval": 1.0, "other": 3.0},
    )
    assert exact == pytest.approx(1.0)
    assert 0.0 < partial < exact
    assert splade_sparse_similarity({}, {"a": 1.0}) == 0.0


def test_colbert_maxsim_rewards_per_query_token_matches():
    query = ((1.0, 0.0), (0.0, 1.0))
    exact_document = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    partial_document = ((1.0, 0.0), (1.0, 0.0))

    exact = colbert_maxsim(query, exact_document)
    partial = colbert_maxsim(query, partial_document)

    assert exact == pytest.approx(1.0)
    assert 0.0 <= partial < exact
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_multivector_aggregation_supports_max_mean_and_top_mean():
    scores = [0.1, 0.9, 0.8, 0.2]
    assert aggregate_multi_vector_scores(scores, mode="max") == pytest.approx(0.9)
    assert aggregate_multi_vector_scores(scores, mode="mean") == pytest.approx(0.5)
    assert aggregate_multi_vector_scores(
        scores, mode="top_mean", top_n=2
    ) == pytest.approx(0.85)


def test_matryoshka_selection_grows_with_budget_complexity_and_uncertainty():
    available = [128, 256, 512, 768]
    cheap = select_matryoshka_dimensions(
        available,
        budget=0.0,
        query_complexity=0.0,
        uncertainty=0.0,
    )
    hard = select_matryoshka_dimensions(
        available,
        budget=1.0,
        query_complexity=1.0,
        uncertainty=1.0,
    )
    middle = select_matryoshka_dimensions(
        available,
        budget=0.5,
        query_complexity=0.5,
        uncertainty=0.5,
    )
    assert cheap.dimensions == 128
    assert hard.dimensions == 768
    assert cheap.dimensions <= middle.dimensions <= hard.dimensions
    assert middle.available_dimensions == tuple(available)


@pytest.mark.parametrize(
    "call",
    [
        lambda: calibrate_score(float("nan")),
        lambda: calibrated_weighted_fusion({"dense": {"a": 2.0}}),
        lambda: splade_sparse_similarity({"x": -1.0}, {"x": 1.0}),
        lambda: cosine_similarity((0.0, 0.0), (1.0, 0.0)),
        lambda: colbert_maxsim(((1.0, 0.0),), ((1.0, 0.0, 0.0),)),
        lambda: aggregate_multi_vector_scores([0.5, math.inf]),
        lambda: select_matryoshka_dimensions([], budget=0.5),
    ],
)
def test_advanced_retrieval_boundaries_fail_closed(call):
    with pytest.raises(ValueError):
        call()
