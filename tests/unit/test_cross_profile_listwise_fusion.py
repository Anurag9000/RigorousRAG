from __future__ import annotations

import hashlib

import pytest

from training.cross_profile_listwise_fusion import (
    FusionRankingCandidate,
    FusionRankingQuery,
    LearnedListwiseFusionArtifact,
    ListwiseFusionTrainingConfig,
    ListwiseFusionTrainingSpec,
    advance_listwise_training,
    fit_listwise_fusion_weights,
    initialize_listwise_training,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec() -> ListwiseFusionTrainingSpec:
    return ListwiseFusionTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=sha("contract"),
        calibration_artifact_sha256s=(("dense", sha("dense-cal")), ("sparse", sha("sparse-cal"))),
        train_split_sha256=sha("train"),
        validation_split_sha256=sha("validation"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=ListwiseFusionTrainingConfig(
            epochs=50,
            batch_size=1,
            learning_rate=0.2,
            patience=8,
            min_delta=1e-8,
            seed=13,
        ),
    )


def query(tag: str, dense_good: bool = True) -> FusionRankingQuery:
    if dense_good:
        candidates = (
            FusionRankingCandidate("good", {"dense": 0.95, "sparse": 0.55}, 2.0),
            FusionRankingCandidate("mid", {"dense": 0.55, "sparse": 0.60}, 1.0),
            FusionRankingCandidate("bad", {"dense": 0.05, "sparse": 0.45}, 0.0),
        )
    else:
        candidates = (
            FusionRankingCandidate("good", {"dense": 0.90, "sparse": 0.52}, 2.0),
            FusionRankingCandidate("mid", {"dense": 0.50, "sparse": 0.58}, 1.0),
            FusionRankingCandidate("bad", {"dense": 0.10, "sparse": 0.48}, 0.0),
        )
    return FusionRankingQuery(sha(f"query:{tag}"), candidates)


def train_queries():
    return (query("a"), query("b", False), query("c"))


def validation_queries():
    return (query("validation-a"), query("validation-b", False))


def test_listwise_training_learns_to_favor_rank_informative_profile() -> None:
    artifact = fit_listwise_fusion_weights(spec(), train_queries(), validation_queries())
    weights = dict(artifact.profile_weights)
    assert weights["dense"] > weights["sparse"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert artifact.score({"dense": 0.95, "sparse": 0.5}) > artifact.score({"dense": 0.05, "sparse": 0.5})


def test_listwise_mid_batch_resume_is_exact() -> None:
    training_spec = spec()
    train = train_queries()
    valid = validation_queries()
    direct = fit_listwise_fusion_weights(training_spec, train, valid)
    state = initialize_listwise_training(training_spec, train, valid)
    while not state.completed:
        state = advance_listwise_training(training_spec, state, train, valid, max_batches=1)
    resumed = LearnedListwiseFusionArtifact.build(training_spec, state)
    assert resumed.artifact_sha256 == direct.artifact_sha256
    assert resumed.profile_weights == direct.profile_weights


def test_listwise_resume_rejects_query_data_drift() -> None:
    training_spec = spec()
    train = train_queries()
    valid = validation_queries()
    state = initialize_listwise_training(training_spec, train, valid)
    changed = train + (query("new"),)
    with pytest.raises(ValueError, match="differs from resumable state"):
        advance_listwise_training(training_spec, state, changed, valid, max_batches=1)


def test_ranking_query_requires_relevance_order_information() -> None:
    with pytest.raises(ValueError, match="two relevance grades"):
        FusionRankingQuery(
            sha("flat-query"),
            (
                FusionRankingCandidate("a", {"dense": 0.9, "sparse": 0.1}, 1.0),
                FusionRankingCandidate("b", {"dense": 0.1, "sparse": 0.9}, 1.0),
            ),
        )


def test_listwise_training_requires_exact_profile_features() -> None:
    training_spec = spec()
    bad = FusionRankingQuery(
        sha("bad-query"),
        (
            FusionRankingCandidate("a", {"dense": 0.9}, 1.0),
            FusionRankingCandidate("b", {"dense": 0.1}, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="exactly cover"):
        initialize_listwise_training(training_spec, (bad,), validation_queries())
