from __future__ import annotations

import hashlib

import pytest

from tools.corpus_fusion import RetrievalCandidate
from tools.cross_profile_fusion import (
    CalibrationContract,
    CrossProfileFusionMode,
    CrossProfileFusionPolicy,
    ProfileRankedList,
    RetrieverScoreProfile,
    ScoreCalibrationExample,
    ScoreDirection,
    evaluate_isotonic_calibrator,
    fit_isotonic_calibrator,
    fuse_cross_profile_rankings,
)
from tools.cross_profile_fusion_governance import (
    GovernedCrossProfileFusionReceipt,
    run_governed_cross_profile_fusion,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def contract(tag: str = "shared") -> CalibrationContract:
    return CalibrationContract(
        dataset_manifest_sha256=sha(f"dataset:{tag}"),
        split_sha256=sha(f"split:{tag}"),
        relevance_contract_sha256=sha(f"relevance:{tag}"),
        candidate_universe_sha256=sha(f"universe:{tag}"),
        domain_id="scientific",
        cohort_id="heldout-v1",
    )


def profile(profile_id: str, family: str, *, lower: bool = False) -> RetrieverScoreProfile:
    return RetrieverScoreProfile(
        profile_id=profile_id,
        family=family,
        scoring_contract_sha256=sha(f"score:{profile_id}"),
        model_profile_sha256=sha(f"model:{profile_id}"),
        score_direction=(
            ScoreDirection.LOWER_IS_BETTER if lower else ScoreDirection.HIGHER_IS_BETTER
        ),
    )


def calibrator(p: RetrieverScoreProfile, c: CalibrationContract, values):
    return fit_isotonic_calibrator(
        profile=p,
        contract=c,
        examples=tuple(ScoreCalibrationExample(*value) for value in values),
    )


def candidate(
    *,
    cid: str,
    retriever: str,
    document: str,
    chunk: str,
    rank: int,
    score: float,
    corpus: str = "papers",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        candidate_id=cid,
        corpus_id=corpus,
        retriever_id=retriever,
        document_id=document,
        chunk_id=chunk,
        rank=rank,
        raw_score=score,
        source_id=document,
    )


def test_isotonic_calibrator_is_monotone_for_higher_scores() -> None:
    p = profile("dense", "dense")
    artifact = calibrator(
        p,
        contract(),
        ((0.0, False), (0.2, False), (0.7, True), (1.0, True)),
    )
    predictions = [artifact.predict(score) for score in (-1.0, 0.1, 0.5, 0.9, 2.0)]
    assert predictions == sorted(predictions)
    diagnostics = evaluate_isotonic_calibrator(
        artifact,
        (
            ScoreCalibrationExample(0.1, False),
            ScoreCalibrationExample(0.9, True),
        ),
        bin_count=2,
    )
    assert 0.0 <= diagnostics.brier <= 1.0
    assert 0.0 <= diagnostics.ece <= 1.0


def test_lower_is_better_profile_orients_calibration_correctly() -> None:
    p = profile("distance", "vector-distance", lower=True)
    artifact = calibrator(
        p,
        contract(),
        ((10.0, False), (7.0, False), (2.0, True), (1.0, True)),
    )
    assert artifact.predict(1.5) > artifact.predict(8.0)


def test_calibrated_fusion_can_override_rank_when_profiles_agree_on_relevance() -> None:
    c = contract()
    dense = profile("dense", "dense")
    sparse = profile("splade", "learned-sparse")
    calibrators = {
        dense.profile_id: calibrator(
            dense, c, ((0.0, False), (0.4, False), (0.6, True), (1.0, True))
        ),
        sparse.profile_id: calibrator(
            sparse, c, ((0.0, False), (5.0, False), (10.0, True), (15.0, True))
        ),
    }
    lists = (
        ProfileRankedList(
            "dense-list",
            dense,
            (
                candidate(cid="dense-b", retriever="dense", document="doc-b", chunk="b", rank=1, score=0.1),
                candidate(cid="dense-a", retriever="dense", document="doc-a", chunk="a", rank=2, score=0.9),
            ),
        ),
        ProfileRankedList(
            "sparse-list",
            sparse,
            (
                candidate(cid="sparse-b", retriever="splade", document="doc-b", chunk="b", rank=1, score=1.0),
                candidate(cid="sparse-a", retriever="splade", document="doc-a", chunk="a", rank=2, score=14.0),
            ),
        ),
    )
    result = fuse_cross_profile_rankings(
        lists,
        calibrators=calibrators,
        policy=CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT),
    )
    assert result.mode is CrossProfileFusionMode.CALIBRATED_LOGIT
    assert result.candidates[0].candidate.document_id == "doc-a"
    assert result.candidates[0].fused_probability is not None


def test_auto_mode_falls_back_to_rank_only_when_a_profile_is_uncalibrated() -> None:
    c = contract()
    dense = profile("dense", "dense")
    sparse = profile("splade", "learned-sparse")
    dense_calibrator = calibrator(
        dense, c, ((0.0, False), (0.4, False), (0.6, True), (1.0, True))
    )
    lists = (
        ProfileRankedList(
            "dense-list",
            dense,
            (
                candidate(cid="dense-b", retriever="dense", document="doc-b", chunk="b", rank=1, score=-10_000.0),
                candidate(cid="dense-a", retriever="dense", document="doc-a", chunk="a", rank=2, score=10_000.0),
            ),
        ),
        ProfileRankedList(
            "sparse-list",
            sparse,
            (
                candidate(cid="sparse-b", retriever="splade", document="doc-b", chunk="b", rank=1, score=-1_000_000.0),
                candidate(cid="sparse-a", retriever="splade", document="doc-a", chunk="a", rank=2, score=1_000_000.0),
            ),
        ),
    )
    result = fuse_cross_profile_rankings(lists, calibrators={"dense": dense_calibrator})
    assert result.mode is CrossProfileFusionMode.RRF_ONLY
    assert result.calibration_contract_sha256 is None
    assert result.candidates[0].candidate.document_id == "doc-b"
    assert all(item.fused_probability is None for item in result.candidates)


def test_strict_calibrated_mode_rejects_missing_or_incompatible_calibrators() -> None:
    dense = profile("dense", "dense")
    sparse = profile("splade", "learned-sparse")
    lists = (
        ProfileRankedList(
            "dense-list",
            dense,
            (candidate(cid="d", retriever="dense", document="doc", chunk="c", rank=1, score=0.9),),
        ),
        ProfileRankedList(
            "sparse-list",
            sparse,
            (candidate(cid="s", retriever="splade", document="doc", chunk="c", rank=1, score=12.0),),
        ),
    )
    dense_calibrator = calibrator(dense, contract("a"), ((0.0, False), (1.0, True)))
    policy = CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT)
    with pytest.raises(ValueError, match="every profile calibrator"):
        fuse_cross_profile_rankings(lists, calibrators={"dense": dense_calibrator}, policy=policy)

    sparse_calibrator = calibrator(sparse, contract("b"), ((0.0, False), (15.0, True)))
    with pytest.raises(ValueError, match="compatible calibrators"):
        fuse_cross_profile_rankings(
            lists,
            calibrators={"dense": dense_calibrator, "splade": sparse_calibrator},
            policy=policy,
        )


def test_duplicate_lists_from_same_profile_contribute_only_once_per_candidate() -> None:
    c = contract()
    dense = profile("dense", "dense")
    artifact = calibrator(dense, c, ((0.0, False), (1.0, True)))
    lists = (
        ProfileRankedList(
            "shard-a",
            dense,
            (candidate(cid="a1", retriever="dense", document="doc", chunk="c", rank=1, score=0.9, corpus="a"),),
        ),
        ProfileRankedList(
            "shard-b",
            dense,
            (candidate(cid="a2", retriever="dense", document="doc", chunk="c", rank=1, score=0.8, corpus="b"),),
        ),
    )
    result = fuse_cross_profile_rankings(
        lists,
        calibrators={"dense": artifact},
        policy=CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT),
    )
    assert len(result.candidates) == 1
    assert len(result.candidates[0].calibrated_contributions) == 1


def test_governed_receipt_is_deterministic_and_tamper_evident() -> None:
    dense = profile("dense", "dense")
    ranked = (
        ProfileRankedList(
            "dense-list",
            dense,
            (candidate(cid="d", retriever="dense", document="doc", chunk="c", rank=1, score=0.9),),
        ),
    )
    policy = CrossProfileFusionPolicy(mode=CrossProfileFusionMode.RRF_ONLY)
    first = run_governed_cross_profile_fusion(ranked, policy=policy)
    second = run_governed_cross_profile_fusion(ranked, policy=policy)
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    assert first.receipt.input_sha256 == second.receipt.input_sha256
    assert first.receipt.policy_sha256 == second.receipt.policy_sha256

    with pytest.raises(ValueError, match="does not match"):
        GovernedCrossProfileFusionReceipt(
            input_sha256=first.receipt.input_sha256,
            policy_sha256=first.receipt.policy_sha256,
            result_sha256=first.receipt.result_sha256,
            mode=first.receipt.mode,
            calibration_contract_sha256=first.receipt.calibration_contract_sha256,
            profile_artifact_sha256s=first.receipt.profile_artifact_sha256s,
            receipt_sha256=sha("tampered"),
        )


def test_profile_list_rejects_retriever_profile_identity_mismatch() -> None:
    dense = profile("dense", "dense")
    with pytest.raises(ValueError, match="retriever_id"):
        ProfileRankedList(
            "bad-list",
            dense,
            (candidate(cid="x", retriever="other", document="doc", chunk="c", rank=1, score=1.0),),
        )
