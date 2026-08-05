from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.evidence_graph_claim_contracts import ClaimEvidenceLocator, ScientificClaimProposal
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)
from tools.evidence_graph_claim_extractor_benchmark import (
    aggregate_scientific_claim_extractor_benchmark,
    build_scientific_claim_extractor_benchmark_case,
)
from tools.evidence_graph_claim_extractor_promotion import (
    ClaimExtractorPromotionAdministratorGrant,
    ClaimExtractorPromotionPolicy,
    ClaimExtractorPromotionThresholds,
    GovernedScientificClaimExtractorPromotionService,
    assess_scientific_claim_extractor_promotion,
)
from tools.evidence_graph_claim_extractor_promotion_transactional import (
    TransactionalScientificClaimExtractorPromotionStore,
)
from tools.evidence_graph_claim_extractor_registry import (
    ClaimExtractorAdministratorGrant,
    ClaimExtractorGovernancePolicy,
    GovernedScientificClaimExtractorService,
    ScientificClaimExtractorRegistry,
)
from tools.evidence_graph_claim_registered_extraction import (
    extract_governed_scientific_claim_proposals,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding


class Section:
    def __init__(self, content, page_number=1):
        self.content = content
        self.page_number = page_number


class Document:
    def __init__(self):
        self.id = "doc1"
        self.text = "Drug A reduced mortality in the randomized cohort."
        self.sections = [Section(self.text)]
        self.metadata = {
            "content_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        }


def actor():
    return ReviewActorBinding.create(
        actor_id="admin",
        binding_method="process_environment",
        loaded_at=1.0,
    )


def registry_policy():
    return ClaimExtractorGovernancePolicy(
        administrators=(
            ClaimExtractorAdministratorGrant(
                administrator_id="admin",
                owners=("alice",),
                extractor_names=("claims",),
                actions=("register", "retire"),
            ),
        )
    )


def thresholds(**overrides):
    values = {
        "minimum_case_count": 1,
        "minimum_gold_count": 1,
        "minimum_precision": 0.8,
        "minimum_recall": 0.8,
        "minimum_f1": 0.8,
        "minimum_exact_evidence_accuracy": 0.8,
        "minimum_exact_locator_accuracy": 0.8,
        "minimum_mean_span_iou": 0.8,
        "minimum_mean_claim_token_f1": 0.8,
        "minimum_claim_type_accuracy": 0.8,
        "minimum_modality_accuracy": 0.8,
        "maximum_confidence_brier_score": 0.1,
    }
    values.update(overrides)
    return ClaimExtractorPromotionThresholds(**values)


def promotion_policy(**threshold_overrides):
    return ClaimExtractorPromotionPolicy(
        thresholds=thresholds(**threshold_overrides),
        administrators=(
            ClaimExtractorPromotionAdministratorGrant(
                administrator_id="admin",
                owners=("alice",),
                extractor_names=("claims",),
                actions=("promote", "rollback"),
            ),
        ),
    )


def register(registry, *, version, implementation_digit):
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=registry_policy(),
        clock=lambda: 1.0,
    )
    return service.register(
        actor=actor(),
        owner_id="alice",
        extractor_name="claims",
        extractor_version=version,
        extractor_kind="model",
        implementation_sha256=implementation_digit * 64,
        configuration_sha256="b" * 64,
        supported_claim_types=("finding",),
        supported_modalities=("asserted",),
        supported_languages=("en",),
    )


def output():
    text = Document().text
    return {
        "schema_version": 1,
        "claims": [
            {
                "claim_key": "claim-1",
                "claim_text": "Drug A reduced mortality.",
                "claim_type": "finding",
                "modality": "asserted",
                "section_index": 0,
                "page_number": 1,
                "char_start": 0,
                "char_end": text.index(" in the"),
                "confidence": 0.95,
            }
        ],
    }


def suite(registry, record, *, benchmark_id="benchmark-1", dataset_digit="e"):
    batch = extract_governed_scientific_claim_proposals(
        Document(),
        output(),
        owner_id="alice",
        generation=1,
        profile_fingerprint="c" * 64,
        proposer_id="runtime",
        extractor_name="claims",
        extractor_version=record.extractor_version,
        language="en",
        registry=registry,
        now=2.0,
    )
    claim = batch.proposals[0]
    gold = ScientificClaimGold(
        gold_id="gold-1",
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256=claim.content_sha256,
        profile_fingerprint=claim.profile_fingerprint,
        claim_text=claim.claim_text,
        claim_type=claim.claim_type,
        modality=claim.modality,
        locator=claim.locator,
    )
    report = evaluate_scientific_claim_extraction(
        gold=(gold,), proposals=(claim,)
    )
    case = build_scientific_claim_extractor_benchmark_case(
        case_id=f"case-{record.extractor_version}",
        dataset_digest=dataset_digit * 64,
        evaluation_report=report,
        proposals=(claim,),
        extractor_record=record,
    )
    return aggregate_scientific_claim_extractor_benchmark(
        benchmark_id=benchmark_id,
        cases=(case,),
    )


def service(tmp_path):
    registry = ScientificClaimExtractorRegistry(tmp_path / "registry.sqlite3")
    store = TransactionalScientificClaimExtractorPromotionStore(
        tmp_path / "promotions.sqlite3"
    )
    return registry, store, GovernedScientificClaimExtractorPromotionService(
        extractor_registry=registry,
        promotion_store=store,
        policy=promotion_policy(),
        clock=lambda: 3.0,
    )


def test_assessment_records_all_threshold_failures_without_activation(tmp_path):
    registry, store, governed = service(tmp_path)
    record = register(registry, version="1", implementation_digit="a")
    benchmark = suite(registry, record)
    strict = promotion_policy(
        minimum_case_count=2,
        minimum_gold_count=2,
        minimum_precision=1.0,
        maximum_confidence_brier_score=0.001,
    )
    report = assess_scientific_claim_extractor_promotion(
        extractor_record=record,
        benchmark_suite=benchmark,
        policy=strict,
        now=3.0,
    )

    assert report.eligible is False
    assert "case_count_below_floor" in report.reasons
    assert "gold_count_below_floor" in report.reasons
    assert "confidence_brier_score_above_ceiling" in report.reasons

    strict_service = GovernedScientificClaimExtractorPromotionService(
        extractor_registry=registry,
        promotion_store=store,
        policy=strict,
        clock=lambda: 3.0,
    )
    stored, activation = strict_service.promote(
        benchmark_suite=benchmark,
        expected_current_activation_id=None,
        actor=actor(),
    )
    assert stored.eligible is False
    assert activation is None
    assert store.current(owner_id="alice", extractor_name="claims") is None


def test_first_promotion_and_exact_retry_preserve_first_timestamps(tmp_path):
    registry, store, governed = service(tmp_path)
    record = register(registry, version="1", implementation_digit="a")
    benchmark = suite(registry, record)

    report, first = governed.promote(
        benchmark_suite=benchmark,
        expected_current_activation_id=None,
        actor=actor(),
    )
    assert report.eligible is True
    assert first is not None
    assert first.previous_activation_id is None
    assert governed.resolve_current(
        owner_id="alice", extractor_name="claims"
    ) == record

    replay_report = replace(report, assessed_at=99.0)
    assert store.store_report(replay_report).assessed_at == report.assessed_at
    replay = store.activate(
        report=replay_report,
        action="promote",
        expected_current_activation_id=None,
        actor=actor(),
        now=99.0,
    )
    assert replay.activation_id == first.activation_id
    assert replay.activated_at == first.activated_at


def test_optimistic_pointer_refusal_upgrade_and_rollback(tmp_path):
    registry, store, governed = service(tmp_path)
    first_record = register(registry, version="1", implementation_digit="a")
    first_report, first_activation = governed.promote(
        benchmark_suite=suite(registry, first_record, dataset_digit="e"),
        expected_current_activation_id=None,
        actor=actor(),
    )
    assert first_activation is not None

    second_record = register(registry, version="2", implementation_digit="d")
    second_suite = suite(
        registry,
        second_record,
        benchmark_id="benchmark-2",
        dataset_digit="f",
    )
    with pytest.raises(RuntimeError, match="current activation changed"):
        governed.promote(
            benchmark_suite=second_suite,
            expected_current_activation_id=None,
            actor=actor(),
        )

    second_report, second_activation = governed.promote(
        benchmark_suite=second_suite,
        expected_current_activation_id=first_activation.activation_id,
        actor=actor(),
    )
    assert second_report.eligible is True
    assert second_activation is not None
    assert second_activation.previous_activation_id == first_activation.activation_id
    assert governed.resolve_current(
        owner_id="alice", extractor_name="claims"
    ) == second_record

    rollback = governed.rollback(
        target_promotion_report_digest=first_report.report_digest,
        expected_current_activation_id=second_activation.activation_id,
        actor=actor(),
    )
    assert rollback.action == "rollback"
    assert rollback.extractor_version == "1"
    assert rollback.previous_activation_id == second_activation.activation_id
    assert governed.resolve_current(
        owner_id="alice", extractor_name="claims"
    ) == first_record
    assert len(store.history(owner_id="alice", extractor_name="claims")) == 3


def test_rollback_refuses_retired_target_and_tampered_record_scope(tmp_path):
    registry, store, governed = service(tmp_path)
    first_record = register(registry, version="1", implementation_digit="a")
    first_report, first_activation = governed.promote(
        benchmark_suite=suite(registry, first_record),
        expected_current_activation_id=None,
        actor=actor(),
    )
    assert first_activation is not None

    registry.retire(
        owner_id="alice",
        extractor_name="claims",
        extractor_version="1",
        actor=actor(),
        now=4.0,
    )
    with pytest.raises(PermissionError, match="retired"):
        governed.rollback(
            target_promotion_report_digest=first_report.report_digest,
            expected_current_activation_id=first_activation.activation_id,
            actor=actor(),
        )

    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE scientific_claim_extractor_promotion_reports "
            "SET extractor_record_digest=? WHERE report_digest=?",
            ("f" * 64, first_report.report_digest),
        )
    with pytest.raises(RuntimeError, match="columns"):
        store.get_report(first_report.report_digest)
