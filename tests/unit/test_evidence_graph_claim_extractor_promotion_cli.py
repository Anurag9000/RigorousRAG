from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from tools import evidence_graph_claim_extractor_promotion_cli as cli
from tools.evidence_graph_claim_contracts import ClaimEvidenceLocator, ScientificClaimProposal
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)
from tools.evidence_graph_claim_extractor_benchmark import (
    aggregate_scientific_claim_extractor_benchmark,
    build_scientific_claim_extractor_benchmark_case,
)
from tools.evidence_graph_claim_extractor_promotion_runtime import (
    clear_scientific_claim_extractor_promotion_runtime_cache,
)
from tools.evidence_graph_claim_extractor_registry import (
    ClaimExtractorAdministratorGrant,
    ClaimExtractorGovernancePolicy,
    GovernedScientificClaimExtractorService,
)
from tools.evidence_graph_claim_extractor_runtime import (
    clear_scientific_claim_extractor_runtime_cache,
    get_scientific_claim_extractor_registry,
)
from tools.evidence_graph_claim_registered_extraction import (
    extract_governed_scientific_claim_proposals,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding


class Section:
    def __init__(self, content):
        self.content = content
        self.page_number = 1


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


def configure(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_REGISTRY_DB_PATH",
        str(tmp_path / "registry.sqlite3"),
    )
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_DB_PATH",
        str(tmp_path / "promotions.sqlite3"),
    )
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "admin")
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", raising=False)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH", raising=False)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_JSON",
        json.dumps(
            {
                "schema_version": 1,
                "thresholds": {
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
                },
                "administrators": [
                    {
                        "administrator_id": "admin",
                        "owners": ["alice"],
                        "extractor_names": ["claims"],
                        "actions": ["promote", "rollback"],
                    }
                ],
            }
        ),
    )
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH",
        raising=False,
    )
    clear_scientific_claim_extractor_runtime_cache()
    clear_scientific_claim_extractor_promotion_runtime_cache()


def register(version, implementation_digit="a"):
    registry = get_scientific_claim_extractor_registry()
    service = GovernedScientificClaimExtractorService(
        registry=registry,
        policy=ClaimExtractorGovernancePolicy(
            administrators=(
                ClaimExtractorAdministratorGrant(
                    administrator_id="admin",
                    owners=("alice",),
                    extractor_names=("claims",),
                    actions=("register", "retire"),
                ),
            )
        ),
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


def suite_file(tmp_path, record, *, benchmark_id, dataset_digit):
    registry = get_scientific_claim_extractor_registry()
    claim = extract_governed_scientific_claim_proposals(
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
    ).proposals[0]
    gold = ScientificClaimGold(
        gold_id=f"gold-{record.extractor_version}",
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
    suite = aggregate_scientific_claim_extractor_benchmark(
        benchmark_id=benchmark_id,
        cases=(case,),
    )
    path = tmp_path / f"suite-{record.extractor_version}.json"
    path.write_text(json.dumps(asdict(suite)), encoding="utf-8")
    return path, suite


def read(capsys):
    captured = capsys.readouterr()
    return (
        None if not captured.out else json.loads(captured.out),
        None if not captured.err else json.loads(captured.err),
        captured.out + captured.err,
    )


def test_assess_promote_current_resolve_and_history_are_text_free(
    tmp_path, monkeypatch, capsys
):
    configure(tmp_path, monkeypatch)
    record = register("1")
    path, suite = suite_file(
        tmp_path, record, benchmark_id="benchmark-1", dataset_digit="e"
    )

    assert cli.main(["assess", str(path)]) == 0
    assessed, error, rendered = read(capsys)
    assert error is None
    assert assessed["eligible"] is True
    assert assessed["mutation_performed"] is False
    assert assessed["activation_performed"] is False
    assert "Drug A" not in rendered

    assert cli.main(["promote", str(path)]) == 0
    promoted, error, rendered = read(capsys)
    assert error is None
    assert promoted["activation_performed"] is True
    assert promoted["report"]["benchmark_suite_digest"] == suite.suite_digest
    first_id = promoted["activation"]["activation_id"]
    assert "Drug A" not in rendered

    assert cli.main([
        "current", "--owner-id", "alice", "--extractor-name", "claims"
    ]) == 0
    current, error, _rendered = read(capsys)
    assert error is None
    assert current["current"]["activation_id"] == first_id
    assert current["mutation_performed"] is False

    assert cli.main([
        "resolve", "--owner-id", "alice", "--extractor-name", "claims"
    ]) == 0
    resolved, error, _rendered = read(capsys)
    assert error is None
    assert resolved["extractor_version"] == "1"
    assert resolved["extractor_record_digest"] == record.record_digest

    assert cli.main([
        "history", "--owner-id", "alice", "--extractor-name", "claims"
    ]) == 0
    history, error, _rendered = read(capsys)
    assert error is None
    assert history["item_count"] == 1
    assert history["mutation_performed"] is False


def test_upgrade_and_exact_rollback_are_append_only(tmp_path, monkeypatch, capsys):
    configure(tmp_path, monkeypatch)
    first = register("1")
    first_path, _suite = suite_file(
        tmp_path, first, benchmark_id="benchmark-1", dataset_digit="e"
    )
    assert cli.main(["promote", str(first_path)]) == 0
    first_result, _error, _rendered = read(capsys)
    first_activation = first_result["activation"]["activation_id"]
    first_report = first_result["report"]["report_digest"]

    second = register("2", "d")
    second_path, _suite = suite_file(
        tmp_path, second, benchmark_id="benchmark-2", dataset_digit="f"
    )
    assert cli.main([
        "promote",
        str(second_path),
        "--expected-current-activation-id",
        first_activation,
    ]) == 0
    second_result, _error, _rendered = read(capsys)
    second_activation = second_result["activation"]["activation_id"]

    assert cli.main([
        "rollback",
        "--target-promotion-report-digest",
        first_report,
        "--expected-current-activation-id",
        "f" * 64,
    ]) == 2
    output, error, _rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}

    assert cli.main([
        "rollback",
        "--target-promotion-report-digest",
        first_report,
        "--expected-current-activation-id",
        second_activation,
    ]) == 0
    rollback, error, _rendered = read(capsys)
    assert error is None
    assert rollback["rollback_performed"] is True
    assert rollback["activation"]["action"] == "rollback"
    assert rollback["activation"]["extractor_version"] == "1"

    assert cli.main([
        "history", "--owner-id", "alice", "--extractor-name", "claims"
    ]) == 0
    history, error, _rendered = read(capsys)
    assert error is None
    assert history["item_count"] == 3


def test_suite_tampering_and_missing_policy_fail_generically(
    tmp_path, monkeypatch, capsys
):
    configure(tmp_path, monkeypatch)
    record = register("1")
    path, _suite = suite_file(
        tmp_path, record, benchmark_id="benchmark-1", dataset_digit="e"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["precision"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert cli.main(["assess", str(path)]) == 2
    output, error, rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert "precision" not in rendered.casefold()

    path, _suite = suite_file(
        tmp_path, record, benchmark_id="benchmark-1", dataset_digit="e"
    )
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_JSON",
        raising=False,
    )
    assert cli.main(["promote", str(path)]) == 2
    output, error, _rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
