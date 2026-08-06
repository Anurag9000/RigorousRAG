from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_capability_ledger import load_and_validate, validate_ledger

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "docs" / "rigorousrag_capability_ledger.json"


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_repository_capability_ledger_is_valid() -> None:
    assert load_and_validate(LEDGER_PATH, REPO_ROOT) == []


def test_ledger_has_broad_auditable_coverage() -> None:
    data = _ledger()
    capabilities = data["capabilities"]
    assert isinstance(capabilities, list)
    assert len(capabilities) >= 30

    identifiers = {item["id"] for item in capabilities}
    assert len(identifiers) == len(capabilities)
    assert {
        "correctness",
        "operations_security",
        "ingestion_lifecycle",
        "retrieval_ranking",
        "adaptive_agentic",
        "answer_verification",
        "graph_scientific",
        "evaluation",
        "domain_multimodal",
    } <= {item["category"] for item in capabilities}

    # The audit must distinguish code existence from validation and release proof.
    assert any(item["implementation"] == "implemented" for item in capabilities)
    assert any(item["implementation"] == "partial" for item in capabilities)
    assert any(item["implementation"] == "not_started" for item in capabilities)
    assert any(item["release"] == "blocked" for item in capabilities)
    assert not any(item["release"] == "release_verified" for item in capabilities)


def test_validator_rejects_unsafe_evidence_path() -> None:
    data = copy.deepcopy(_ledger())
    data["capabilities"][0]["evidence"] = ["../outside-repository"]
    errors = validate_ledger(data, REPO_ROOT)
    assert any("unsafe path" in error for error in errors)


def test_validator_rejects_unknown_dependency() -> None:
    data = copy.deepcopy(_ledger())
    data["capabilities"][0]["dependencies"] = ["UNKNOWN-999"]
    errors = validate_ledger(data, REPO_ROOT)
    assert any("unknown dependency UNKNOWN-999" in error for error in errors)


def test_validator_rejects_inconsistent_release_claim() -> None:
    data = copy.deepcopy(_ledger())
    capability = data["capabilities"][0]
    capability["implementation"] = "partial"
    capability["validation"] = "unit_validated"
    capability["release"] = "release_verified"
    errors = validate_ledger(data, REPO_ROOT)
    assert any("release_verified requires" in error for error in errors)


def test_validator_rejects_dependency_cycle() -> None:
    data = copy.deepcopy(_ledger())
    first = data["capabilities"][0]
    second = data["capabilities"][1]
    first["dependencies"] = [second["id"]]
    second["dependencies"] = [first["id"]]
    errors = validate_ledger(data, REPO_ROOT)
    assert any("dependency cycle" in error for error in errors)
