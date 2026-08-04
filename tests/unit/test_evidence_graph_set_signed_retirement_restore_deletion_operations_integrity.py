from __future__ import annotations

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_operations as raw,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_operations_boundary as boundary,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_audit import (
    _CLASSIFICATIONS as PERMIT_CLASSIFICATIONS,
)


def test_operational_report_reconstruction_revalidates_counts_and_digest():
    counts = {name: 0 for name in sorted(raw._CLASSIFICATIONS)}
    stable = {
        "scope": "rigorousrag-restore-deletion-operational-audit-v1",
        "owner_id": "alice",
        "generated_at": 1.0,
        "item_count": 0,
        "classification_counts": counts,
        "items": [],
    }
    report = boundary.RestoreDeletionOperationalReport(
        owner_id="alice",
        generated_at=1.0,
        item_count=0,
        classification_counts=counts,
        items=(),
        report_digest=raw._canonical_digest(stable),
    )
    assert report.item_count == 0
    with pytest.raises(ValueError, match="report_digest"):
        boundary.RestoreDeletionOperationalReport(
            **{**report.__dict__, "report_digest": "f" * 64}
        )
    with pytest.raises(ValueError, match="counts"):
        boundary.RestoreDeletionOperationalReport(
            **{
                **report.__dict__,
                "classification_counts": {
                    **counts,
                    "completed": 1,
                },
            }
        )


def test_retention_plan_reconstruction_revalidates_candidates_and_digest():
    stable = {
        "scope": "rigorousrag-restore-deletion-retention-plan-v1",
        "owner_id": "alice",
        "generated_at": 1.0,
        "minimum_age_seconds": 10.0,
        "retain_latest_per_restore": 1,
        "include_completed": False,
        "candidate_count": 0,
        "items": [],
    }
    plan = boundary.RestoreDeletionRetentionPlan(
        owner_id="alice",
        generated_at=1.0,
        minimum_age_seconds=10.0,
        retain_latest_per_restore=1,
        include_completed=False,
        candidate_count=0,
        items=(),
        plan_digest=raw._canonical_digest(stable),
    )
    assert plan.candidate_count == 0
    with pytest.raises(ValueError, match="plan_digest"):
        boundary.RestoreDeletionRetentionPlan(
            **{**plan.__dict__, "plan_digest": "f" * 64}
        )
    with pytest.raises(ValueError, match="candidate_count"):
        boundary.RestoreDeletionRetentionPlan(
            **{**plan.__dict__, "candidate_count": 1}
        )


def test_permit_report_reconstruction_revalidates_safety_and_digest():
    counts = {name: 0 for name in sorted(PERMIT_CLASSIFICATIONS)}
    stable = {
        "scope": "rigorousrag-restore-hold-permit-audit-v1",
        "owner_id": "alice",
        "generated_at": 1.0,
        "item_count": 0,
        "classification_counts": counts,
        "items": [],
    }
    report = boundary.RestoreHoldPermitAuditReport(
        owner_id="alice",
        generated_at=1.0,
        item_count=0,
        classification_counts=counts,
        items=(),
        report_digest=raw._canonical_digest(stable),
    )
    assert report.permit_released is False
    with pytest.raises(ValueError, match="must be false"):
        boundary.RestoreHoldPermitAuditReport(
            **{**report.__dict__, "permit_released": True}
        )
    with pytest.raises(ValueError, match="report_digest"):
        boundary.RestoreHoldPermitAuditReport(
            **{**report.__dict__, "report_digest": "f" * 64}
        )
