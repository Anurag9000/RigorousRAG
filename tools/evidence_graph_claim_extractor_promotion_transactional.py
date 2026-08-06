"""Canonical replay-safe transactional store for claim extractor promotion."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from tools.evidence_graph_claim_contracts import _identifier, _timestamp
from tools.evidence_graph_claim_extractor_promotion import (
    _PROMOTION_ACTIONS,
    _optional_digest,
    ScientificClaimExtractorActivation,
    ScientificClaimExtractorPromotionReport,
    ScientificClaimExtractorPromotionStore,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("stored promotion payload contains a duplicate JSON key.")
        result[key] = value
    return result


def _payload(value: str, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > 20_000_000:
        raise RuntimeError(f"stored {label} is corrupt.")
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RuntimeError(f"stored {label} is corrupt.") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"stored {label} is corrupt.")
    return raw


def _report_payload(value: str) -> dict[str, Any]:
    """Restore immutable tuple fields lost by canonical JSON serialization."""

    raw = _payload(value, "promotion report")
    reasons = raw.get("reasons")
    if isinstance(reasons, list):
        raw["reasons"] = tuple(reasons)
    return raw


def _report_scope(value: ScientificClaimExtractorPromotionReport) -> tuple[Any, ...]:
    return (
        value.owner_id,
        value.extractor_name,
        value.extractor_version,
        value.extractor_record_digest,
        value.benchmark_id,
        value.benchmark_suite_digest,
        value.policy_digest,
        value.thresholds_digest,
        value.eligible,
        value.reasons,
        value.report_digest,
        value.schema_version,
    )


def _activation_scope(value: ScientificClaimExtractorActivation) -> tuple[Any, ...]:
    return (
        value.activation_id,
        value.owner_id,
        value.extractor_name,
        value.extractor_version,
        value.extractor_record_digest,
        value.promotion_report_digest,
        value.action,
        value.previous_activation_id,
        value.actor_id,
        value.actor_binding_method,
        value.actor_binding_digest,
        value.schema_version,
    )


class TransactionalScientificClaimExtractorPromotionStore(
    ScientificClaimExtractorPromotionStore
):
    """Preserve first timestamps and recover exact pointer-activation replays."""

    @staticmethod
    def _report(row: Any) -> ScientificClaimExtractorPromotionReport:
        try:
            value = ScientificClaimExtractorPromotionReport(
                **_report_payload(row["payload_json"])
            )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("stored promotion report is corrupt.") from exc
        if (
            value.report_digest != row["report_digest"]
            or value.owner_id != row["owner_id"]
            or value.extractor_name != row["extractor_name"]
            or value.extractor_version != row["extractor_version"]
            or value.extractor_record_digest != row["extractor_record_digest"]
            or int(value.eligible) != int(row["eligible"])
            or value.assessed_at != float(row["assessed_at"])
        ):
            raise RuntimeError("stored promotion report columns are corrupt.")
        return value

    @staticmethod
    def _activation(row: Any) -> ScientificClaimExtractorActivation:
        try:
            value = ScientificClaimExtractorActivation(
                **_payload(row["payload_json"], "extractor activation")
            )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("stored activation is corrupt.") from exc
        if (
            value.activation_id != row["activation_id"]
            or value.owner_id != row["owner_id"]
            or value.extractor_name != row["extractor_name"]
            or value.extractor_version != row["extractor_version"]
            or value.extractor_record_digest != row["extractor_record_digest"]
            or value.promotion_report_digest != row["promotion_report_digest"]
            or value.action != row["action"]
            or value.previous_activation_id != row["previous_activation_id"]
            or value.activated_at != float(row["activated_at"])
        ):
            raise RuntimeError("stored activation columns are corrupt.")
        return value

    def store_report(
        self,
        report: ScientificClaimExtractorPromotionReport,
    ) -> ScientificClaimExtractorPromotionReport:
        if not isinstance(report, ScientificClaimExtractorPromotionReport):
            raise ValueError("report must be ScientificClaimExtractorPromotionReport.")
        payload = json.dumps(
            asdict(report),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractor_promotion_reports "
                    "WHERE report_digest=?",
                    (report.report_digest,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO scientific_claim_extractor_promotion_reports "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            report.report_digest,
                            report.owner_id,
                            report.extractor_name,
                            report.extractor_version,
                            report.extractor_record_digest,
                            int(report.eligible),
                            payload,
                            report.assessed_at,
                        ),
                    )
                    stored = report
                else:
                    stored = self._report(row)
                    if _report_scope(stored) != _report_scope(report):
                        raise RuntimeError(
                            "promotion report digest collision detected."
                        )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return stored

    def activate(
        self,
        *,
        report: ScientificClaimExtractorPromotionReport,
        action: str,
        expected_current_activation_id: str | None,
        actor: ReviewActorBinding,
        now: float,
    ) -> ScientificClaimExtractorActivation:
        if (
            not isinstance(report, ScientificClaimExtractorPromotionReport)
            or not report.eligible
        ):
            raise PermissionError(
                "only eligible promotion reports may be activated."
            )
        selected_action = _identifier(action, "action", 30)
        if selected_action not in _PROMOTION_ACTIONS:
            raise ValueError("activation action is unsupported.")
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        expected = _optional_digest(
            expected_current_activation_id,
            "expected_current_activation_id",
        )
        timestamp = _timestamp(now, "now")
        stored_report = self.store_report(report)

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pointer = connection.execute(
                    "SELECT activation_id FROM scientific_claim_extractor_current "
                    "WHERE owner_id=? AND extractor_name=?",
                    (stored_report.owner_id, stored_report.extractor_name),
                ).fetchone()
                current_id = None if pointer is None else pointer["activation_id"]

                if current_id != expected:
                    if current_id is None:
                        raise RuntimeError(
                            "extractor current activation changed."
                        )
                    current_row = connection.execute(
                        "SELECT * FROM scientific_claim_extractor_activations "
                        "WHERE activation_id=?",
                        (current_id,),
                    ).fetchone()
                    if current_row is None:
                        raise RuntimeError(
                            "extractor current activation pointer is corrupt."
                        )
                    current = self._activation(current_row)
                    replay_scope = (
                        current.owner_id == stored_report.owner_id
                        and current.extractor_name == stored_report.extractor_name
                        and current.extractor_version == stored_report.extractor_version
                        and current.extractor_record_digest
                        == stored_report.extractor_record_digest
                        and current.promotion_report_digest
                        == stored_report.report_digest
                        and current.action == selected_action
                        and current.previous_activation_id == expected
                        and current.actor_id == actor.actor_id
                        and current.actor_binding_method == actor.binding_method
                        and current.actor_binding_digest == actor.binding_digest
                    )
                    if replay_scope:
                        connection.execute("COMMIT")
                        return current
                    raise RuntimeError(
                        "extractor current activation changed."
                    )

                candidate = ScientificClaimExtractorActivation.create(
                    owner_id=stored_report.owner_id,
                    extractor_name=stored_report.extractor_name,
                    extractor_version=stored_report.extractor_version,
                    extractor_record_digest=stored_report.extractor_record_digest,
                    promotion_report_digest=stored_report.report_digest,
                    action=selected_action,
                    previous_activation_id=current_id,
                    actor=actor,
                    activated_at=timestamp,
                )
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractor_activations "
                    "WHERE activation_id=?",
                    (candidate.activation_id,),
                ).fetchone()
                if row is None:
                    activation = candidate
                    connection.execute(
                        "INSERT INTO scientific_claim_extractor_activations "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            activation.activation_id,
                            activation.owner_id,
                            activation.extractor_name,
                            activation.extractor_version,
                            activation.extractor_record_digest,
                            activation.promotion_report_digest,
                            activation.action,
                            activation.previous_activation_id,
                            json.dumps(
                                asdict(activation),
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            activation.activated_at,
                        ),
                    )
                else:
                    activation = self._activation(row)
                    if _activation_scope(activation) != _activation_scope(candidate):
                        raise RuntimeError(
                            "activation identity collision detected."
                        )

                connection.execute(
                    "INSERT INTO scientific_claim_extractor_current "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(owner_id, extractor_name) DO UPDATE "
                    "SET activation_id=excluded.activation_id",
                    (
                        activation.owner_id,
                        activation.extractor_name,
                        activation.activation_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return activation


__all__ = ["TransactionalScientificClaimExtractorPromotionStore"]
