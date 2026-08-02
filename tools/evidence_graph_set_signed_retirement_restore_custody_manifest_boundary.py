"""Canonical replay-stable boundary for restore custody manifests."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_post_restore_comparison_receipt,
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_manifest import (
    SignedRetirementRestoreCustodyStore,
    _actor_fields,
)


class GovernedSignedRetirementRestoreCustodyStore(
    SignedRetirementRestoreCustodyStore
):
    """Custody store whose exact replays preserve original audit timestamps."""

    def bind_pre(
        self,
        *,
        restore_id: str,
        pre_receipt_path: Any,
        backup_path: Any,
        restore_journal: Any,
        actor: Any,
        now: float | None = None,
    ):
        selected = _digest(restore_id, "restore_id")
        try:
            stored = self.get_for_restore(selected)
        except KeyError:
            return super().bind_pre(
                restore_id=selected,
                pre_receipt_path=pre_receipt_path,
                backup_path=backup_path,
                restore_journal=restore_journal,
                actor=actor,
                now=now,
            )
        receipt = verify_pre_restore_backup_receipt(
            receipt_path=pre_receipt_path,
            backup_path=backup_path,
        )
        restore = restore_journal.get(selected)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        actor_id, method, binding = _actor_fields(actor, now=timestamp)
        if (
            stored.owner_id != restore.owner_id
            or stored.snapshot_digest != restore.snapshot_digest
            or stored.target_path_digest != restore.target_path_digest
            or stored.pre_receipt_digest != receipt.receipt_digest
            or stored.backup_sha256 != receipt.backup_sha256
            or stored.backup_size_bytes != receipt.backup_size_bytes
            or stored.pre_bound_actor_id != actor_id
            or stored.pre_bound_method != method
            or stored.pre_bound_binding_digest != binding
        ):
            raise RuntimeError("restore custody manifest collision detected.")
        return stored

    def bind_post(
        self,
        *,
        restore_id: str,
        post_receipt_path: Any,
        restore_journal: Any,
        actor: Any,
        now: float | None = None,
    ):
        selected = _digest(restore_id, "restore_id")
        stored = self.get_for_restore(selected)
        if stored.state != "post_bound":
            return super().bind_post(
                restore_id=selected,
                post_receipt_path=post_receipt_path,
                restore_journal=restore_journal,
                actor=actor,
                now=now,
            )
        receipt = verify_post_restore_comparison_receipt(post_receipt_path)
        restore = restore_journal.get(selected)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        actor_id, method, binding = _actor_fields(actor, now=timestamp)
        if (
            restore.state != "completed"
            or restore.phase != "verified"
            or receipt.restore_id != selected
            or stored.post_receipt_digest != receipt.receipt_digest
            or stored.target_verification_digest
            != receipt.target_verification_digest
            or stored.post_bound_actor_id != actor_id
            or stored.post_bound_method != method
            or stored.post_bound_binding_digest != binding
        ):
            raise RuntimeError("post custody manifest collision detected.")
        return stored


__all__ = ["GovernedSignedRetirementRestoreCustodyStore"]
