"""Canonical phase-guarded custody artifact publication journal."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal import (
    RestoreCustodyArtifactJournal,
)


class GovernedRestoreCustodyArtifactJournal(RestoreCustodyArtifactJournal):
    """Require durable publication intent before any artifact terminal state."""

    def _require_publication_intent(
        self,
        artifact_id: str,
        *,
        worker_id: str,
    ) -> None:
        current = self.get(artifact_id)
        if (
            current.state != "running"
            or current.lease_owner != worker_id
            or current.phase != "publication_intent"
        ):
            raise RuntimeError(
                "artifact terminal transition requires publication intent."
            )

    def complete(self, artifact_id: str, *, worker_id: str, **kwargs: Any):
        self._require_publication_intent(
            artifact_id,
            worker_id=worker_id,
        )
        return super().complete(
            artifact_id,
            worker_id=worker_id,
            **kwargs,
        )

    def orphan(self, artifact_id: str, *, worker_id: str, **kwargs: Any):
        self._require_publication_intent(
            artifact_id,
            worker_id=worker_id,
        )
        return super().orphan(
            artifact_id,
            worker_id=worker_id,
            **kwargs,
        )


__all__ = ["GovernedRestoreCustodyArtifactJournal"]
