"""Process-local factory for signed actor-use graph-set publication attempts."""

from __future__ import annotations

import os
from pathlib import Path

from tools.evidence_graph_set_publish_attempts import EvidenceGraphSetPublicationJournal
from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_publications.sqlite3"


def get_evidence_graph_set_signed_publication_journal(
    path: str | os.PathLike[str] | None = None,
) -> EvidenceGraphSetPublicationJournal:
    """Return the journal reserved for signed-provenance publication.

    The signed command family must never share durable attempt state with the
    authorization-only publication family. Reusing the common journal could let a
    candidate created without signed actor-use metadata be resumed by the stronger
    command path after the candidate-build phase had already completed.
    """

    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
        _DEFAULT_PATH,
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return get_evidence_graph_set_publication_journal(candidate)


__all__ = ["get_evidence_graph_set_signed_publication_journal"]
