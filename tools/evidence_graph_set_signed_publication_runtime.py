"""Process-local factory for signed actor-use graph-set publication attempts."""

from __future__ import annotations

import os
from pathlib import Path

from tools.evidence_graph_set_publish_attempts import EvidenceGraphSetPublicationJournal
from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_publications.sqlite3"
_COMMON_DEFAULT_PATH = "data/evidence_graph_set_publications.sqlite3"


def _absolute(value: str | os.PathLike[str]) -> Path:
    candidate = Path(os.fspath(value))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        left_info = left.stat()
        right_info = right.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError("publication journal identity could not be validated.") from exc
    return (int(left_info.st_dev), int(left_info.st_ino)) == (
        int(right_info.st_dev),
        int(right_info.st_ino),
    )


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
    common_selected = os.getenv(
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        _COMMON_DEFAULT_PATH,
    )
    candidate = _absolute(selected)
    common = _absolute(common_selected)
    if os.path.normcase(str(candidate)) == os.path.normcase(str(common)):
        raise RuntimeError(
            "signed and authorization-only publication journals must use distinct paths."
        )
    if _same_existing_file(candidate, common):
        raise RuntimeError(
            "signed and authorization-only publication journals may not alias one file."
        )
    return get_evidence_graph_set_publication_journal(candidate)


__all__ = ["get_evidence_graph_set_signed_publication_journal"]
