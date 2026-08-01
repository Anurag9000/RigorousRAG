"""IndexCoordinator variant whose internal deletion bypasses public lifecycle hooks."""

from __future__ import annotations

from tools.index_coordinator import (
    IndexCoordinationError,
    IndexCoordinator,
    _document_lock,
    _identifier,
)
from tools.security import normalize_owner_id
from tools.vector_generation import delete_vector_generation


class RawDeleteIndexCoordinator(IndexCoordinator):
    """Use raw vector deletion so public RAG deletion may coordinate all stores."""

    def delete_document(self, *, owner_id: str, doc_id: str) -> bool:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id", 200)
        lock = _document_lock(owner, document_id)
        with lock:
            prior = self.snapshot(owner_id=owner, doc_id=document_id)
            existed = prior.vector.row_count > 0 or prior.sparse is not None
            if not existed:
                return False
            try:
                delete_vector_generation(
                    self.rag,
                    owner_id=owner,
                    doc_id=document_id,
                )
                self.sparse.delete_document(
                    owner_id=owner,
                    doc_id=document_id,
                )
                return True
            except Exception as exc:
                rollback_errors = self._restore(
                    owner_id=owner,
                    doc_id=document_id,
                    snapshot=prior,
                )
                raise IndexCoordinationError(
                    f"Cross-store deletion failed ({type(exc).__name__}).",
                    rollback_errors=rollback_errors,
                ) from exc


__all__ = ["RawDeleteIndexCoordinator"]
