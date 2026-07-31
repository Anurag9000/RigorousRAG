"""Owner-scoped persistent fielded sparse retrieval with exact snapshots."""

from tools.sparse_backend import SparseBackend
from tools.sparse_search import SparseSearchMixin
from tools.sparse_snapshot import SparseSnapshotMixin
from tools.sparse_types import (
    SparseDocumentSnapshot,
    SparseField,
    SparseFieldSnapshot,
    SparseMatch,
    SparseSearchHit,
)
from tools.sparse_utils import DEFAULT_FIELD_WEIGHTS, _is_redirecting, tokenize
from tools.sparse_write import SparseWriteMixin


class SparseIndex(
    SparseWriteMixin,
    SparseSnapshotMixin,
    SparseSearchMixin,
    SparseBackend,
):
    """Single-host transactional sparse index with owner/document isolation."""


__all__ = [
    "DEFAULT_FIELD_WEIGHTS",
    "SparseDocumentSnapshot",
    "SparseField",
    "SparseFieldSnapshot",
    "SparseIndex",
    "SparseMatch",
    "SparseSearchHit",
    "tokenize",
]
