"""Truthful retained-source capability boundary.

The complete SQLite registry implementation remains in ``document_store_legacy``.
This module narrows only the public verification flags: a source is "verified" only
when the requested byte-identity and PDF-safety checks actually pass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tools import document_store_legacy as _implementation

_original_document_store = _implementation.DocumentStore


class DocumentStore(_original_document_store):
    """Registry whose verification flags distinguish attempted from successful checks."""

    def get(
        self,
        *,
        owner_id: str,
        doc_id: str,
        verify_visual: bool = False,
    ) -> Optional[Dict[str, Any]]:
        record = super().get(
            owner_id=owner_id,
            doc_id=doc_id,
            verify_visual=verify_visual,
        )
        if record is None:
            return None
        raw_path = str(record.get("source_path") or "")
        check_performed = bool(
            verify_visual
            and raw_path
            and Path(raw_path).suffix.lower() == ".pdf"
        )
        record["visual_source_check_performed"] = check_performed
        record["visual_source_verified"] = bool(
            check_performed and record.get("visual_source_available")
        )
        return record


_implementation.DocumentStore = DocumentStore
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
