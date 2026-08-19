"""Compatibility entry point for the authoritative advanced-RAG operator.

Historically this module carried an independent copy of validation/training/export logic. That
created two executable authority paths that could drift in checkpoint, cache and dataset
validation. The implementation now lives exclusively in :mod:`training.advanced_rag_operator`;
this module preserves the existing import and ``python -m training.advanced_rag_cli`` surface
without retaining a weaker duplicate workflow.
"""
from __future__ import annotations

from training.advanced_rag_operator import (
    build_evaluation_receipt_from_config,
    export_from_config,
    load_artifact_from_config,
    main,
    qualify_artifact,
    train_from_config,
    validate_config,
    verify_artifact,
    verify_checkpoint_from_config,
    verify_promotion_evidence,
)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_evaluation_receipt_from_config",
    "export_from_config",
    "load_artifact_from_config",
    "main",
    "qualify_artifact",
    "train_from_config",
    "validate_config",
    "verify_artifact",
    "verify_checkpoint_from_config",
    "verify_promotion_evidence",
]
