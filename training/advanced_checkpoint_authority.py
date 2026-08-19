"""Advanced-RAG checkpoint-root authority layered over the generic checkpoint manager."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from training.checkpointing import CheckpointManager


def assert_safe_advanced_checkpoint_root(path: str | Path) -> Path:
    """Reject symlinked roots/ancestors before the generic manager canonicalizes the path."""
    raw = Path(path).expanduser()
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"advanced checkpoint path traverses symlink: {candidate}")
    resolved = absolute.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("advanced checkpoint root must be a directory when it exists")
    return resolved


class AdvancedCheckpointManager(CheckpointManager):
    """Generic manager with a stricter pre-resolution path authority check."""
    def __init__(self, root: str | Path) -> None:
        safe = assert_safe_advanced_checkpoint_root(root)
        super().__init__(safe)


__all__ = ["AdvancedCheckpointManager", "assert_safe_advanced_checkpoint_root"]
