"""Non-loading checkpoint inspection and explicit pointer management helpers.

These helpers keep stage selection/best-checkpoint bookkeeping outside the immutable
content-addressed manifest, avoiding impossible self-referential digest fields.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from training.checkpointing import CheckpointManager, TrainerCursor, TrainerState, canonical_digest


def inspect_trainer_state(manager: CheckpointManager, checkpoint_digest: str) -> TrainerState:
    path, manifest = manager.verify(checkpoint_digest)
    payload = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    state = TrainerState(
        run_id=payload["run_id"],
        cursor=TrainerCursor(**payload["cursor"]),
        best_metric=payload.get("best_metric"),
        best_checkpoint_digest=payload.get("best_checkpoint_digest"),
        early_stopping_bad_steps=int(payload.get("early_stopping_bad_steps", 0)),
        stage_name=payload.get("stage_name"),
    )
    from dataclasses import asdict

    if canonical_digest(asdict(state)) != manifest.trainer_state_digest:
        raise RuntimeError("checkpoint trainer state does not match manifest digest")
    return state


def set_checkpoint_pointer(manager: CheckpointManager, pointer: str, checkpoint_digest: str) -> None:
    """Atomically point ``<pointer>.json`` at an already verified checkpoint."""

    selected_pointer = pointer.strip()
    if not selected_pointer or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in selected_pointer):
        raise ValueError("checkpoint pointer name contains unsupported characters")
    _, manifest = manager.verify(checkpoint_digest)
    destination = manager.root / f"{selected_pointer}.json"
    payload = json.dumps(
        {"checkpoint_digest": manifest.digest},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{selected_pointer}-", suffix=".tmp", dir=manager.root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


__all__ = ["inspect_trainer_state", "set_checkpoint_pointer"]
