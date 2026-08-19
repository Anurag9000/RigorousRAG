"""Non-loading checkpoint inspection and explicit pointer management helpers.

These helpers keep stage selection/best-checkpoint bookkeeping outside the immutable
content-addressed manifest, avoiding impossible self-referential digest fields. Pointer reads
are convenience resolution only: every resolved digest is re-verified through the configured
checkpoint manager before it is returned.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from training.checkpointing import CheckpointManager, TrainerCursor, TrainerState, canonical_digest

_MAX_POINTER_BYTES = 4096
_HEX = frozenset("0123456789abcdef")


def _pointer_name(pointer: str) -> str:
    if not isinstance(pointer, str):
        raise ValueError("checkpoint pointer name must be a string")
    selected = pointer.strip()
    if not selected or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in selected):
        raise ValueError("checkpoint pointer name contains unsupported characters")
    return selected


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
    if canonical_digest(asdict(state)) != manifest.trainer_state_digest:
        raise RuntimeError("checkpoint trainer state does not match manifest digest")
    return state


def set_checkpoint_pointer(manager: CheckpointManager, pointer: str, checkpoint_digest: str) -> None:
    """Atomically point ``<pointer>.json`` at an already verified checkpoint."""
    selected_pointer = _pointer_name(pointer)
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


def read_checkpoint_pointer(manager: CheckpointManager, pointer: str) -> str:
    """Resolve a persisted pointer and return only a presently verified checkpoint digest."""
    selected_pointer = _pointer_name(pointer)
    path = manager.root / f"{selected_pointer}.json"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint pointer {selected_pointer!r} does not exist")
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint pointer must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_POINTER_BYTES:
        raise ValueError("checkpoint pointer exceeds byte safety bound")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except Exception as exc:
        raise ValueError("checkpoint pointer is not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"checkpoint_digest"}:
        raise ValueError("checkpoint pointer has an unsupported schema")
    digest = str(payload["checkpoint_digest"]).strip().lower()
    if len(digest) != 64 or any(ch not in _HEX for ch in digest):
        raise ValueError("checkpoint pointer digest must be SHA-256")
    _, manifest = manager.verify(digest)
    if manifest.digest != digest:
        raise RuntimeError("checkpoint pointer verification returned a different digest")
    return digest


__all__ = ["inspect_trainer_state", "read_checkpoint_pointer", "set_checkpoint_pointer"]
