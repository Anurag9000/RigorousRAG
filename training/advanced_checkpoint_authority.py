"""Advanced-RAG checkpoint-root authority layered over the generic checkpoint manager."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from training.checkpointing import CheckpointArtifact, CheckpointManager, TensorCheckpointManifest

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MANIFEST_FIELDS = {
    "version",
    "run_id",
    "source_commit",
    "training_config_digest",
    "dataset_manifest_digest",
    "model_architecture",
    "trainer_state_digest",
    "artifacts",
    "parent_checkpoint_digest",
    "stage_boundary",
    "metric_snapshot",
}


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


def _strict_manifest_payload(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("advanced checkpoint manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise ValueError("advanced checkpoint manifest exceeds byte safety bound")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except Exception as exc:
        raise ValueError("advanced checkpoint manifest is not strict JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("advanced checkpoint manifest fields differ from the closed v1 schema")
    if not isinstance(payload.get("artifacts"), list):
        raise ValueError("advanced checkpoint manifest artifacts must be an array")
    return payload


class AdvancedCheckpointManager(CheckpointManager):
    """Generic manager with stricter path, manifest and closed-directory authority."""

    def __init__(self, root: str | Path) -> None:
        safe = assert_safe_advanced_checkpoint_root(root)
        super().__init__(safe)

    def read_manifest(self, checkpoint_path: str | Path) -> TensorCheckpointManifest:
        path = Path(checkpoint_path).resolve(strict=True)
        if path.parent != self.root or not path.is_dir() or path.is_symlink():
            raise ValueError("checkpoint path must be an immediate non-symlink child of configured root")
        payload = _strict_manifest_payload(path / "manifest.json")
        artifacts = tuple(CheckpointArtifact(**value) for value in payload["artifacts"])
        return TensorCheckpointManifest(
            version=payload["version"],
            run_id=payload["run_id"],
            source_commit=payload["source_commit"],
            training_config_digest=payload["training_config_digest"],
            dataset_manifest_digest=payload["dataset_manifest_digest"],
            model_architecture=payload["model_architecture"],
            trainer_state_digest=payload["trainer_state_digest"],
            artifacts=artifacts,
            parent_checkpoint_digest=payload["parent_checkpoint_digest"],
            stage_boundary=payload["stage_boundary"],
            metric_snapshot=payload["metric_snapshot"],
        )

    def verify(self, checkpoint_digest: str) -> tuple[Path, TensorCheckpointManifest]:
        path, manifest = super().verify(checkpoint_digest)
        expected = {"manifest.json", *(artifact.filename for artifact in manifest.artifacts)}
        children = {item.name: item for item in path.iterdir()}
        if set(children) != expected:
            raise ValueError(
                "advanced checkpoint directory contains files outside its closed manifest; "
                f"unexpected={sorted(set(children) - expected)}, missing={sorted(expected - set(children))}"
            )
        for name, child in children.items():
            if child.is_symlink() or not child.is_file():
                raise ValueError(f"advanced checkpoint child {name!r} must be a regular non-symlink file")
        return path, manifest


__all__ = ["AdvancedCheckpointManager", "assert_safe_advanced_checkpoint_root"]
