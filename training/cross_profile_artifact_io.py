"""Canonical, atomic persistence for cross-profile calibration/training artifacts.

The cross-profile stack is intentionally dependency-free, so its small model/control
artifacts use canonical JSON rather than torch checkpoints.  This module centralizes that
persistence boundary: schema-discriminated envelopes, SHA-256 content identity, bounded
reads, atomic replacement, and symlink/reparse rejection.  Loading reconstructs the
existing validated dataclasses instead of trusting serialized Python objects.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt
from evaluation.fusion_weight_promotion import FusionWeightPromotionReceipt
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionReceipt
from tools.cross_profile_fusion import IsotonicBin, IsotonicCalibrationArtifact, RetrieverScoreProfile, ScoreDirection
from training.cross_profile_fusion_fitting import FusionWeightTrainingState, LearnedFusionWeightArtifact
from training.cross_profile_listwise_fusion import ListwiseFusionTrainingState, LearnedListwiseFusionArtifact

_MAX_BYTES = 64 * 1024 * 1024
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

_KINDS = {
    "isotonic_calibration",
    "calibration_qualification",
    "pointwise_training_state",
    "pointwise_learned_artifact",
    "pointwise_promotion",
    "listwise_training_state",
    "listwise_learned_artifact",
    "listwise_promotion",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _redirecting(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT)


def _safe_parent(value: str | os.PathLike[str]) -> tuple[Path, Path]:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("artifact path must be a filesystem path")
    rendered = os.fspath(value)
    if not rendered or len(rendered) > 4096 or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
        raise ValueError("artifact path is invalid")
    path = Path(rendered)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = Path(os.path.abspath(path))
    parent = path.parent
    if not parent.is_dir():
        raise ValueError("artifact parent directory must already exist")
    for component in (parent, *parent.parents):
        if _redirecting(component):
            raise ValueError("artifact path may not traverse symbolic links or reparse points")
    if path.exists() and _redirecting(path):
        raise ValueError("artifact path may not be a symbolic link or reparse point")
    return path, parent


def _encode(value: object) -> tuple[str, dict[str, Any]]:
    if isinstance(value, IsotonicCalibrationArtifact):
        payload = {
            "profile": {
                "profile_id": value.profile.profile_id,
                "family": value.profile.family,
                "scoring_contract_sha256": value.profile.scoring_contract_sha256,
                "model_profile_sha256": value.profile.model_profile_sha256,
                "score_direction": value.profile.score_direction.value,
            },
            "calibration_contract_sha256": value.calibration_contract_sha256,
            "examples_sha256": value.examples_sha256,
            "example_count": value.example_count,
            "bins": [asdict(item) for item in value.bins],
            "artifact_sha256": value.artifact_sha256,
        }
        return "isotonic_calibration", payload
    if isinstance(value, CalibrationQualificationReceipt):
        return "calibration_qualification", asdict(value)
    if isinstance(value, FusionWeightTrainingState):
        return "pointwise_training_state", asdict(value)
    if isinstance(value, LearnedFusionWeightArtifact):
        return "pointwise_learned_artifact", asdict(value)
    if isinstance(value, FusionWeightPromotionReceipt):
        return "pointwise_promotion", asdict(value)
    if isinstance(value, ListwiseFusionTrainingState):
        return "listwise_training_state", asdict(value)
    if isinstance(value, LearnedListwiseFusionArtifact):
        return "listwise_learned_artifact", asdict(value)
    if isinstance(value, ListwiseFusionPromotionReceipt):
        return "listwise_promotion", asdict(value)
    raise ValueError(f"unsupported cross-profile artifact type: {type(value).__name__}")


def _envelope(value: object) -> dict[str, Any]:
    kind, payload = _encode(value)
    content = {
        "schema": "rigorousrag-cross-profile-artifact-envelope/v1",
        "kind": kind,
        "payload": payload,
    }
    content["content_sha256"] = _sha256_bytes(_canonical_bytes(content))
    return content


def save_cross_profile_artifact(path: str | os.PathLike[str], value: object) -> str:
    """Atomically persist one supported artifact and return the envelope content digest."""

    target, parent = _safe_parent(path)
    envelope = _envelope(value)
    encoded = _canonical_bytes(envelope) + b"\n"
    if len(encoded) > _MAX_BYTES:
        raise ValueError("artifact exceeds the persistence byte limit")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    temporary = Path(temporary_name)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if _redirecting(target):
            raise ValueError("artifact destination became redirecting before commit")
        os.replace(temporary, target)
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return envelope["content_sha256"]


def _tuple_pairs(value: Any, label: str) -> tuple[tuple[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    output = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{label} entries must be pairs")
        output.append((row[0], row[1]))
    return tuple(output)


def _validate_listwise_state(value: ListwiseFusionTrainingState) -> ListwiseFusionTrainingState:
    for label, digest in (
        ("spec_sha256", value.spec_sha256),
        ("train_queries_sha256", value.train_queries_sha256),
        ("validation_queries_sha256", value.validation_queries_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    for label, number in (("epoch", value.epoch), ("batch_index", value.batch_index), ("stale_epochs", value.stale_epochs)):
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    theta = tuple(value.theta)
    best = tuple(value.best_theta)
    if not theta or len(theta) != len(best):
        raise ValueError("listwise theta vectors must be non-empty and equally sized")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not __import__("math").isfinite(float(item)) for item in theta + best):
        raise ValueError("listwise theta vectors must be finite")
    if value.best_validation_loss is not None:
        loss = float(value.best_validation_loss)
        if not __import__("math").isfinite(loss) or loss < 0.0:
            raise ValueError("best_validation_loss must be finite and non-negative")
    if value.best_epoch is not None and (isinstance(value.best_epoch, bool) or not isinstance(value.best_epoch, int) or value.best_epoch < 0):
        raise ValueError("best_epoch must be non-negative when set")
    if not isinstance(value.completed, bool):
        raise ValueError("completed must be boolean")
    return value


def _decode(kind: str, payload: Any) -> object:
    if kind not in _KINDS or not isinstance(payload, dict):
        raise ValueError("unsupported or malformed cross-profile artifact envelope")
    if kind == "isotonic_calibration":
        profile_raw = payload["profile"]
        profile = RetrieverScoreProfile(
            profile_id=profile_raw["profile_id"],
            family=profile_raw["family"],
            scoring_contract_sha256=profile_raw["scoring_contract_sha256"],
            model_profile_sha256=profile_raw["model_profile_sha256"],
            score_direction=ScoreDirection(profile_raw["score_direction"]),
        )
        return IsotonicCalibrationArtifact(
            profile=profile,
            calibration_contract_sha256=payload["calibration_contract_sha256"],
            examples_sha256=payload["examples_sha256"],
            example_count=payload["example_count"],
            bins=tuple(IsotonicBin(**row) for row in payload["bins"]),
            artifact_sha256=payload["artifact_sha256"],
        )
    if kind == "calibration_qualification":
        data = dict(payload)
        data["reason_codes"] = tuple(data["reason_codes"])
        return CalibrationQualificationReceipt(**data)
    if kind == "pointwise_training_state":
        data = dict(payload)
        data["theta"] = tuple(data["theta"])
        data["best_theta"] = tuple(data["best_theta"])
        return FusionWeightTrainingState(**data)
    if kind == "pointwise_learned_artifact":
        data = dict(payload)
        data["profile_weights"] = _tuple_pairs(data["profile_weights"], "profile_weights")
        data["calibration_artifact_sha256s"] = _tuple_pairs(data["calibration_artifact_sha256s"], "calibration_artifact_sha256s")
        return LearnedFusionWeightArtifact(**data)
    if kind == "pointwise_promotion":
        data = dict(payload)
        data["reason_codes"] = tuple(data["reason_codes"])
        return FusionWeightPromotionReceipt(**data)
    if kind == "listwise_training_state":
        data = dict(payload)
        data["theta"] = tuple(data["theta"])
        data["best_theta"] = tuple(data["best_theta"])
        return _validate_listwise_state(ListwiseFusionTrainingState(**data))
    if kind == "listwise_learned_artifact":
        data = dict(payload)
        data["profile_weights"] = _tuple_pairs(data["profile_weights"], "profile_weights")
        data["calibration_artifact_sha256s"] = _tuple_pairs(data["calibration_artifact_sha256s"], "calibration_artifact_sha256s")
        return LearnedListwiseFusionArtifact(**data)
    data = dict(payload)
    data["reason_codes"] = tuple(data["reason_codes"])
    return ListwiseFusionPromotionReceipt(**data)


def load_cross_profile_artifact(path: str | os.PathLike[str]) -> object:
    """Load one canonical artifact, verify its envelope digest, and reconstruct its type."""

    target, _ = _safe_parent(path)
    if not target.is_file() or _redirecting(target):
        raise ValueError("artifact must be a non-redirecting regular file")
    metadata = target.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 2 or metadata.st_size > _MAX_BYTES:
        raise ValueError("artifact file type or size is invalid")
    raw = target.read_bytes()
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError("artifact is not valid UTF-8 JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != "rigorousrag-cross-profile-artifact-envelope/v1":
        raise ValueError("artifact envelope schema is invalid")
    if raw != _canonical_bytes(envelope) + b"\n":
        raise ValueError("artifact encoding is not canonical")
    provided = envelope.get("content_sha256")
    if not isinstance(provided, str) or len(provided) != 64:
        raise ValueError("artifact envelope content digest is invalid")
    unsigned = dict(envelope)
    unsigned.pop("content_sha256", None)
    expected = _sha256_bytes(_canonical_bytes(unsigned))
    if provided != expected:
        raise ValueError("artifact envelope digest verification failed")
    return _decode(envelope.get("kind"), envelope.get("payload"))


__all__ = ["load_cross_profile_artifact", "save_cross_profile_artifact"]
