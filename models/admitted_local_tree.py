"""Govern a deterministic local artifact directory with an admission proof.

Unlike Hugging Face model bindings, some runtime assets are a single logical directory
without a paired tokenizer tree (for example Tesseract ``tessdata``).  This module reuses
the existing deterministic tree digest and artifact-attestation proof while keeping the
artifact type explicit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.admitted_local_artifacts import AdmittedArtifactProof
from models.local_hf_adapters import artifact_tree_digest

_HEX = frozenset("0123456789abcdef")
_ALLOWED_TREE_TYPES = frozenset({"model", "tokenizer", "index", "dataset", "configuration"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class AdmittedLocalArtifactTree:
    root: str
    tree_sha256: str
    proof: AdmittedArtifactProof
    binding_sha256: str

    def __post_init__(self) -> None:
        selected = Path(self.root).expanduser().resolve(strict=True)
        if selected.is_symlink() or not selected.is_dir():
            raise ValueError("root must resolve to a non-symlink directory")
        object.__setattr__(self, "root", str(selected))
        object.__setattr__(self, "tree_sha256", _sha(self.tree_sha256, "tree_sha256"))
        if not isinstance(self.proof, AdmittedArtifactProof):
            raise ValueError("proof must be AdmittedArtifactProof")
        artifact_type = self.proof.statement.subject.artifact_type
        if artifact_type not in _ALLOWED_TREE_TYPES:
            raise ValueError("admitted local tree requires a directory-compatible artifact type")
        if self.proof.artifact_sha256 != self.tree_sha256:
            raise ValueError("artifact proof digest differs from local tree digest")
        expected = _digest(self._payload())
        provided = _sha(self.binding_sha256, "binding_sha256")
        if expected != provided:
            raise ValueError("binding_sha256 does not match admitted local tree")
        object.__setattr__(self, "binding_sha256", provided)

    @property
    def artifact_type(self) -> str:
        return self.proof.statement.subject.artifact_type

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-admitted-local-artifact-tree/v1",
            "tree_sha256": self.tree_sha256,
            "artifact_type": self.proof.statement.subject.artifact_type,
            "proof_sha256": self.proof.proof_sha256,
        }

    @classmethod
    def build(cls, root: str | Path, *, proof: AdmittedArtifactProof) -> "AdmittedLocalArtifactTree":
        if not isinstance(proof, AdmittedArtifactProof):
            raise ValueError("proof must be AdmittedArtifactProof")
        selected = Path(root).expanduser().resolve(strict=True)
        tree_sha = artifact_tree_digest(selected)
        payload = {
            "schema": "rigorousrag-admitted-local-artifact-tree/v1",
            "tree_sha256": tree_sha,
            "artifact_type": proof.statement.subject.artifact_type,
            "proof_sha256": proof.proof_sha256,
        }
        return cls(str(selected), tree_sha, proof, _digest(payload))

    def verify(self, *, required_artifact_type: str | None = None) -> str:
        if required_artifact_type is not None and self.artifact_type != required_artifact_type:
            raise RuntimeError("admitted local tree has the wrong artifact type")
        selected = Path(self.root).resolve(strict=True)
        if selected.is_symlink() or not selected.is_dir():
            raise RuntimeError("admitted local tree is no longer a regular directory")
        current = artifact_tree_digest(selected)
        if current != self.tree_sha256 or current != self.proof.artifact_sha256:
            raise RuntimeError("local tree no longer matches admitted artifact subject")
        return str(selected)


__all__ = ["AdmittedLocalArtifactTree"]
