"""Bind an exact local executable file to a governed binary admission proof.

This is the binary analogue of ``models.admitted_local_artifacts``.  It is intended for
serving/runtime tools such as OCR engines where the authoritative dependency is an
executable file rather than a Hugging Face model tree.  The binding rejects symlinks,
requires a regular file, re-hashes bytes before every authoritative use, and reuses the
repository's verifier-neutral artifact admission decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.admitted_local_artifacts import AdmittedArtifactProof

_HEX = frozenset("0123456789abcdef")
_MAX_EXECUTABLE_BYTES = 4 * 1024 * 1024 * 1024


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


def executable_file_sha256(path: str | Path) -> str:
    selected = Path(path).expanduser().resolve(strict=True)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError("executable path must be a non-symlink regular file")
    try:
        stat = selected.stat()
    except OSError as exc:
        raise RuntimeError("could not stat executable") from exc
    if stat.st_size < 1 or stat.st_size > _MAX_EXECUTABLE_BYTES:
        raise ValueError("executable size is outside the configured safety bound")
    digest = hashlib.sha256()
    total = 0
    try:
        with selected.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_EXECUTABLE_BYTES:
                    raise ValueError("executable exceeds the configured safety bound")
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError("could not read executable for identity verification") from exc
    return digest.hexdigest()


@dataclass(frozen=True)
class AdmittedLocalExecutable:
    path: str
    executable_sha256: str
    proof: AdmittedArtifactProof
    binding_sha256: str

    def __post_init__(self) -> None:
        selected = Path(self.path).expanduser().resolve(strict=True)
        if selected.is_symlink() or not selected.is_file():
            raise ValueError("path must resolve to a non-symlink regular file")
        object.__setattr__(self, "path", str(selected))
        object.__setattr__(self, "executable_sha256", _sha(self.executable_sha256, "executable_sha256"))
        if not isinstance(self.proof, AdmittedArtifactProof):
            raise ValueError("proof must be AdmittedArtifactProof")
        if self.proof.statement.subject.artifact_type != "binary":
            raise ValueError("executable proof subject must have artifact_type=binary")
        if self.proof.artifact_sha256 != self.executable_sha256:
            raise ValueError("binary admission proof differs from executable digest")
        expected = _digest(self._payload())
        provided = _sha(self.binding_sha256, "binding_sha256")
        if provided != expected:
            raise ValueError("binding_sha256 does not match admitted executable binding")
        object.__setattr__(self, "binding_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        # Do not include an operator-specific absolute path in the portable binding identity.
        return {
            "schema": "rigorousrag-admitted-local-executable/v1",
            "executable_sha256": self.executable_sha256,
            "proof_sha256": self.proof.proof_sha256,
        }

    @classmethod
    def build(cls, path: str | Path, *, proof: AdmittedArtifactProof) -> "AdmittedLocalExecutable":
        if not isinstance(proof, AdmittedArtifactProof):
            raise ValueError("proof must be AdmittedArtifactProof")
        selected = Path(path).expanduser().resolve(strict=True)
        sha = executable_file_sha256(selected)
        payload = {
            "schema": "rigorousrag-admitted-local-executable/v1",
            "executable_sha256": sha,
            "proof_sha256": proof.proof_sha256,
        }
        return cls(str(selected), sha, proof, _digest(payload))

    def verify(self) -> str:
        """Return the exact executable path after re-verifying current local bytes."""

        selected = Path(self.path).resolve(strict=True)
        if selected.is_symlink() or not selected.is_file():
            raise RuntimeError("admitted executable path is no longer a regular file")
        current = executable_file_sha256(selected)
        if current != self.executable_sha256 or current != self.proof.artifact_sha256:
            raise RuntimeError("local executable no longer matches admitted binary subject")
        if os.name != "nt" and not os.access(selected, os.X_OK):
            raise RuntimeError("admitted executable is not executable by the current process")
        return str(selected)


def require_admitted_local_executable(value: AdmittedLocalExecutable) -> str:
    if not isinstance(value, AdmittedLocalExecutable):
        raise ValueError("value must be AdmittedLocalExecutable")
    return value.verify()


__all__ = [
    "AdmittedLocalExecutable",
    "executable_file_sha256",
    "require_admitted_local_executable",
]
