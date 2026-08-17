"""Immutable owner-scoped persistence for signed human-review decisions.

Only the privacy-bounded review manifest from ``tools.review_attestation`` is persisted.
Raw review queries, reasons, and arbitrary metadata are intentionally absent: the captured
manifest contains only their digests plus the decision identity needed for verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.manifest_attestation import ManifestAttestation, canonical_manifest_bytes
from tools.review_attestation import ReviewDecisionAttestation
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000
_MAX_MANIFEST_BYTES = 64 * 1024


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: str, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} is invalid")
    return parsed


def _manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("captured_manifest must be a mapping")
    selected = dict(value)
    encoded = canonical_manifest_bytes(selected)
    if not encoded or len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError("captured_manifest exceeds the byte limit")
    return selected


def _attestation_from_mapping(value: Mapping[str, Any]) -> ManifestAttestation:
    return ManifestAttestation(
        subject_id=str(value["subject_id"]),
        manifest_sha256=str(value["manifest_sha256"]),
        key_id=str(value["key_id"]),
        algorithm=str(value["algorithm"]),
        signature_b64=str(value["signature_b64"]),
        signed_at=float(value["signed_at"]),
    )


def _review_attestation_from_mapping(value: Mapping[str, Any]) -> ReviewDecisionAttestation:
    envelope = value.get("attestation")
    if not isinstance(envelope, Mapping):
        raise ValueError("stored review attestation envelope is invalid")
    return ReviewDecisionAttestation(
        request_id=str(value["request_id"]),
        owner_id=str(value["owner_id"]),
        review_manifest_sha256=str(value["review_manifest_sha256"]),
        attestation=_attestation_from_mapping(envelope),
    )


@dataclass(frozen=True)
class StoredReviewAttestation:
    attestation_id: str
    owner_id: str
    request_id: str
    lease_token: int
    reviewer_id: str
    resolution: str
    captured_manifest: Mapping[str, Any]
    signed: ReviewDecisionAttestation
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "attestation_id", _sha(self.attestation_id, "attestation_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id", 500))
        if isinstance(self.lease_token, bool) or not isinstance(self.lease_token, int) or self.lease_token < 1:
            raise ValueError("lease_token must be positive")
        object.__setattr__(self, "reviewer_id", _text(self.reviewer_id, "reviewer_id", 500))
        object.__setattr__(self, "resolution", _text(self.resolution, "resolution", 500))
        manifest = _manifest(self.captured_manifest)
        object.__setattr__(self, "captured_manifest", manifest)
        if not isinstance(self.signed, ReviewDecisionAttestation):
            raise TypeError("signed must be ReviewDecisionAttestation")
        if self.signed.owner_id != self.owner_id or self.signed.request_id != self.request_id:
            raise ValueError("signed attestation identity does not match stored identity")
        digest = hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
        if digest != self.signed.review_manifest_sha256:
            raise ValueError("captured manifest does not match review attestation digest")
        if self.signed.attestation.manifest_sha256 != digest:
            raise ValueError("manifest attestation digest does not match captured manifest")
        if self.signed.fingerprint != self.attestation_id:
            raise ValueError("attestation_id does not match signed attestation fingerprint")
        manifest_owner = str(manifest.get("owner_id", ""))
        manifest_request = str(manifest.get("request_id", ""))
        manifest_reviewer = str(manifest.get("reviewer_id", ""))
        manifest_resolution = str(manifest.get("resolution", ""))
        try:
            manifest_lease = int(manifest.get("lease_token", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("captured manifest lease token is invalid") from exc
        if (
            manifest_owner != self.owner_id
            or manifest_request != self.request_id
            or manifest_reviewer != self.reviewer_id
            or manifest_resolution != self.resolution
            or manifest_lease != self.lease_token
        ):
            raise ValueError("stored review metadata does not match captured manifest")
        object.__setattr__(self, "created_at", _finite(self.created_at, "created_at"))

    @property
    def fingerprint(self) -> str:
        return self.attestation_id


class ReviewAttestationStore:
    """SQLite immutable reference store for review attestations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            mode = self.path.stat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("review attestation database may not be a symlink")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS review_attestations (
                    owner_id TEXT NOT NULL,
                    attestation_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    lease_token INTEGER NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    captured_manifest_json TEXT NOT NULL,
                    signed_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id,attestation_id)
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_attestation_request ON review_attestations(owner_id,request_id,created_at,attestation_id)"
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _row(row: sqlite3.Row | Mapping[str, Any]) -> StoredReviewAttestation:
        captured = json.loads(str(row["captured_manifest_json"]))
        signed_raw = json.loads(str(row["signed_json"]))
        if not isinstance(captured, Mapping) or not isinstance(signed_raw, Mapping):
            raise RuntimeError("stored review attestation JSON is invalid")
        return StoredReviewAttestation(
            attestation_id=str(row["attestation_id"]),
            owner_id=str(row["owner_id"]),
            request_id=str(row["request_id"]),
            lease_token=int(row["lease_token"]),
            reviewer_id=str(row["reviewer_id"]),
            resolution=str(row["resolution"]),
            captured_manifest=dict(captured),
            signed=_review_attestation_from_mapping(signed_raw),
            created_at=float(row["created_at"]),
        )

    def put(
        self,
        *,
        owner_id: str,
        captured_manifest: Mapping[str, Any],
        signed: ReviewDecisionAttestation,
        now: float | None = None,
    ) -> StoredReviewAttestation:
        owner = normalize_owner_id(owner_id)
        if not isinstance(signed, ReviewDecisionAttestation) or signed.owner_id != owner:
            raise ValueError("signed review attestation owner does not match store owner")
        manifest = _manifest(captured_manifest)
        request_id = _text(str(manifest.get("request_id", "")), "request_id", 500)
        reviewer_id = _text(str(manifest.get("reviewer_id", "")), "reviewer_id", 500)
        resolution = _text(str(manifest.get("resolution", "")), "resolution", 500)
        try:
            lease_token = int(manifest.get("lease_token", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("captured manifest lease token is invalid") from exc
        selected_now = time.time() if now is None else _finite(now, "now")
        item = StoredReviewAttestation(
            attestation_id=signed.fingerprint,
            owner_id=owner,
            request_id=request_id,
            lease_token=lease_token,
            reviewer_id=reviewer_id,
            resolution=resolution,
            captured_manifest=manifest,
            signed=signed,
            created_at=selected_now,
        )
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        signed_json = json.dumps(asdict(signed), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO review_attestations
                    (owner_id,attestation_id,request_id,lease_token,reviewer_id,resolution,captured_manifest_json,signed_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    item.owner_id,
                    item.attestation_id,
                    item.request_id,
                    item.lease_token,
                    item.reviewer_id,
                    item.resolution,
                    manifest_json,
                    signed_json,
                    item.created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM review_attestations WHERE owner_id=? AND attestation_id=?",
                (owner, item.attestation_id),
            ).fetchone()
            connection.commit()
        if row is None:
            raise RuntimeError("review attestation persistence failed")
        stored = self._row(row)
        if stored != item:
            # A content-addressed ID must never resolve to divergent persisted content.
            raise RuntimeError("review attestation identity collision")
        return stored

    def get(self, *, owner_id: str, attestation_id: str) -> StoredReviewAttestation | None:
        owner = normalize_owner_id(owner_id)
        identity = _sha(attestation_id, "attestation_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_attestations WHERE owner_id=? AND attestation_id=?",
                (owner, identity),
            ).fetchone()
        return None if row is None else self._row(row)

    def list(
        self,
        *,
        owner_id: str,
        request_id: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredReviewAttestation, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError("limit is invalid")
        params: list[Any] = [owner]
        where = "owner_id=?"
        if request_id is not None:
            where += " AND request_id=?"
            params.append(_text(request_id, "request_id", 500))
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM review_attestations WHERE {where} ORDER BY created_at DESC,attestation_id DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return tuple(self._row(row) for row in rows)


__all__ = ["ReviewAttestationStore", "StoredReviewAttestation"]
