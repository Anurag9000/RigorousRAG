"""Backup retention, immutability, and legal-hold policy planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackupRecord:
    backup_id: str
    created_at: float
    manifest_sha256: str
    immutable_until: float = 0.0
    legal_hold: bool = False
    verified: bool = True

    def __post_init__(self) -> None:
        if not self.backup_id.strip():
            raise ValueError("backup_id must be non-empty")
        digest = self.manifest_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("manifest_sha256 must be a SHA-256 digest")
        if self.created_at < 0.0 or self.immutable_until < 0.0:
            raise ValueError("backup timestamps must not be negative")


@dataclass(frozen=True)
class RetentionPolicy:
    minimum_recovery_points: int = 3
    max_age_seconds: float = 30.0 * 24.0 * 3600.0
    retain_unverified: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.minimum_recovery_points, bool) or self.minimum_recovery_points < 1:
            raise ValueError("minimum_recovery_points must be positive")
        if self.max_age_seconds < 0.0:
            raise ValueError("max_age_seconds must not be negative")


@dataclass(frozen=True)
class RetentionPlan:
    retain: tuple[str, ...]
    delete: tuple[str, ...]
    protected_reasons: tuple[tuple[str, tuple[str, ...]], ...]


def plan_retention(
    records: tuple[BackupRecord, ...] | list[BackupRecord],
    *,
    now: float,
    policy: RetentionPolicy | None = None,
) -> RetentionPlan:
    """Select only backups that are safe to delete under every configured protection."""

    selected = policy or RetentionPolicy()
    current = float(now)
    if current < 0.0:
        raise ValueError("now must not be negative")
    by_id = {record.backup_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("backup_id values must be unique")

    newest = sorted(records, key=lambda item: (item.created_at, item.backup_id), reverse=True)
    minimum_ids = {item.backup_id for item in newest[: selected.minimum_recovery_points]}
    retain: list[str] = []
    delete: list[str] = []
    reasons: list[tuple[str, tuple[str, ...]]] = []
    for record in sorted(records, key=lambda item: (item.created_at, item.backup_id)):
        protected: list[str] = []
        if record.backup_id in minimum_ids:
            protected.append("minimum_recovery_point")
        if record.legal_hold:
            protected.append("legal_hold")
        if record.immutable_until > current:
            protected.append("immutability_window")
        if selected.retain_unverified and not record.verified:
            protected.append("unverified_backup")
        age = max(0.0, current - record.created_at)
        if age <= selected.max_age_seconds:
            protected.append("retention_window")
        if protected:
            retain.append(record.backup_id)
            reasons.append((record.backup_id, tuple(protected)))
        else:
            delete.append(record.backup_id)
    return RetentionPlan(tuple(retain), tuple(delete), tuple(reasons))


__all__ = ["BackupRecord", "RetentionPlan", "RetentionPolicy", "plan_retention"]
