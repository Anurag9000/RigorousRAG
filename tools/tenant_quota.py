"""Durable owner-scoped quota reservations with leases and monotonic fencing."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id

_STATES = frozenset({"reserved", "committed", "released", "expired"})
_MAX_LIMIT = 1_000_000_000
_MAX_SECONDS = 31_536_000


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _positive_integer(value: Any, label: str, maximum: int = _MAX_LIMIT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be a positive bounded integer.")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_LIMIT:
        raise ValueError(f"{label} must be a non-negative bounded integer.")
    return value


def _positive(value: Any, label: str, maximum: float = 1.0e15) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive.") from exc
    if not math.isfinite(result) or not 0.0 < result <= maximum:
        raise ValueError(f"{label} must be finite and positive.")
    return result


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


class QuotaExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class TenantQuotaConfig:
    request_limit: int
    unit_limit: float
    inflight_limit: int
    window_seconds: int
    lease_seconds: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_limit",
            _positive_integer(self.request_limit, "request_limit"),
        )
        object.__setattr__(self, "unit_limit", _positive(self.unit_limit, "unit_limit"))
        object.__setattr__(
            self,
            "inflight_limit",
            _positive_integer(self.inflight_limit, "inflight_limit"),
        )
        window = _positive_integer(self.window_seconds, "window_seconds", _MAX_SECONDS)
        lease = _positive_integer(self.lease_seconds, "lease_seconds", _MAX_SECONDS)
        if lease > window:
            raise ValueError("lease_seconds may not exceed window_seconds.")
        object.__setattr__(self, "window_seconds", window)
        object.__setattr__(self, "lease_seconds", lease)


@dataclass(frozen=True)
class QuotaReservation:
    reservation_id: str
    owner_id: str
    window_start: float
    reserved_units: float
    state: str
    fencing_token: int
    lease_expires_at: float
    committed_units: float | None
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reservation_id",
            _identifier(self.reservation_id, "reservation_id"),
        )
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "window_start", _timestamp(self.window_start, "window_start"))
        object.__setattr__(self, "reserved_units", _positive(self.reserved_units, "reserved_units"))
        if self.state not in _STATES:
            raise ValueError("reservation state is unsupported.")
        object.__setattr__(
            self,
            "fencing_token",
            _positive_integer(self.fencing_token, "fencing_token", 2**63 - 1),
        )
        object.__setattr__(
            self,
            "lease_expires_at",
            _timestamp(self.lease_expires_at, "lease_expires_at"),
        )
        if self.committed_units is not None:
            committed = _positive(self.committed_units, "committed_units")
            if committed > self.reserved_units:
                raise ValueError("committed_units may not exceed reserved_units.")
            object.__setattr__(self, "committed_units", committed)
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.state == "committed" and self.committed_units is None:
            raise ValueError("committed reservation requires committed_units.")
        if self.state != "committed" and self.committed_units is not None:
            raise ValueError("non-committed reservation may not contain committed_units.")


@dataclass(frozen=True)
class TenantQuotaSnapshot:
    owner_id: str
    window_start: float
    committed_requests: int
    committed_units: float
    reserved_requests: int
    reserved_units: float
    active_inflight: int
    remaining_requests: int
    remaining_units: float


class TenantQuotaStore:
    """Atomic quota accounting. Reservation IDs carry no request payload or text."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        candidate = Path(os.fspath(path))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        self.path = candidate.absolute()
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_quota_config (
                    owner_id TEXT PRIMARY KEY,
                    request_limit INTEGER NOT NULL,
                    unit_limit REAL NOT NULL,
                    inflight_limit INTEGER NOT NULL,
                    window_seconds INTEGER NOT NULL,
                    lease_seconds INTEGER NOT NULL,
                    revision INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_quota_usage (
                    owner_id TEXT NOT NULL,
                    window_start REAL NOT NULL,
                    committed_requests INTEGER NOT NULL,
                    committed_units REAL NOT NULL,
                    PRIMARY KEY(owner_id, window_start)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_quota_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    window_start REAL NOT NULL,
                    reserved_units REAL NOT NULL,
                    state TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    lease_expires_at REAL NOT NULL,
                    committed_units REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS tenant_quota_active "
                "ON tenant_quota_reservations(owner_id, state, lease_expires_at)"
            )
        except sqlite3.Error as exc:
            raise RuntimeError("tenant quota store initialization failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def configure(self, owner_id: str, config: TenantQuotaConfig) -> int:
        owner = normalize_owner_id(owner_id)
        if not isinstance(config, TenantQuotaConfig):
            raise ValueError("config must be TenantQuotaConfig.")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT revision FROM tenant_quota_config WHERE owner_id=?",
                    (owner,),
                ).fetchone()
                revision = 1 if row is None else int(row[0]) + 1
                self._connection.execute(
                    """
                    INSERT INTO tenant_quota_config (
                        owner_id, request_limit, unit_limit, inflight_limit,
                        window_seconds, lease_seconds, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_id) DO UPDATE SET
                        request_limit=excluded.request_limit,
                        unit_limit=excluded.unit_limit,
                        inflight_limit=excluded.inflight_limit,
                        window_seconds=excluded.window_seconds,
                        lease_seconds=excluded.lease_seconds,
                        revision=excluded.revision
                    """,
                    (
                        owner,
                        config.request_limit,
                        config.unit_limit,
                        config.inflight_limit,
                        config.window_seconds,
                        config.lease_seconds,
                        revision,
                    ),
                )
                self._connection.execute("COMMIT")
                return revision
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota configuration failed.") from exc

    def _config(self, owner: str) -> TenantQuotaConfig:
        row = self._connection.execute(
            """
            SELECT request_limit, unit_limit, inflight_limit, window_seconds, lease_seconds
            FROM tenant_quota_config WHERE owner_id=?
            """,
            (owner,),
        ).fetchone()
        if row is None:
            raise RuntimeError("tenant quota is not configured for this owner.")
        try:
            return TenantQuotaConfig(*row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("tenant quota configuration is corrupt.") from exc

    @staticmethod
    def _window_start(now: float, config: TenantQuotaConfig) -> float:
        return float(math.floor(now / config.window_seconds) * config.window_seconds)

    @staticmethod
    def _reservation(row: tuple[Any, ...] | None) -> QuotaReservation | None:
        if row is None:
            return None
        try:
            return QuotaReservation(*row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("tenant quota reservation is corrupt.") from exc

    def _get_reservation(self, reservation_id: str) -> QuotaReservation | None:
        row = self._connection.execute(
            """
            SELECT reservation_id, owner_id, window_start, reserved_units, state,
                   fencing_token, lease_expires_at, committed_units, created_at, updated_at
            FROM tenant_quota_reservations WHERE reservation_id=?
            """,
            (reservation_id,),
        ).fetchone()
        return self._reservation(row)

    def _expire_locked(self, owner: str, now: float) -> int:
        cursor = self._connection.execute(
            """
            UPDATE tenant_quota_reservations
            SET state='expired', updated_at=?
            WHERE owner_id=? AND state='reserved' AND lease_expires_at<=?
            """,
            (now, owner, now),
        )
        return max(int(cursor.rowcount), 0)

    def reserve(
        self,
        *,
        owner_id: str,
        reservation_id: str,
        units: float,
        now: float,
    ) -> QuotaReservation:
        owner = normalize_owner_id(owner_id)
        reservation = _identifier(reservation_id, "reservation_id")
        requested_units = _positive(units, "units")
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                config = self._config(owner)
                self._expire_locked(owner, current_time)
                existing = self._get_reservation(reservation)
                if existing is not None:
                    if existing.owner_id == owner and existing.reserved_units == requested_units:
                        self._connection.execute("COMMIT")
                        return existing
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("reservation ID collision.")
                window_start = self._window_start(current_time, config)
                usage = self._connection.execute(
                    """
                    SELECT committed_requests, committed_units
                    FROM tenant_quota_usage WHERE owner_id=? AND window_start=?
                    """,
                    (owner, window_start),
                ).fetchone() or (0, 0.0)
                reserved = self._connection.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(reserved_units), 0.0)
                    FROM tenant_quota_reservations
                    WHERE owner_id=? AND window_start=? AND state='reserved' AND lease_expires_at>?
                    """,
                    (owner, window_start, current_time),
                ).fetchone()
                inflight = self._connection.execute(
                    """
                    SELECT COUNT(*) FROM tenant_quota_reservations
                    WHERE owner_id=? AND state='reserved' AND lease_expires_at>?
                    """,
                    (owner, current_time),
                ).fetchone()[0]
                committed_requests, committed_units = int(usage[0]), float(usage[1])
                reserved_requests, reserved_units = int(reserved[0]), float(reserved[1])
                if committed_requests + reserved_requests + 1 > config.request_limit:
                    self._connection.execute("ROLLBACK")
                    raise QuotaExceededError("request quota exceeded.")
                if committed_units + reserved_units + requested_units > config.unit_limit:
                    self._connection.execute("ROLLBACK")
                    raise QuotaExceededError("unit quota exceeded.")
                if int(inflight) + 1 > config.inflight_limit:
                    self._connection.execute("ROLLBACK")
                    raise QuotaExceededError("inflight quota exceeded.")
                lease_expires = current_time + config.lease_seconds
                self._connection.execute(
                    """
                    INSERT INTO tenant_quota_reservations (
                        reservation_id, owner_id, window_start, reserved_units, state,
                        fencing_token, lease_expires_at, committed_units, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'reserved', 1, ?, NULL, ?, ?)
                    """,
                    (
                        reservation,
                        owner,
                        window_start,
                        requested_units,
                        lease_expires,
                        current_time,
                        current_time,
                    ),
                )
                self._connection.execute("COMMIT")
                return self._get_reservation(reservation)  # type: ignore[return-value]
            except (RuntimeError, QuotaExceededError):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota reservation failed.") from exc

    def renew(
        self,
        *,
        owner_id: str,
        reservation_id: str,
        fencing_token: int,
        now: float,
    ) -> QuotaReservation:
        owner = normalize_owner_id(owner_id)
        reservation = _identifier(reservation_id, "reservation_id")
        token = _positive_integer(fencing_token, "fencing_token", 2**63 - 1)
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                config = self._config(owner)
                self._expire_locked(owner, current_time)
                current = self._get_reservation(reservation)
                if (
                    current is None
                    or current.owner_id != owner
                    or current.state != "reserved"
                    or current.fencing_token != token
                    or current.lease_expires_at <= current_time
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("reservation lease is stale or unavailable.")
                next_token = token + 1
                self._connection.execute(
                    """
                    UPDATE tenant_quota_reservations
                    SET fencing_token=?, lease_expires_at=?, updated_at=?
                    WHERE reservation_id=? AND owner_id=? AND state='reserved' AND fencing_token=?
                    """,
                    (
                        next_token,
                        current_time + config.lease_seconds,
                        current_time,
                        reservation,
                        owner,
                        token,
                    ),
                )
                self._connection.execute("COMMIT")
                return self._get_reservation(reservation)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota renewal failed.") from exc

    def commit(
        self,
        *,
        owner_id: str,
        reservation_id: str,
        fencing_token: int,
        now: float,
        actual_units: float | None = None,
    ) -> QuotaReservation:
        owner = normalize_owner_id(owner_id)
        reservation = _identifier(reservation_id, "reservation_id")
        token = _positive_integer(fencing_token, "fencing_token", 2**63 - 1)
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._config(owner)
                self._expire_locked(owner, current_time)
                current = self._get_reservation(reservation)
                if (
                    current is None
                    or current.owner_id != owner
                    or current.state != "reserved"
                    or current.fencing_token != token
                    or current.lease_expires_at <= current_time
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("reservation lease is stale or unavailable.")
                consumed = current.reserved_units if actual_units is None else _positive(
                    actual_units,
                    "actual_units",
                )
                if consumed > current.reserved_units:
                    self._connection.execute("ROLLBACK")
                    raise ValueError("actual_units may not exceed reserved_units.")
                self._connection.execute(
                    """
                    INSERT INTO tenant_quota_usage (
                        owner_id, window_start, committed_requests, committed_units
                    ) VALUES (?, ?, 1, ?)
                    ON CONFLICT(owner_id, window_start) DO UPDATE SET
                        committed_requests=committed_requests + 1,
                        committed_units=committed_units + excluded.committed_units
                    """,
                    (owner, current.window_start, consumed),
                )
                self._connection.execute(
                    """
                    UPDATE tenant_quota_reservations
                    SET state='committed', committed_units=?, updated_at=?
                    WHERE reservation_id=? AND owner_id=? AND state='reserved' AND fencing_token=?
                    """,
                    (consumed, current_time, reservation, owner, token),
                )
                self._connection.execute("COMMIT")
                return self._get_reservation(reservation)  # type: ignore[return-value]
            except (RuntimeError, ValueError):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota commit failed.") from exc

    def release(
        self,
        *,
        owner_id: str,
        reservation_id: str,
        fencing_token: int,
        now: float,
    ) -> QuotaReservation:
        owner = normalize_owner_id(owner_id)
        reservation = _identifier(reservation_id, "reservation_id")
        token = _positive_integer(fencing_token, "fencing_token", 2**63 - 1)
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._expire_locked(owner, current_time)
                current = self._get_reservation(reservation)
                if (
                    current is None
                    or current.owner_id != owner
                    or current.state != "reserved"
                    or current.fencing_token != token
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("reservation is stale or unavailable.")
                self._connection.execute(
                    """
                    UPDATE tenant_quota_reservations SET state='released', updated_at=?
                    WHERE reservation_id=? AND owner_id=? AND state='reserved' AND fencing_token=?
                    """,
                    (current_time, reservation, owner, token),
                )
                self._connection.execute("COMMIT")
                return self._get_reservation(reservation)  # type: ignore[return-value]
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota release failed.") from exc

    def snapshot(self, owner_id: str, *, now: float) -> TenantQuotaSnapshot:
        owner = normalize_owner_id(owner_id)
        current_time = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                config = self._config(owner)
                self._expire_locked(owner, current_time)
                window_start = self._window_start(current_time, config)
                usage = self._connection.execute(
                    """
                    SELECT committed_requests, committed_units FROM tenant_quota_usage
                    WHERE owner_id=? AND window_start=?
                    """,
                    (owner, window_start),
                ).fetchone() or (0, 0.0)
                reserved = self._connection.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(reserved_units), 0.0)
                    FROM tenant_quota_reservations
                    WHERE owner_id=? AND window_start=? AND state='reserved' AND lease_expires_at>?
                    """,
                    (owner, window_start, current_time),
                ).fetchone()
                inflight = int(
                    self._connection.execute(
                        """
                        SELECT COUNT(*) FROM tenant_quota_reservations
                        WHERE owner_id=? AND state='reserved' AND lease_expires_at>?
                        """,
                        (owner, current_time),
                    ).fetchone()[0]
                )
                self._connection.execute("COMMIT")
                committed_requests = _nonnegative_integer(int(usage[0]), "committed_requests")
                committed_units = float(usage[1])
                reserved_requests = _nonnegative_integer(int(reserved[0]), "reserved_requests")
                reserved_units = float(reserved[1])
                return TenantQuotaSnapshot(
                    owner_id=owner,
                    window_start=window_start,
                    committed_requests=committed_requests,
                    committed_units=committed_units,
                    reserved_requests=reserved_requests,
                    reserved_units=reserved_units,
                    active_inflight=inflight,
                    remaining_requests=max(
                        config.request_limit - committed_requests - reserved_requests,
                        0,
                    ),
                    remaining_units=max(
                        config.unit_limit - committed_units - reserved_units,
                        0.0,
                    ),
                )
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("tenant quota snapshot failed.") from exc


__all__ = [
    "QuotaExceededError",
    "QuotaReservation",
    "TenantQuotaConfig",
    "TenantQuotaSnapshot",
    "TenantQuotaStore",
]
