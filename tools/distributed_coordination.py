"""Fenced distributed lease contracts with in-memory, SQL, and Redis providers."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Lease:
    name: str
    holder: str
    token: int
    expires_at: float


class LeaseCoordinator(Protocol):
    def acquire(self, *, name: str, holder: str, ttl_seconds: float) -> Lease | None: ...
    def renew(self, lease: Lease, *, ttl_seconds: float) -> Lease | None: ...
    def release(self, lease: Lease) -> bool: ...


def _text(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _ttl(value: float) -> float:
    parsed = float(value)
    if not 0.05 <= parsed <= 86_400.0:
        raise ValueError("ttl_seconds is outside the supported range.")
    return parsed


class InMemoryLeaseCoordinator:
    """Deterministic test/local coordinator with monotonically increasing fencing tokens."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._leases: dict[str, Lease] = {}
        self._tokens: dict[str, int] = {}

    def acquire(self, *, name: str, holder: str, ttl_seconds: float) -> Lease | None:
        key = _text(name, "name")
        owner = _text(holder, "holder")
        ttl = _ttl(ttl_seconds)
        with self._lock:
            now = self._clock()
            current = self._leases.get(key)
            if current is not None and current.expires_at > now and current.holder != owner:
                return None
            token = self._tokens.get(key, 0) + 1
            self._tokens[key] = token
            lease = Lease(key, owner, token, now + ttl)
            self._leases[key] = lease
            return lease

    def renew(self, lease: Lease, *, ttl_seconds: float) -> Lease | None:
        ttl = _ttl(ttl_seconds)
        with self._lock:
            current = self._leases.get(lease.name)
            now = self._clock()
            if current != lease or lease.expires_at <= now:
                return None
            renewed = Lease(lease.name, lease.holder, lease.token, now + ttl)
            self._leases[lease.name] = renewed
            return renewed

    def release(self, lease: Lease) -> bool:
        with self._lock:
            if self._leases.get(lease.name) != lease:
                return False
            del self._leases[lease.name]
            return True


class SQLLeaseCoordinator:
    """Durable fenced coordinator over DB-API SQL.

    The default factory uses SQLite for single-host durability.  A PostgreSQL connection
    factory can be supplied with ``placeholder='%s'``; the state transition remains a
    compare-and-swap transaction and fencing tokens monotonically increase per lease name.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        connection_factory: Callable[[], Any] | None = None,
        placeholder: str = "?",
        clock: Callable[[], float] = time.time,
    ) -> None:
        if connection_factory is None:
            if path is None:
                raise ValueError("path or connection_factory is required.")
            selected_path = Path(path)
            selected_path.parent.mkdir(parents=True, exist_ok=True)
            connection_factory = lambda: sqlite3.connect(str(selected_path), timeout=10.0)
        if placeholder not in {"?", "%s"}:
            raise ValueError("placeholder is unsupported.")
        self._connect = connection_factory
        self._p = placeholder
        self._clock = clock
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS rigorousrag_leases ("
                "name TEXT PRIMARY KEY, holder TEXT NOT NULL, token BIGINT NOT NULL, expires_at DOUBLE PRECISION NOT NULL)"
            )

    def acquire(self, *, name: str, holder: str, ttl_seconds: float) -> Lease | None:
        key = _text(name, "name")
        owner = _text(holder, "holder")
        ttl = _ttl(ttl_seconds)
        now = self._clock()
        expires = now + ttl
        p = self._p
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"INSERT INTO rigorousrag_leases(name,holder,token,expires_at) "
                f"VALUES({p},{p},{p},{p}) ON CONFLICT(name) DO NOTHING",
                (key, owner, 1, expires),
            )
            if cursor.rowcount == 1:
                return Lease(key, owner, 1, expires)
            cursor.execute(
                f"SELECT holder,token,expires_at FROM rigorousrag_leases WHERE name={p}", (key,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            current_holder, token, current_expires = str(row[0]), int(row[1]), float(row[2])
            if current_expires > now and current_holder != owner:
                return None
            next_token = token + 1
            cursor.execute(
                f"UPDATE rigorousrag_leases SET holder={p},token={p},expires_at={p} "
                f"WHERE name={p} AND token={p}",
                (owner, next_token, expires, key, token),
            )
            if cursor.rowcount != 1:
                return None
            return Lease(key, owner, next_token, expires)

    def renew(self, lease: Lease, *, ttl_seconds: float) -> Lease | None:
        ttl = _ttl(ttl_seconds)
        now = self._clock()
        if lease.expires_at <= now:
            return None
        expires = now + ttl
        p = self._p
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"UPDATE rigorousrag_leases SET expires_at={p} WHERE name={p} AND holder={p} "
                f"AND token={p} AND expires_at>{p}",
                (expires, lease.name, lease.holder, lease.token, now),
            )
            if cursor.rowcount != 1:
                return None
        return Lease(lease.name, lease.holder, lease.token, expires)

    def release(self, lease: Lease) -> bool:
        p = self._p
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"DELETE FROM rigorousrag_leases WHERE name={p} AND holder={p} AND token={p}",
                (lease.name, lease.holder, lease.token),
            )
            return cursor.rowcount == 1


class RedisLeaseCoordinator:
    """Concrete redis-py compatible fenced lease adapter.

    The injected client must provide ``set``, ``get``, ``incr`` and ``eval`` methods. Lease
    ownership is represented by an opaque value containing holder and fencing token; renew and
    release use compare-and-set Lua scripts so an expired holder cannot affect its successor.
    """

    _RENEW = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('pexpire', KEYS[1], ARGV[2])
    end
    return 0
    """
    _RELEASE = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('del', KEYS[1])
    end
    return 0
    """

    def __init__(
        self,
        client: Any,
        *,
        prefix: str = "rigorousrag:lease",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._prefix = _text(prefix, "prefix")
        self._clock = clock

    def _keys(self, name: str) -> tuple[str, str]:
        key = _text(name, "name")
        return f"{self._prefix}:{key}", f"{self._prefix}:token:{key}"

    @staticmethod
    def _value(holder: str, token: int) -> str:
        return f"{holder}:{token}"

    def acquire(self, *, name: str, holder: str, ttl_seconds: float) -> Lease | None:
        owner = _text(holder, "holder")
        ttl = _ttl(ttl_seconds)
        key, token_key = self._keys(name)
        token = int(self._client.incr(token_key))
        value = self._value(owner, token)
        milliseconds = max(1, int(ttl * 1_000))
        acquired = self._client.set(key, value, nx=True, px=milliseconds)
        if not acquired:
            return None
        return Lease(_text(name, "name"), owner, token, self._clock() + ttl)

    def renew(self, lease: Lease, *, ttl_seconds: float) -> Lease | None:
        ttl = _ttl(ttl_seconds)
        if lease.expires_at <= self._clock():
            return None
        key, _ = self._keys(lease.name)
        milliseconds = max(1, int(ttl * 1_000))
        result = self._client.eval(
            self._RENEW, 1, key, self._value(lease.holder, lease.token), milliseconds
        )
        if int(result or 0) != 1:
            return None
        return Lease(lease.name, lease.holder, lease.token, self._clock() + ttl)

    def release(self, lease: Lease) -> bool:
        key, _ = self._keys(lease.name)
        result = self._client.eval(self._RELEASE, 1, key, self._value(lease.holder, lease.token))
        return int(result or 0) == 1


def new_worker_id(prefix: str = "worker") -> str:
    return f"{_text(prefix, 'prefix', 100)}-{secrets.token_hex(8)}"


__all__ = [
    "InMemoryLeaseCoordinator",
    "Lease",
    "LeaseCoordinator",
    "RedisLeaseCoordinator",
    "SQLLeaseCoordinator",
    "new_worker_id",
]
