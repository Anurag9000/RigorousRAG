"""Production-facing contracts with deterministic local implementations.

This module unifies object storage, queues, secrets, malware scanning, parser isolation,
egress policy and distributed-style admission semantics. Cloud/broker providers can be
implemented behind the same interfaces without changing retrieval/ingestion code.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from tools.security import normalize_owner_id


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ObjectRecord:
    owner_id: str
    object_id: str
    content_sha256: str
    size_bytes: int
    content_type: str
    version: int
    metadata: Mapping[str, str] = field(default_factory=dict)
    deleted: bool = False


class ObjectStore(Protocol):
    def put(self, owner_id: str, object_id: str, payload: bytes, *, content_type: str, metadata: Mapping[str, str] = {}) -> ObjectRecord: ...
    def get(self, owner_id: str, object_id: str, *, version: int | None = None) -> bytes: ...
    def head(self, owner_id: str, object_id: str) -> ObjectRecord: ...
    def delete(self, owner_id: str, object_id: str) -> ObjectRecord: ...


class InMemoryVersionedObjectStore:
    """Bounded reference backend with owner isolation and immutable versions."""

    def __init__(self, *, max_object_bytes: int = 250_000_000, max_objects: int = 100_000) -> None:
        self.max_object_bytes = max_object_bytes
        self.max_objects = max_objects
        self._records: dict[tuple[str, str], list[tuple[ObjectRecord, bytes]]] = {}
        self._lock = threading.RLock()

    def put(self, owner_id: str, object_id: str, payload: bytes, *, content_type: str, metadata: Mapping[str, str] = {}) -> ObjectRecord:
        owner = normalize_owner_id(owner_id)
        identifier = _text(object_id, "object_id", 500)
        if not isinstance(payload, bytes) or len(payload) > self.max_object_bytes:
            raise ValueError("object payload is invalid or exceeds the size limit")
        ctype = _text(content_type, "content_type", 200)
        if not isinstance(metadata, Mapping) or len(metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        safe_metadata = {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 500) for k, v in metadata.items()}
        key = (owner, identifier)
        with self._lock:
            if key not in self._records and len(self._records) >= self.max_objects:
                raise RuntimeError("object store capacity reached")
            versions = self._records.setdefault(key, [])
            version = len(versions) + 1
            record = ObjectRecord(owner, identifier, _digest(payload), len(payload), ctype, version, safe_metadata, False)
            versions.append((record, payload))
            return record

    def get(self, owner_id: str, object_id: str, *, version: int | None = None) -> bytes:
        owner = normalize_owner_id(owner_id)
        key = (owner, _text(object_id, "object_id", 500))
        with self._lock:
            versions = self._records[key]
            if version is None:
                record, payload = versions[-1]
            else:
                if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= len(versions):
                    raise KeyError("object version not found")
                record, payload = versions[version - 1]
            if record.deleted:
                raise KeyError("object is deleted")
            if _digest(payload) != record.content_sha256:
                raise RuntimeError("object content hash mismatch")
            return payload

    def head(self, owner_id: str, object_id: str) -> ObjectRecord:
        owner = normalize_owner_id(owner_id)
        with self._lock:
            return self._records[(owner, _text(object_id, "object_id", 500))][-1][0]

    def delete(self, owner_id: str, object_id: str) -> ObjectRecord:
        owner = normalize_owner_id(owner_id)
        identifier = _text(object_id, "object_id", 500)
        key = (owner, identifier)
        with self._lock:
            latest, _ = self._records[key][-1]
            record = ObjectRecord(owner, identifier, latest.content_sha256, 0, latest.content_type, latest.version + 1, latest.metadata, True)
            self._records[key].append((record, b""))
            return record


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    owner_id: str
    operation: str
    payload_sha256: str
    payload: bytes
    priority: int = 0
    available_at: float = 0.0
    attempt: int = 0


@dataclass(frozen=True)
class QueueLease:
    message: QueueMessage
    lease_id: str
    lease_until: float


class DurableQueue(Protocol):
    def enqueue(self, message: QueueMessage) -> None: ...
    def claim(self, worker_id: str, *, lease_seconds: float) -> QueueLease | None: ...
    def ack(self, lease: QueueLease) -> None: ...
    def retry(self, lease: QueueLease, *, delay_seconds: float) -> None: ...


class InMemoryLeaseQueue:
    def __init__(self, *, max_messages: int = 100_000, max_payload_bytes: int = 1_000_000) -> None:
        self.max_messages = max_messages
        self.max_payload_bytes = max_payload_bytes
        self._pending: list[QueueMessage] = []
        self._leased: dict[str, QueueLease] = {}
        self._lock = threading.RLock()

    def enqueue(self, message: QueueMessage) -> None:
        if not isinstance(message, QueueMessage) or len(message.payload) > self.max_payload_bytes or _digest(message.payload) != message.payload_sha256:
            raise ValueError("queue message is invalid")
        with self._lock:
            if len(self._pending) + len(self._leased) >= self.max_messages:
                raise RuntimeError("queue capacity reached")
            self._pending.append(message)

    def _requeue_expired(self, now: float) -> None:
        expired = [lease_id for lease_id, lease in self._leased.items() if lease.lease_until <= now]
        for lease_id in expired:
            lease = self._leased.pop(lease_id)
            msg = lease.message
            self._pending.append(QueueMessage(msg.message_id, msg.owner_id, msg.operation, msg.payload_sha256, msg.payload, msg.priority, now, msg.attempt + 1))

    def claim(self, worker_id: str, *, lease_seconds: float) -> QueueLease | None:
        worker = _text(worker_id, "worker_id", 256)
        del worker
        duration = float(lease_seconds)
        if not 1.0 <= duration <= 86_400.0:
            raise ValueError("lease_seconds is invalid")
        now = time.time()
        with self._lock:
            self._requeue_expired(now)
            eligible = [item for item in self._pending if item.available_at <= now]
            if not eligible:
                return None
            eligible.sort(key=lambda item: (-item.priority, item.available_at, item.message_id))
            message = eligible[0]
            self._pending.remove(message)
            lease = QueueLease(message, uuid.uuid4().hex, now + duration)
            self._leased[lease.lease_id] = lease
            return lease

    def ack(self, lease: QueueLease) -> None:
        with self._lock:
            current = self._leased.get(lease.lease_id)
            if current != lease:
                raise RuntimeError("queue lease is stale or unknown")
            del self._leased[lease.lease_id]

    def retry(self, lease: QueueLease, *, delay_seconds: float) -> None:
        delay = float(delay_seconds)
        if not 0 <= delay <= 7 * 86_400:
            raise ValueError("delay_seconds is invalid")
        with self._lock:
            current = self._leased.get(lease.lease_id)
            if current != lease:
                raise RuntimeError("queue lease is stale or unknown")
            del self._leased[lease.lease_id]
            msg = lease.message
            self._pending.append(QueueMessage(msg.message_id, msg.owner_id, msg.operation, msg.payload_sha256, msg.payload, msg.priority, time.time() + delay, msg.attempt + 1))


class SecretProvider(Protocol):
    def get(self, reference: str) -> str: ...


class EnvironmentSecretProvider:
    """Local provider using ``env://NAME`` references; secret values are never rendered."""
    def get(self, reference: str) -> str:
        ref = _text(reference, "secret reference", 500)
        if not ref.startswith("env://"):
            raise ValueError("environment secret references must use env://")
        name = ref[6:]
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", name):
            raise ValueError("environment secret name is invalid")
        value = os.environ.get(name)
        if value is None:
            raise KeyError("secret reference is unavailable")
        return value


@dataclass(frozen=True)
class ScanReceipt:
    scanner_id: str
    engine_version: str
    content_sha256: str
    status: str
    scanned_at: float


class MalwareScanner(Protocol):
    def scan(self, payload: bytes) -> ScanReceipt: ...


class ParserSandbox(Protocol):
    def parse(self, payload: bytes, *, filename: str, media_type: str, timeout_seconds: float, max_output_bytes: int) -> bytes: ...


@dataclass(frozen=True)
class EgressRule:
    host_pattern: str
    ports: tuple[int, ...] = (443,)
    schemes: tuple[str, ...] = ("https",)

    def __post_init__(self) -> None:
        pattern = _text(self.host_pattern, "host_pattern", 253).lower()
        if pattern.startswith("*."):
            base = pattern[2:]
            if not base or "*" in base:
                raise ValueError("wildcard host_pattern is invalid")
        elif "*" in pattern:
            raise ValueError("wildcards are only supported as a leading '*.'")
        object.__setattr__(self, "host_pattern", pattern)
        if not self.ports or len(self.ports) > 32 or any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in self.ports):
            raise ValueError("ports are invalid")
        schemes = tuple(dict.fromkeys(_text(item, "scheme", 16).lower() for item in self.schemes))
        if any(item not in {"http", "https"} for item in schemes):
            raise ValueError("unsupported egress scheme")
        object.__setattr__(self, "schemes", schemes)

    def matches_host(self, host: str) -> bool:
        host = host.rstrip(".").lower()
        if self.host_pattern.startswith("*."):
            base = self.host_pattern[2:]
            return host.endswith("." + base) and host != base
        return host == self.host_pattern


class EgressPolicy:
    def __init__(self, rules: Sequence[EgressRule]) -> None:
        if len(rules) > 256 or any(not isinstance(rule, EgressRule) for rule in rules):
            raise ValueError("egress rules are invalid")
        self.rules = tuple(rules)

    def authorize(self, url: str, *, resolver: Callable[[str], Sequence[str]] | None = None) -> tuple[str, int]:
        parsed = urlsplit(_text(url, "url", 4096))
        if parsed.username or parsed.password or parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise PermissionError("URL is not eligible for outbound access")
        host = parsed.hostname.rstrip(".").lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not any(rule.matches_host(host) and parsed.scheme in rule.schemes and port in rule.ports for rule in self.rules):
            raise PermissionError("destination is not allowlisted")
        addresses = list((resolver or _resolve_host)(host))
        if not addresses:
            raise PermissionError("destination did not resolve")
        for raw in addresses:
            address = ipaddress.ip_address(raw)
            if not address.is_global:
                raise PermissionError("destination resolved to a non-public address")
        return host, port


def _resolve_host(host: str) -> Sequence[str]:
    values = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(item[4][0] for item in values))


@dataclass
class _Bucket:
    tokens: float
    updated: float
    concurrent: int = 0


class AdmissionController:
    """Tenant/route token-bucket plus concurrent-work admission primitive."""

    def __init__(self, *, rate_per_second: float, burst: float, max_concurrent: int) -> None:
        self.rate = float(rate_per_second)
        self.burst = float(burst)
        self.max_concurrent = max_concurrent
        if self.rate <= 0 or self.burst < 1 or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("admission limits are invalid")
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    def acquire(self, key: str, *, cost: float = 1.0) -> bool:
        selected = _text(key, "admission key", 500)
        requested = float(cost)
        if requested <= 0 or requested > self.burst:
            raise ValueError("admission cost is invalid")
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(selected, _Bucket(self.burst, now))
            bucket.tokens = min(self.burst, bucket.tokens + max(0.0, now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.concurrent >= self.max_concurrent or bucket.tokens < requested:
                return False
            bucket.tokens -= requested
            bucket.concurrent += 1
            return True

    def release(self, key: str) -> None:
        selected = _text(key, "admission key", 500)
        with self._lock:
            bucket = self._buckets.get(selected)
            if bucket is None or bucket.concurrent <= 0:
                raise RuntimeError("no matching admitted operation")
            bucket.concurrent -= 1


__all__ = [
    "AdmissionController", "DurableQueue", "EgressPolicy", "EgressRule",
    "EnvironmentSecretProvider", "InMemoryLeaseQueue", "InMemoryVersionedObjectStore",
    "MalwareScanner", "ObjectRecord", "ObjectStore", "ParserSandbox", "QueueLease",
    "QueueMessage", "ScanReceipt", "SecretProvider",
]
