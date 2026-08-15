"""Injected production-provider adapters without SDK installation or network setup.

Applications provide already-configured SDK/client objects.  These adapters enforce the
RigorousRAG object/queue/secret contracts without importing or downloading provider
packages themselves.  They intentionally avoid provider-specific credential discovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from tools.production_runtime import ObjectRecord, QueueLease, QueueMessage
from tools.security import normalize_owner_id

_MAX_OBJECT_BYTES = 2_000_000_000
_MAX_QUEUE_BYTES = 1_000_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class InjectedS3ObjectStore:
    """S3-compatible owner/version object store over an injected client.

    Required client methods are ``put_object``, ``get_object``, ``head_object`` and
    ``delete_object``.  The adapter does not create buckets or discover credentials.
    """

    def __init__(self, *, client: Any, bucket: str, prefix: str = "rigorousrag") -> None:
        if client is None:
            raise ValueError("client must be supplied")
        self.client = client
        self.bucket = _text(bucket, "bucket", 255)
        raw_prefix = _text(prefix, "prefix", 500).strip("/")
        if ".." in raw_prefix.split("/"):
            raise ValueError("prefix may not contain parent traversal")
        self.prefix = raw_prefix

    def _key(self, owner_id: str, object_id: str) -> str:
        owner = normalize_owner_id(owner_id)
        identifier = _text(object_id, "object_id", 500)
        if identifier.startswith("/") or ".." in identifier.split("/"):
            raise ValueError("object_id is not a safe relative identifier")
        return f"{self.prefix}/{owner}/{identifier}"

    def put(self, owner_id: str, object_id: str, payload: bytes, *, content_type: str, metadata: Mapping[str, str] = {}) -> ObjectRecord:
        if not isinstance(payload, bytes) or len(payload) > _MAX_OBJECT_BYTES:
            raise ValueError("object payload is invalid")
        if not isinstance(metadata, Mapping) or len(metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        owner = normalize_owner_id(owner_id)
        key = self._key(owner, object_id)
        digest = _sha_bytes(payload)
        safe_metadata = {str(k)[:100]: str(v)[:500] for k, v in metadata.items()}
        safe_metadata.update({"rigorousrag-sha256": digest, "rigorousrag-owner": owner})
        result = self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType=_text(content_type, "content_type", 200),
            Metadata=safe_metadata,
        )
        version_raw = result.get("VersionId") if isinstance(result, Mapping) else None
        version = 1
        if version_raw:
            version = int(hashlib.sha256(str(version_raw).encode("utf-8")).hexdigest()[:12], 16)
        return ObjectRecord(owner, object_id, digest, len(payload), content_type, version, safe_metadata, False)

    def get(self, owner_id: str, object_id: str, *, version: int | None = None) -> bytes:
        if version is not None:
            raise ValueError("integer ObjectRecord versions are opaque; use provider lifecycle/version IDs outside this generic contract")
        owner = normalize_owner_id(owner_id)
        key = self._key(owner, object_id)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response.get("Body") if isinstance(response, Mapping) else None
        if body is None or not hasattr(body, "read"):
            raise RuntimeError("object provider returned no readable body")
        payload = body.read(_MAX_OBJECT_BYTES + 1)
        if not isinstance(payload, bytes) or len(payload) > _MAX_OBJECT_BYTES:
            raise RuntimeError("object provider returned an invalid or oversized payload")
        metadata = response.get("Metadata", {}) if isinstance(response, Mapping) else {}
        expected = metadata.get("rigorousrag-sha256") if isinstance(metadata, Mapping) else None
        if expected and expected != _sha_bytes(payload):
            raise RuntimeError("object content hash does not match provider metadata")
        return payload

    def head(self, owner_id: str, object_id: str) -> ObjectRecord:
        owner = normalize_owner_id(owner_id)
        response = self.client.head_object(Bucket=self.bucket, Key=self._key(owner, object_id))
        if not isinstance(response, Mapping):
            raise RuntimeError("object provider returned invalid metadata")
        metadata = response.get("Metadata", {})
        digest = str(metadata.get("rigorousrag-sha256", "")) if isinstance(metadata, Mapping) else ""
        if len(digest) != 64:
            raise RuntimeError("object provider metadata lacks the authoritative SHA-256")
        size = int(response.get("ContentLength", 0))
        if not 0 <= size <= _MAX_OBJECT_BYTES:
            raise RuntimeError("object provider reports an invalid size")
        return ObjectRecord(owner, object_id, digest, size, str(response.get("ContentType", "application/octet-stream")), 1, {str(k): str(v) for k, v in metadata.items()}, False)

    def delete(self, owner_id: str, object_id: str) -> ObjectRecord:
        current = self.head(owner_id, object_id)
        self.client.delete_object(Bucket=self.bucket, Key=self._key(owner_id, object_id))
        return ObjectRecord(current.owner_id, current.object_id, current.content_sha256, 0, current.content_type, current.version + 1, current.metadata, True)


class InjectedRedisLeaseQueue:
    """Redis-compatible lease queue over an injected client.

    Uses sorted sets for availability and hashes for immutable message bodies/leases.
    Atomic claim/retry operations use client ``eval``; callers supply a Redis-compatible
    connection but this module does not import redis-py.
    """

    _CLAIM_SCRIPT = """
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 1)
if #ids == 0 then return nil end
local id = ids[1]
if redis.call('ZREM', KEYS[1], id) ~= 1 then return nil end
local lease = ARGV[2]
local until_at = ARGV[3]
redis.call('HSET', KEYS[2], lease, id .. '|' .. until_at)
return id
"""

    def __init__(self, *, client: Any, namespace: str = "rigorousrag:queue") -> None:
        if client is None:
            raise ValueError("client must be supplied")
        self.client = client
        self.namespace = _text(namespace, "namespace", 200)
        self.pending_key = f"{self.namespace}:pending"
        self.messages_key = f"{self.namespace}:messages"
        self.leases_key = f"{self.namespace}:leases"

    def enqueue(self, message: QueueMessage) -> None:
        if not isinstance(message, QueueMessage) or len(message.payload) > _MAX_QUEUE_BYTES or _sha_bytes(message.payload) != message.payload_sha256:
            raise ValueError("queue message is invalid")
        payload = _json({**asdict(message), "payload": message.payload.hex()})
        existing = self.client.hget(self.messages_key, message.message_id)
        if existing is not None:
            existing_text = existing.decode("utf-8") if isinstance(existing, bytes) else str(existing)
            if existing_text != payload:
                raise RuntimeError("queue message ID collision")
            return
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hset(self.messages_key, message.message_id, payload)
        pipeline.zadd(self.pending_key, {message.message_id: float(message.available_at)})
        pipeline.execute()

    def claim(self, worker_id: str, *, lease_seconds: float) -> QueueLease | None:
        _text(worker_id, "worker_id", 256)
        duration = float(lease_seconds)
        if not math.isfinite(duration) or not 1 <= duration <= 86_400:
            raise ValueError("lease_seconds is invalid")
        now = time.time()
        lease_id = uuid.uuid4().hex
        until_at = now + duration
        raw_id = self.client.eval(self._CLAIM_SCRIPT, 2, self.pending_key, self.leases_key, now, lease_id, until_at)
        if raw_id is None:
            return None
        message_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
        raw = self.client.hget(self.messages_key, message_id)
        if raw is None:
            self.client.hdel(self.leases_key, lease_id)
            raise RuntimeError("claimed queue message body is missing")
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        payload = bytes.fromhex(data.pop("payload"))
        message = QueueMessage(payload=payload, **data)
        if _sha_bytes(payload) != message.payload_sha256:
            raise RuntimeError("claimed queue payload hash mismatch")
        return QueueLease(message, lease_id, until_at)

    def ack(self, lease: QueueLease) -> None:
        raw = self.client.hget(self.leases_key, lease.lease_id)
        if raw is None:
            raise RuntimeError("queue lease is stale or unknown")
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        message_id = text.split("|", 1)[0]
        if message_id != lease.message.message_id:
            raise RuntimeError("queue lease identity mismatch")
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hdel(self.leases_key, lease.lease_id)
        pipeline.hdel(self.messages_key, message_id)
        pipeline.execute()

    def retry(self, lease: QueueLease, *, delay_seconds: float) -> None:
        delay = float(delay_seconds)
        if not math.isfinite(delay) or not 0 <= delay <= 7 * 86_400:
            raise ValueError("delay_seconds is invalid")
        raw = self.client.hget(self.leases_key, lease.lease_id)
        if raw is None:
            raise RuntimeError("queue lease is stale or unknown")
        message_id = lease.message.message_id
        current = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if current.split("|", 1)[0] != message_id:
            raise RuntimeError("queue lease identity mismatch")
        message = QueueMessage(
            message_id=lease.message.message_id,
            owner_id=lease.message.owner_id,
            operation=lease.message.operation,
            payload_sha256=lease.message.payload_sha256,
            payload=lease.message.payload,
            priority=lease.message.priority,
            available_at=time.time() + delay,
            attempt=lease.message.attempt + 1,
        )
        payload = _json({**asdict(message), "payload": message.payload.hex()})
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hset(self.messages_key, message_id, payload)
        pipeline.hdel(self.leases_key, lease.lease_id)
        pipeline.zadd(self.pending_key, {message_id: message.available_at})
        pipeline.execute()


class InjectedSecretProvider:
    """Generic secret provider over an injected read callable/client method."""

    def __init__(self, *, client: Any, scheme: str, read_method: str = "read") -> None:
        if client is None:
            raise ValueError("client must be supplied")
        self.client = client
        self.scheme = _text(scheme, "scheme", 32).lower()
        self.read_method = _text(read_method, "read_method", 64)

    def get(self, reference: str) -> str:
        ref = _text(reference, "secret reference", 1000)
        prefix = f"{self.scheme}://"
        if not ref.startswith(prefix):
            raise ValueError(f"secret reference must use {prefix}")
        path = ref[len(prefix):]
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("secret reference path is invalid")
        method = getattr(self.client, self.read_method, None)
        if not callable(method):
            raise RuntimeError("secret client does not expose the configured read method")
        result = method(path)
        if isinstance(result, Mapping):
            value = result.get("value")
            if value is None and isinstance(result.get("data"), Mapping):
                value = result["data"].get("value")
        else:
            value = result
        if not isinstance(value, str) or not value:
            raise RuntimeError("secret provider returned no string value")
        return value


__all__ = ["InjectedRedisLeaseQueue", "InjectedS3ObjectStore", "InjectedSecretProvider"]
