"""Durable blue/green physical vector-collection registry and route journal."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from tools.embedding_models import EmbeddingProfile
from tools.security import normalize_owner_id

_COLLECTION_STATES = frozenset({"ready", "retired"})
_ROUTE_ACTIONS = frozenset({"bootstrap", "switch", "rollback", "generation_advance"})
_MAX_TIME = 1.0e15
_MAX_DIMENSIONS = 1_000_000
_MAX_HISTORY = 10_000


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


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _positive_integer(value: Any, label: str, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= _MAX_TIME:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PhysicalVectorCollection:
    collection_id: str
    collection_name: str
    profile_alias: str
    profile_fingerprint: str
    model_name: str
    dimensions: int
    state: str
    created_at: float
    retired_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collection_id", _digest(self.collection_id, "collection_id"))
        object.__setattr__(
            self,
            "collection_name",
            _identifier(self.collection_name, "collection_name", 200),
        )
        object.__setattr__(
            self,
            "profile_alias",
            _identifier(self.profile_alias, "profile_alias", 128),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(self, "model_name", _identifier(self.model_name, "model_name", 300))
        object.__setattr__(
            self,
            "dimensions",
            _positive_integer(self.dimensions, "dimensions", _MAX_DIMENSIONS),
        )
        if self.state not in _COLLECTION_STATES:
            raise ValueError("collection state is unsupported.")
        created = _timestamp(self.created_at, "created_at")
        object.__setattr__(self, "created_at", created)
        if self.retired_at is not None:
            retired = _timestamp(self.retired_at, "retired_at")
            if retired < created:
                raise ValueError("retired_at may not precede created_at.")
            object.__setattr__(self, "retired_at", retired)
        if self.state == "ready" and self.retired_at is not None:
            raise ValueError("ready collection may not have retired_at.")
        if self.state == "retired" and self.retired_at is None:
            raise ValueError("retired collection requires retired_at.")
        expected_id = _canonical_digest(
            {
                "contract": "rigorousrag-physical-vector-collection-v1",
                "profile_alias": self.profile_alias,
                "profile_fingerprint": self.profile_fingerprint,
                "model_name": self.model_name,
                "dimensions": self.dimensions,
            }
        )
        if self.collection_id != expected_id:
            raise ValueError("collection_id does not match the immutable collection specification.")
        expected_name = f"rrag-{self.profile_fingerprint[:20]}-d{self.dimensions}"
        if self.collection_name != expected_name:
            raise ValueError("collection_name does not match the deterministic collection identity.")


@dataclass(frozen=True)
class VectorRouteRevision:
    owner_id: str
    doc_id: str
    revision: int
    collection_id: str
    profile_fingerprint: str
    generation_sequence: int
    action: str
    operation_id: str | None
    previous_revision: int | None
    previous_collection_id: str | None
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "revision", _positive_integer(self.revision, "revision"))
        object.__setattr__(self, "collection_id", _digest(self.collection_id, "collection_id"))
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "generation_sequence",
            _positive_integer(self.generation_sequence, "generation_sequence"),
        )
        if self.action not in _ROUTE_ACTIONS:
            raise ValueError("route action is unsupported.")
        if self.operation_id is not None:
            object.__setattr__(self, "operation_id", _digest(self.operation_id, "operation_id"))
        if self.previous_revision is not None:
            object.__setattr__(
                self,
                "previous_revision",
                _positive_integer(self.previous_revision, "previous_revision"),
            )
        if self.previous_collection_id is not None:
            object.__setattr__(
                self,
                "previous_collection_id",
                _digest(self.previous_collection_id, "previous_collection_id"),
            )
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.action == "bootstrap":
            if (
                self.revision != 1
                or self.operation_id is not None
                or self.previous_revision is not None
                or self.previous_collection_id is not None
            ):
                raise ValueError("bootstrap route identity is inconsistent.")
        else:
            if (
                self.revision <= 1
                or self.operation_id is None
                or self.previous_revision != self.revision - 1
                or self.previous_collection_id is None
            ):
                raise ValueError("transition route identity is inconsistent.")

    @property
    def route_digest(self) -> str:
        return _canonical_digest(asdict(self))


class VectorCollectionRegistry:
    """Append-only route history with a CAS head over immutable collection specs."""

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
                CREATE TABLE IF NOT EXISTS vector_collections (
                    collection_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL UNIQUE,
                    profile_alias TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL UNIQUE,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    retired_at REAL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_route_revisions (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    collection_id TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    generation_sequence INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    operation_id TEXT,
                    previous_revision INTEGER,
                    previous_collection_id TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, doc_id, revision)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_route_heads (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, doc_id)
                )
                """
            )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS vector_route_operation_unique "
                "ON vector_route_revisions(owner_id, doc_id, operation_id) "
                "WHERE operation_id IS NOT NULL"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS vector_route_collection_heads "
                "ON vector_route_revisions(collection_id, owner_id, doc_id, revision)"
            )
        except sqlite3.Error as exc:
            raise RuntimeError("vector collection registry initialization failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _collection(row: tuple[Any, ...] | None) -> PhysicalVectorCollection | None:
        if row is None:
            return None
        try:
            return PhysicalVectorCollection(*row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("physical vector collection record is corrupt.") from exc

    @staticmethod
    def _route(row: tuple[Any, ...] | None) -> VectorRouteRevision | None:
        if row is None:
            return None
        try:
            return VectorRouteRevision(*row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("vector route revision is corrupt.") from exc

    def _get_collection(self, collection_id: str) -> PhysicalVectorCollection | None:
        row = self._connection.execute(
            """
            SELECT collection_id, collection_name, profile_alias, profile_fingerprint,
                   model_name, dimensions, state, created_at, retired_at
            FROM vector_collections WHERE collection_id=?
            """,
            (collection_id,),
        ).fetchone()
        return self._collection(row)

    def get_collection(self, collection_id: str) -> PhysicalVectorCollection | None:
        selected = _digest(collection_id, "collection_id")
        with self._lock:
            try:
                return self._get_collection(selected)
            except sqlite3.Error as exc:
                raise RuntimeError("vector collection registry read failed.") from exc

    def collection_for_profile(self, profile_fingerprint: str) -> PhysicalVectorCollection | None:
        fingerprint = _digest(profile_fingerprint, "profile_fingerprint")
        with self._lock:
            try:
                row = self._connection.execute(
                    """
                    SELECT collection_id, collection_name, profile_alias, profile_fingerprint,
                           model_name, dimensions, state, created_at, retired_at
                    FROM vector_collections WHERE profile_fingerprint=?
                    """,
                    (fingerprint,),
                ).fetchone()
                return self._collection(row)
            except sqlite3.Error as exc:
                raise RuntimeError("vector collection registry read failed.") from exc

    def register_collection(
        self,
        profile: EmbeddingProfile,
        *,
        now: float,
    ) -> PhysicalVectorCollection:
        if not isinstance(profile, EmbeddingProfile):
            raise ValueError("profile must be EmbeddingProfile.")
        if profile.dimensions is None:
            raise ValueError("physical vector collection requires explicit dimensions.")
        created = _timestamp(now, "now")
        collection_id = _canonical_digest(
            {
                "contract": "rigorousrag-physical-vector-collection-v1",
                "profile_alias": profile.alias,
                "profile_fingerprint": profile.fingerprint,
                "model_name": profile.model_name,
                "dimensions": profile.dimensions,
            }
        )
        value = PhysicalVectorCollection(
            collection_id=collection_id,
            collection_name=f"rrag-{profile.fingerprint[:20]}-d{profile.dimensions}",
            profile_alias=profile.alias,
            profile_fingerprint=profile.fingerprint,
            model_name=profile.model_name,
            dimensions=profile.dimensions,
            state="ready",
            created_at=created,
        )
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._get_collection(collection_id)
                if existing is not None:
                    self._connection.execute("COMMIT")
                    if (
                        existing.profile_fingerprint != value.profile_fingerprint
                        or existing.model_name != value.model_name
                        or existing.dimensions != value.dimensions
                        or existing.collection_name != value.collection_name
                    ):
                        raise RuntimeError("physical collection identity collision.")
                    return existing
                conflicting = self._connection.execute(
                    "SELECT collection_id FROM vector_collections WHERE profile_fingerprint=?",
                    (profile.fingerprint,),
                ).fetchone()
                if conflicting is not None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("embedding profile is already bound to another collection.")
                self._connection.execute(
                    """
                    INSERT INTO vector_collections (
                        collection_id, collection_name, profile_alias, profile_fingerprint,
                        model_name, dimensions, state, created_at, retired_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, NULL)
                    """,
                    (
                        value.collection_id,
                        value.collection_name,
                        value.profile_alias,
                        value.profile_fingerprint,
                        value.model_name,
                        value.dimensions,
                        value.created_at,
                    ),
                )
                self._connection.execute("COMMIT")
                return value
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("physical vector collection registration failed.") from exc

    def _current_route(self, owner: str, doc_id: str) -> VectorRouteRevision | None:
        row = self._connection.execute(
            """
            SELECT r.owner_id, r.doc_id, r.revision, r.collection_id,
                   r.profile_fingerprint, r.generation_sequence, r.action,
                   r.operation_id, r.previous_revision, r.previous_collection_id,
                   r.created_at
            FROM vector_route_heads h
            JOIN vector_route_revisions r
              ON r.owner_id=h.owner_id AND r.doc_id=h.doc_id AND r.revision=h.revision
            WHERE h.owner_id=? AND h.doc_id=?
            """,
            (owner, doc_id),
        ).fetchone()
        return self._route(row)

    def current_route(self, owner_id: str, doc_id: str) -> VectorRouteRevision | None:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id", 200)
        with self._lock:
            try:
                return self._current_route(owner, document)
            except sqlite3.Error as exc:
                raise RuntimeError("vector route read failed.") from exc

    def bootstrap_route(
        self,
        *,
        owner_id: str,
        doc_id: str,
        collection_id: str,
        generation_sequence: int,
        now: float,
    ) -> VectorRouteRevision:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id", 200)
        collection_key = _digest(collection_id, "collection_id")
        generation = _positive_integer(generation_sequence, "generation_sequence")
        created = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if self._current_route(owner, document) is not None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("vector route already exists for this document.")
                collection = self._get_collection(collection_key)
                if collection is None or collection.state != "ready":
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("target physical vector collection is not ready.")
                route = VectorRouteRevision(
                    owner_id=owner,
                    doc_id=document,
                    revision=1,
                    collection_id=collection.collection_id,
                    profile_fingerprint=collection.profile_fingerprint,
                    generation_sequence=generation,
                    action="bootstrap",
                    operation_id=None,
                    previous_revision=None,
                    previous_collection_id=None,
                    created_at=created,
                )
                self._insert_route(route)
                self._connection.execute(
                    "INSERT INTO vector_route_heads(owner_id, doc_id, revision) VALUES (?, ?, 1)",
                    (owner, document),
                )
                self._connection.execute("COMMIT")
                return route
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("vector route bootstrap failed.") from exc

    def _insert_route(self, route: VectorRouteRevision) -> None:
        self._connection.execute(
            """
            INSERT INTO vector_route_revisions (
                owner_id, doc_id, revision, collection_id, profile_fingerprint,
                generation_sequence, action, operation_id, previous_revision,
                previous_collection_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                route.owner_id,
                route.doc_id,
                route.revision,
                route.collection_id,
                route.profile_fingerprint,
                route.generation_sequence,
                route.action,
                route.operation_id,
                route.previous_revision,
                route.previous_collection_id,
                route.created_at,
            ),
        )

    def transition_route(
        self,
        *,
        owner_id: str,
        doc_id: str,
        expected_revision: int,
        expected_collection_id: str,
        expected_profile_fingerprint: str,
        expected_generation_sequence: int,
        target_collection_id: str,
        target_generation_sequence: int,
        operation_id: str,
        action: str,
        now: float,
    ) -> VectorRouteRevision:
        if action not in {"switch", "rollback", "generation_advance"}:
            raise ValueError(
                "route transition action must be switch, rollback or generation_advance."
            )
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id", 200)
        revision = _positive_integer(expected_revision, "expected_revision")
        expected_collection = _digest(expected_collection_id, "expected_collection_id")
        expected_profile = _digest(expected_profile_fingerprint, "expected_profile_fingerprint")
        expected_generation = _positive_integer(
            expected_generation_sequence,
            "expected_generation_sequence",
        )
        target_collection = _digest(target_collection_id, "target_collection_id")
        target_generation = _positive_integer(
            target_generation_sequence,
            "target_generation_sequence",
        )
        operation = _digest(operation_id, "operation_id")
        created = _timestamp(now, "now")
        if target_generation <= expected_generation:
            raise ValueError("target_generation_sequence must advance monotonically.")
        same_collection = target_collection == expected_collection
        if action == "generation_advance" and not same_collection:
            raise ValueError("generation_advance must retain the physical collection.")
        if action in {"switch", "rollback"} and same_collection:
            raise ValueError(f"{action} requires a different physical collection.")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                current = self._current_route(owner, document)
                if current is None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("vector route is unavailable.")
                existing_operation = self._connection.execute(
                    """
                    SELECT owner_id, doc_id, revision, collection_id, profile_fingerprint,
                           generation_sequence, action, operation_id, previous_revision,
                           previous_collection_id, created_at
                    FROM vector_route_revisions
                    WHERE owner_id=? AND doc_id=? AND operation_id=?
                    """,
                    (owner, document, operation),
                ).fetchone()
                if existing_operation is not None:
                    recorded = self._route(existing_operation)
                    if (
                        recorded is None
                        or recorded.collection_id != target_collection
                        or recorded.generation_sequence != target_generation
                        or recorded.action != action
                    ):
                        self._connection.execute("ROLLBACK")
                        raise RuntimeError("route operation ID collision.")
                    if current.revision != recorded.revision or current != recorded:
                        self._connection.execute("ROLLBACK")
                        raise RuntimeError("route operation already completed and was superseded.")
                    self._connection.execute("COMMIT")
                    return recorded
                if (
                    current.revision != revision
                    or current.collection_id != expected_collection
                    or current.profile_fingerprint != expected_profile
                    or current.generation_sequence != expected_generation
                ):
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("vector route compare-and-swap precondition failed.")
                target = self._get_collection(target_collection)
                if target is None or target.state != "ready":
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("target physical vector collection is not ready.")
                if action == "generation_advance" and target.profile_fingerprint != expected_profile:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("generation advance may not change the embedding profile.")
                route = VectorRouteRevision(
                    owner_id=owner,
                    doc_id=document,
                    revision=revision + 1,
                    collection_id=target.collection_id,
                    profile_fingerprint=target.profile_fingerprint,
                    generation_sequence=target_generation,
                    action=action,
                    operation_id=operation,
                    previous_revision=revision,
                    previous_collection_id=current.collection_id,
                    created_at=created,
                )
                self._insert_route(route)
                cursor = self._connection.execute(
                    """
                    UPDATE vector_route_heads SET revision=?
                    WHERE owner_id=? AND doc_id=? AND revision=?
                    """,
                    (route.revision, owner, document, revision),
                )
                if cursor.rowcount != 1:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("vector route compare-and-swap lost the head race.")
                self._connection.execute("COMMIT")
                return route
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("vector route transition failed.") from exc

    def route_history(
        self,
        owner_id: str,
        doc_id: str,
        *,
        limit: int = 100,
    ) -> tuple[VectorRouteRevision, ...]:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id", 200)
        count = _positive_integer(limit, "limit", _MAX_HISTORY)
        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT owner_id, doc_id, revision, collection_id, profile_fingerprint,
                           generation_sequence, action, operation_id, previous_revision,
                           previous_collection_id, created_at
                    FROM vector_route_revisions
                    WHERE owner_id=? AND doc_id=?
                    ORDER BY revision DESC LIMIT ?
                    """,
                    (owner, document, count),
                ).fetchall()
                return tuple(self._route(row) for row in rows if row is not None)  # type: ignore[misc]
            except sqlite3.Error as exc:
                raise RuntimeError("vector route history read failed.") from exc

    def current_routes(
        self,
        owner_id: str,
        *,
        limit: int = 1_000,
    ) -> tuple[VectorRouteRevision, ...]:
        owner = normalize_owner_id(owner_id)
        count = _positive_integer(limit, "limit", _MAX_HISTORY)
        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT r.owner_id, r.doc_id, r.revision, r.collection_id,
                           r.profile_fingerprint, r.generation_sequence, r.action,
                           r.operation_id, r.previous_revision, r.previous_collection_id,
                           r.created_at
                    FROM vector_route_heads h
                    JOIN vector_route_revisions r
                      ON r.owner_id=h.owner_id AND r.doc_id=h.doc_id AND r.revision=h.revision
                    WHERE h.owner_id=? ORDER BY r.doc_id LIMIT ?
                    """,
                    (owner, count),
                ).fetchall()
                return tuple(self._route(row) for row in rows if row is not None)  # type: ignore[misc]
            except sqlite3.Error as exc:
                raise RuntimeError("current vector routes read failed.") from exc

    def retire_collection(
        self,
        collection_id: str,
        *,
        confirm_collection_id: str,
        now: float,
    ) -> PhysicalVectorCollection:
        selected = _digest(collection_id, "collection_id")
        confirmation = _digest(confirm_collection_id, "confirm_collection_id")
        if confirmation != selected:
            raise ValueError("retirement confirmation does not match collection_id.")
        retired_at = _timestamp(now, "now")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                collection = self._get_collection(selected)
                if collection is None:
                    self._connection.execute("ROLLBACK")
                    raise KeyError(selected)
                if collection.state == "retired":
                    self._connection.execute("COMMIT")
                    return collection
                referenced = self._connection.execute(
                    """
                    SELECT 1
                    FROM vector_route_heads h
                    JOIN vector_route_revisions r
                      ON r.owner_id=h.owner_id AND r.doc_id=h.doc_id AND r.revision=h.revision
                    WHERE r.collection_id=? LIMIT 1
                    """,
                    (selected,),
                ).fetchone()
                if referenced is not None:
                    self._connection.execute("ROLLBACK")
                    raise RuntimeError("current vector routes still reference this collection.")
                self._connection.execute(
                    """
                    UPDATE vector_collections SET state='retired', retired_at=?
                    WHERE collection_id=? AND state='ready'
                    """,
                    (retired_at, selected),
                )
                self._connection.execute("COMMIT")
                result = self._get_collection(selected)
                if result is None:
                    raise RuntimeError("retired collection disappeared.")
                return result
            except (RuntimeError, KeyError):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("physical vector collection retirement failed.") from exc


class VectorCollectionRouter:
    """Resolve document queries through the current physical collection route."""

    def __init__(
        self,
        registry: VectorCollectionRegistry,
        layer_factory: Callable[[PhysicalVectorCollection], Any],
    ) -> None:
        if not isinstance(registry, VectorCollectionRegistry):
            raise ValueError("registry must be VectorCollectionRegistry.")
        if not callable(layer_factory):
            raise ValueError("layer_factory must be callable.")
        self.registry = registry
        self.layer_factory = layer_factory
        self._layers: dict[str, Any] = {}
        self._lock = threading.RLock()

    def resolve(
        self,
        owner_id: str,
        doc_id: str,
    ) -> tuple[VectorRouteRevision, PhysicalVectorCollection, Any]:
        route = self.registry.current_route(owner_id, doc_id)
        if route is None:
            raise KeyError("document has no current vector route.")
        collection = self.registry.get_collection(route.collection_id)
        if (
            collection is None
            or collection.state != "ready"
            or collection.profile_fingerprint != route.profile_fingerprint
        ):
            raise RuntimeError("current vector route references an unavailable collection.")
        with self._lock:
            layer = self._layers.get(collection.collection_id)
            if layer is None:
                layer = self.layer_factory(collection)
                if not callable(getattr(layer, "query", None)):
                    raise ValueError("layer_factory must return an object exposing query().")
                self._layers[collection.collection_id] = layer
        return route, collection, layer

    def query_document(
        self,
        query_text: str,
        *,
        owner_id: str,
        doc_id: str,
        n_results: int = 5,
        **kwargs: Any,
    ) -> Any:
        route, _, layer = self.resolve(owner_id, doc_id)
        return layer.query(
            query_text,
            n_results=n_results,
            owner_id=route.owner_id,
            doc_id=route.doc_id,
            **kwargs,
        )


__all__ = [
    "PhysicalVectorCollection",
    "VectorCollectionRegistry",
    "VectorCollectionRouter",
    "VectorRouteRevision",
]
