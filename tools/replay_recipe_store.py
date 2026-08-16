"""Opt-in encrypted replay recipes for automatic research-result recomputation.

Raw queries are never written to this database. Trusted application bootstrap code may
inject a cipher implementing ``seal``/``open``; only ciphertext, key identifier and
content hashes are persisted. Without such a cipher RigorousRAG remains hash-only and
result recomputation must be supplied externally.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_QUERY_BYTES = 100_000
_MAX_CIPHERTEXT_BYTES = 1_000_000


class ReplayCipher(Protocol):
    @property
    def key_id(self) -> str: ...
    def seal(self, plaintext: bytes, *, aad: bytes) -> bytes: ...
    def open(self, ciphertext: bytes, *, aad: bytes) -> bytes: ...


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("replay recipe database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("replay recipe path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _text(value: str, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: str, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _aad(owner_id: str, result_id: str, query_sha256: str) -> bytes:
    return f"RigorousRAG/replay/v1\x1f{owner_id}\x1f{result_id}\x1f{query_sha256}".encode("utf-8")


@dataclass(frozen=True)
class ReplayRecipe:
    owner_id: str
    result_id: str
    query_sha256: str
    query: str
    model: str
    strategy: str
    key_id: str
    created_at: float


@dataclass(frozen=True)
class ReplayRecipeMetadata:
    """Non-secret recipe metadata safe to expose to the owning principal.

    The encrypted query and ciphertext digest deliberately do not appear here. This lets
    product/API surfaces communicate replay availability and privacy state without ever
    decrypting the stored query merely to render a status page.
    """

    owner_id: str
    result_id: str
    query_sha256: str
    model: str
    strategy: str
    key_id: str
    created_at: float


class EncryptedReplayRecipeStore:
    def __init__(self, path: str | Path, *, cipher: ReplayCipher) -> None:
        if cipher is None or not callable(getattr(cipher, "seal", None)) or not callable(getattr(cipher, "open", None)):
            raise TypeError("cipher must implement seal/open")
        key_id = getattr(cipher, "key_id", None)
        if not isinstance(key_id, str) or not key_id.strip() or len(key_id) > 256:
            raise ValueError("cipher key_id is invalid")
        self.path = _safe_path(path)
        self.cipher = cipher
        self.key_id = key_id.strip()
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS replay_recipes (
                    owner_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    query_sha256 CHAR(64) NOT NULL,
                    query_ciphertext BLOB NOT NULL,
                    model TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    ciphertext_sha256 CHAR(64) NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, result_id)
                );
                CREATE INDEX IF NOT EXISTS replay_recipes_owner_query_idx
                  ON replay_recipes(owner_id, query_sha256, created_at DESC);
                """
            )

    @staticmethod
    def _metadata_from_row(owner_id: str, row: sqlite3.Row) -> ReplayRecipeMetadata:
        return ReplayRecipeMetadata(
            owner_id=owner_id,
            result_id=_sha(str(row["result_id"]), "result_id"),
            query_sha256=_sha(str(row["query_sha256"]), "query_sha256"),
            model=str(row["model"]),
            strategy=str(row["strategy"]),
            key_id=str(row["key_id"]),
            created_at=float(row["created_at"]),
        )

    def put(
        self,
        owner_id: str,
        *,
        result_id: str,
        query_sha256: str,
        query: str,
        model: str,
        strategy: str,
    ) -> None:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")
        query_digest = _sha(query_sha256, "query_sha256")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        plaintext = query.encode("utf-8")
        if not plaintext or len(plaintext) > _MAX_QUERY_BYTES:
            raise ValueError("query exceeds the replay recipe limit")
        if hashlib.sha256(plaintext).hexdigest() != query_digest:
            raise ValueError("query_sha256 does not match query")
        model_value = model if isinstance(model, str) else ""
        strategy_value = _text(strategy, "strategy", 128)
        aad = _aad(owner, result, query_digest)
        ciphertext = self.cipher.seal(plaintext, aad=aad)
        if not isinstance(ciphertext, bytes) or not ciphertext or len(ciphertext) > _MAX_CIPHERTEXT_BYTES:
            raise RuntimeError("cipher returned invalid replay ciphertext")
        ciphertext_sha = hashlib.sha256(ciphertext).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT query_sha256,ciphertext_sha256,model,strategy,key_id FROM replay_recipes WHERE owner_id=? AND result_id=?",
                (owner, result),
            ).fetchone()
            if existing is not None:
                if str(existing["query_sha256"]) != query_digest or str(existing["model"]) != model_value or str(existing["strategy"]) != strategy_value:
                    raise RuntimeError("replay recipe identity collision")
                # Re-encryption can be nondeterministic; preserve the first ciphertext.
                return
            connection.execute(
                """INSERT INTO replay_recipes
                   (owner_id,result_id,query_sha256,query_ciphertext,model,strategy,key_id,ciphertext_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (owner, result, query_digest, sqlite3.Binary(ciphertext), model_value, strategy_value, self.key_id, ciphertext_sha, created_at),
            )

    def metadata(self, owner_id: str, result_id: str) -> ReplayRecipeMetadata:
        """Read owner-scoped replay metadata without decrypting the query."""

        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT result_id,query_sha256,model,strategy,key_id,created_at
                   FROM replay_recipes WHERE owner_id=? AND result_id=?""",
                (owner, result),
            ).fetchone()
        if row is None:
            raise KeyError(result)
        return self._metadata_from_row(owner, row)

    def list_metadata(self, owner_id: str, *, limit: int = 100) -> tuple[ReplayRecipeMetadata, ...]:
        """List recent recipes for one owner without decrypting any stored query."""

        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT result_id,query_sha256,model,strategy,key_id,created_at
                   FROM replay_recipes WHERE owner_id=?
                   ORDER BY created_at DESC,result_id LIMIT ?""",
                (owner, limit),
            ).fetchall()
        return tuple(self._metadata_from_row(owner, row) for row in rows)

    def get(self, owner_id: str, result_id: str) -> ReplayRecipe:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM replay_recipes WHERE owner_id=? AND result_id=?",
                (owner, result),
            ).fetchone()
        if row is None:
            raise KeyError(result)
        if str(row["key_id"]) != self.key_id:
            raise RuntimeError("replay recipe requires a different cipher key")
        ciphertext = bytes(row["query_ciphertext"])
        if hashlib.sha256(ciphertext).hexdigest() != str(row["ciphertext_sha256"]):
            raise RuntimeError("replay recipe ciphertext integrity check failed")
        query_digest = _sha(str(row["query_sha256"]), "query_sha256")
        plaintext = self.cipher.open(ciphertext, aad=_aad(owner, result, query_digest))
        if not isinstance(plaintext, bytes) or not plaintext or len(plaintext) > _MAX_QUERY_BYTES:
            raise RuntimeError("cipher returned invalid replay plaintext")
        if hashlib.sha256(plaintext).hexdigest() != query_digest:
            raise RuntimeError("replay plaintext no longer matches its query digest")
        try:
            query = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("replay plaintext is not UTF-8") from exc
        return ReplayRecipe(
            owner_id=owner,
            result_id=result,
            query_sha256=query_digest,
            query=query,
            model=str(row["model"]),
            strategy=str(row["strategy"]),
            key_id=str(row["key_id"]),
            created_at=float(row["created_at"]),
        )

    def delete(self, owner_id: str, result_id: str) -> bool:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM replay_recipes WHERE owner_id=? AND result_id=?",
                (owner, result),
            )
        return bool(cursor.rowcount)


__all__ = [
    "EncryptedReplayRecipeStore",
    "ReplayCipher",
    "ReplayRecipe",
    "ReplayRecipeMetadata",
]
