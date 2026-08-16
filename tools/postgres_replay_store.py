"""PostgreSQL storage for encrypted replay recipes.

Only ciphertext and non-secret metadata are persisted. Encryption/decryption remains in the
injected ``ReplayCipher`` and AAD continues to bind owner/result/query identities exactly as
in the SQLite reference implementation.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Sequence

from tools.postgres_research_stores import _PostgresMixin, _row
from tools.replay_recipe_store import (
    EncryptedReplayRecipeStore,
    ReplayCipher,
    ReplayRecipe,
    ReplayRecipeMetadata,
    _MAX_CIPHERTEXT_BYTES,
    _MAX_QUERY_BYTES,
    _aad,
    _sha,
    _text,
)
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike


class PostgresEncryptedReplayRecipeStore(_PostgresMixin, EncryptedReplayRecipeStore):
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        cipher: ReplayCipher,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        if cipher is None or not callable(getattr(cipher, "seal", None)) or not callable(
            getattr(cipher, "open", None)
        ):
            raise TypeError("cipher must implement seal/open")
        key_id = getattr(cipher, "key_id", None)
        if not isinstance(key_id, str) or not key_id.strip() or len(key_id) > 256:
            raise ValueError("cipher key_id is invalid")
        self.cipher = cipher
        self.key_id = key_id.strip()
        _PostgresMixin.__init__(
            self,
            connection_factory,
            schema=schema,
            initialize=initialize,
        )

    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.replay_recipes (
                    owner_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    query_sha256 CHAR(64) NOT NULL,
                    query_ciphertext BYTEA NOT NULL,
                    model TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    ciphertext_sha256 CHAR(64) NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,result_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS replay_recipes_owner_query_idx ON {schema}.replay_recipes(owner_id,query_sha256,created_at DESC)",
            )
        )

    @staticmethod
    def _metadata_from_row(
        owner_id: str,
        row: Sequence[Any] | Mapping[str, Any],
    ) -> ReplayRecipeMetadata:
        return ReplayRecipeMetadata(
            owner_id=owner_id,
            result_id=_sha(str(_row(row, "result_id", 0)), "result_id"),
            query_sha256=_sha(str(_row(row, "query_sha256", 1)), "query_sha256"),
            model=str(_row(row, "model", 2)),
            strategy=str(_row(row, "strategy", 3)),
            key_id=str(_row(row, "key_id", 4)),
            created_at=float(_row(row, "created_at", 5)),
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
        ciphertext = self.cipher.seal(plaintext, aad=_aad(owner, result, query_digest))
        if (
            not isinstance(ciphertext, bytes)
            or not ciphertext
            or len(ciphertext) > _MAX_CIPHERTEXT_BYTES
        ):
            raise RuntimeError("cipher returned invalid replay ciphertext")
        ciphertext_sha = hashlib.sha256(ciphertext).hexdigest()
        created_at = time.time()

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""SELECT query_sha256,model,strategy,key_id
                    FROM {self.schema}.replay_recipes
                    WHERE owner_id=%s AND result_id=%s FOR UPDATE""",
                (owner, result),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if (
                    str(_row(existing, "query_sha256", 0)) != query_digest
                    or str(_row(existing, "model", 1)) != model_value
                    or str(_row(existing, "strategy", 2)) != strategy_value
                ):
                    raise RuntimeError("replay recipe identity collision")
                return
            cursor.execute(
                f"""INSERT INTO {self.schema}.replay_recipes
                    (owner_id,result_id,query_sha256,query_ciphertext,model,strategy,key_id,ciphertext_sha256,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    owner,
                    result,
                    query_digest,
                    ciphertext,
                    model_value,
                    strategy_value,
                    self.key_id,
                    ciphertext_sha,
                    created_at,
                ),
            )

        self._transaction(operation)

    def metadata(self, owner_id: str, result_id: str) -> ReplayRecipeMetadata:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")

        def operation(cursor: CursorLike) -> ReplayRecipeMetadata:
            cursor.execute(
                f"""SELECT result_id,query_sha256,model,strategy,key_id,created_at
                    FROM {self.schema}.replay_recipes WHERE owner_id=%s AND result_id=%s""",
                (owner, result),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(result)
            return self._metadata_from_row(owner, row)

        return self._transaction(operation)

    def list_metadata(
        self,
        owner_id: str,
        *,
        limit: int = 100,
    ) -> tuple[ReplayRecipeMetadata, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[ReplayRecipeMetadata, ...]:
            cursor.execute(
                f"""SELECT result_id,query_sha256,model,strategy,key_id,created_at
                    FROM {self.schema}.replay_recipes WHERE owner_id=%s
                    ORDER BY created_at DESC,result_id LIMIT %s""",
                (owner, limit),
            )
            return tuple(self._metadata_from_row(owner, row) for row in cursor.fetchall())

        return self._transaction(operation)

    def get(self, owner_id: str, result_id: str) -> ReplayRecipe:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")

        def operation(cursor: CursorLike) -> ReplayRecipe:
            cursor.execute(
                f"""SELECT query_sha256,query_ciphertext,model,strategy,key_id,ciphertext_sha256,created_at
                    FROM {self.schema}.replay_recipes WHERE owner_id=%s AND result_id=%s""",
                (owner, result),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(result)
            key_id = str(_row(row, "key_id", 4))
            if key_id != self.key_id:
                raise RuntimeError("replay recipe requires a different cipher key")
            raw_ciphertext = _row(row, "query_ciphertext", 1)
            ciphertext = bytes(raw_ciphertext)
            expected_ciphertext_sha = str(_row(row, "ciphertext_sha256", 5))
            if hashlib.sha256(ciphertext).hexdigest() != expected_ciphertext_sha:
                raise RuntimeError("replay recipe ciphertext integrity check failed")
            query_digest = _sha(str(_row(row, "query_sha256", 0)), "query_sha256")
            plaintext = self.cipher.open(
                ciphertext,
                aad=_aad(owner, result, query_digest),
            )
            if (
                not isinstance(plaintext, bytes)
                or not plaintext
                or len(plaintext) > _MAX_QUERY_BYTES
            ):
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
                model=str(_row(row, "model", 2)),
                strategy=str(_row(row, "strategy", 3)),
                key_id=key_id,
                created_at=float(_row(row, "created_at", 6)),
            )

        return self._transaction(operation)

    def delete(self, owner_id: str, result_id: str) -> bool:
        owner = normalize_owner_id(owner_id)
        result = _sha(result_id, "result_id")

        def operation(cursor: CursorLike) -> bool:
            cursor.execute(
                f"DELETE FROM {self.schema}.replay_recipes WHERE owner_id=%s AND result_id=%s",
                (owner, result),
            )
            return bool(cursor.rowcount)

        return self._transaction(operation)


__all__ = ["PostgresEncryptedReplayRecipeStore"]
