"""PostgreSQL backend for immutable hydrology derivation recipes."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from tools.hydrology_derivation_store import HydrologyDerivationRecipe
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike

_ALLOWED_KINDS = frozenset({"retrieval_plan", "evidence_projection", "evidence_report"})


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _schema(value: str) -> str:
    cleaned = _text(value, "schema", 63)
    if not cleaned.replace("_", "").isalnum() or cleaned[0].isdigit():
        raise ValueError("schema must be a simple SQL identifier")
    return cleaned


def _digest(value: str, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _row(value: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    return value[key] if isinstance(value, Mapping) else value[index]


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise RuntimeError("database recipe JSON has unexpected type")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise RuntimeError("database recipe JSON must be an object")
    return parsed


class PostgresHydrologyDerivationStore:
    def __init__(self, connection_factory: ConnectionFactory, *, schema: str = "rigorousrag", initialize: bool = True) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connect = connection_factory
        self.schema = _schema(schema)
        if initialize:
            self.initialize()

    def _transaction(self, operation):
        connection = self._connect()
        cursor = connection.cursor()
        try:
            result = operation(cursor)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def initialize(self) -> None:
        schema = self.schema
        statements = (
            f"CREATE SCHEMA IF NOT EXISTS {schema}",
            f"""CREATE TABLE IF NOT EXISTS {schema}.hydrology_derivation_recipes (
                owner_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                artifact_kind TEXT NOT NULL,
                logical_id TEXT NOT NULL,
                artifact_fingerprint CHAR(64) NOT NULL,
                recipe_sha256 CHAR(64) NOT NULL,
                inputs JSONB NOT NULL,
                parameters JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(owner_id, recipe_sha256)
            )""",
            f"CREATE INDEX IF NOT EXISTS hydrology_recipe_artifact_idx ON {schema}.hydrology_derivation_recipes(owner_id,artifact_kind,artifact_fingerprint,created_at,recipe_sha256)",
            f"CREATE INDEX IF NOT EXISTS hydrology_recipe_project_idx ON {schema}.hydrology_derivation_recipes(owner_id,project_id,created_at DESC,recipe_sha256)",
        )

        def operation(cursor: CursorLike) -> None:
            for statement in statements:
                cursor.execute(statement)

        self._transaction(operation)

    def put(self, recipe: HydrologyDerivationRecipe) -> HydrologyDerivationRecipe:
        if not isinstance(recipe, HydrologyDerivationRecipe):
            raise TypeError("recipe must be HydrologyDerivationRecipe")
        payload_inputs = json.dumps(recipe.inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        payload_parameters = json.dumps(recipe.parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        schema = self.schema

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {schema}.hydrology_derivation_recipes
                    (owner_id,project_id,artifact_kind,logical_id,artifact_fingerprint,recipe_sha256,inputs,parameters,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                    ON CONFLICT(owner_id,recipe_sha256) DO NOTHING""",
                (
                    recipe.owner_id,
                    recipe.project_id,
                    recipe.artifact_kind,
                    recipe.logical_id,
                    recipe.artifact_fingerprint,
                    recipe.recipe_sha256,
                    payload_inputs,
                    payload_parameters,
                    recipe.created_at,
                ),
            )

        self._transaction(operation)
        return recipe

    def for_artifact(self, owner_id: str, artifact_kind: str, artifact_fingerprint: str) -> HydrologyDerivationRecipe:
        owner = normalize_owner_id(owner_id)
        kind = _text(artifact_kind, "artifact_kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported hydrology derivation artifact kind")
        fingerprint = _digest(artifact_fingerprint, "artifact_fingerprint")
        schema = self.schema

        def operation(cursor: CursorLike) -> HydrologyDerivationRecipe:
            cursor.execute(
                f"""SELECT owner_id,project_id,artifact_kind,logical_id,artifact_fingerprint,recipe_sha256,
                           inputs::text,parameters::text,created_at
                    FROM {schema}.hydrology_derivation_recipes
                    WHERE owner_id=%s AND artifact_kind=%s AND artifact_fingerprint=%s
                    ORDER BY created_at,recipe_sha256 LIMIT 3""",
                (owner, kind, fingerprint),
            )
            rows = cursor.fetchall()
            if not rows:
                raise KeyError(fingerprint)
            distinct = {str(_row(row, "recipe_sha256", 5)) for row in rows}
            if len(distinct) != 1:
                raise RuntimeError("hydrology artifact has multiple derivation recipes; automatic recompute is ambiguous")
            return self._from_row(rows[0])

        return self._transaction(operation)

    def list_project(self, owner_id: str, project_id: str, *, limit: int = 1000) -> tuple[HydrologyDerivationRecipe, ...]:
        owner = normalize_owner_id(owner_id)
        project = _text(project_id, "project_id", 256)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        schema = self.schema

        def operation(cursor: CursorLike) -> tuple[HydrologyDerivationRecipe, ...]:
            cursor.execute(
                f"""SELECT owner_id,project_id,artifact_kind,logical_id,artifact_fingerprint,recipe_sha256,
                           inputs::text,parameters::text,created_at
                    FROM {schema}.hydrology_derivation_recipes
                    WHERE owner_id=%s AND project_id=%s
                    ORDER BY created_at DESC,recipe_sha256 LIMIT %s""",
                (owner, project, limit),
            )
            return tuple(self._from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)

    @staticmethod
    def _from_row(row: Sequence[Any] | Mapping[str, Any]) -> HydrologyDerivationRecipe:
        recipe = HydrologyDerivationRecipe(
            str(_row(row, "owner_id", 0)),
            str(_row(row, "project_id", 1)),
            str(_row(row, "artifact_kind", 2)),
            str(_row(row, "logical_id", 3)),
            str(_row(row, "artifact_fingerprint", 4)),
            _mapping(_row(row, "inputs", 6)),
            _mapping(_row(row, "parameters", 7)),
            float(_row(row, "created_at", 8)),
        )
        if recipe.recipe_sha256 != str(_row(row, "recipe_sha256", 5)):
            raise RuntimeError("stored hydrology derivation recipe failed integrity verification")
        return recipe


__all__ = ["PostgresHydrologyDerivationStore"]
