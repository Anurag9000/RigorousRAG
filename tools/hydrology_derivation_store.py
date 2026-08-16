"""Immutable derivation recipes for deterministic hydrology artifact recomputation.

Recipes store identifiers and bounded JSON parameters only. They never contain source text.
Multiple recipes may legitimately produce the same output fingerprint; automatic replay is
therefore allowed only when the artifact has exactly one distinct recipe fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_ALLOWED_KINDS = frozenset({"retrieval_plan", "evidence_projection", "evidence_report"})
_MAX_JSON_BYTES = 2 * 1024 * 1024


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("hydrology derivation database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("hydrology derivation path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _canonical(value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("hydrology derivation recipe exceeds the JSON size limit")
    return encoded


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    copied = dict(value)
    _canonical(copied)
    return copied


@dataclass(frozen=True)
class HydrologyDerivationRecipe:
    owner_id: str
    project_id: str
    artifact_kind: str
    logical_id: str
    artifact_fingerprint: str
    inputs: Mapping[str, Any]
    parameters: Mapping[str, Any]
    created_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        kind = _text(self.artifact_kind, "artifact_kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported hydrology derivation artifact kind")
        object.__setattr__(self, "artifact_kind", kind)
        object.__setattr__(self, "logical_id", _text(self.logical_id, "logical_id", 500))
        object.__setattr__(self, "artifact_fingerprint", _digest(self.artifact_fingerprint, "artifact_fingerprint"))
        object.__setattr__(self, "inputs", _mapping(self.inputs, "inputs"))
        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        created = float(self.created_at or time.time())
        if not math.isfinite(created) or created < 0:
            raise ValueError("created_at is invalid")
        object.__setattr__(self, "created_at", created)

    @property
    def recipe_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "owner_id": self.owner_id,
                    "project_id": self.project_id,
                    "artifact_kind": self.artifact_kind,
                    "logical_id": self.logical_id,
                    "artifact_fingerprint": self.artifact_fingerprint,
                    "inputs": dict(self.inputs),
                    "parameters": dict(self.parameters),
                }
            )
        ).hexdigest()


def plan_recipe(
    owner_id: str,
    project_id: str,
    *,
    logical_id: str,
    artifact_fingerprint: str,
    topology_id: str,
    topology_fingerprint: str,
    package_id: str,
    package_fingerprint: str,
    spec: Mapping[str, Any],
    reach_travel_seconds: Mapping[str, float],
    limit: int,
) -> HydrologyDerivationRecipe:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
        raise ValueError("plan recipe limit is invalid")
    travel: dict[str, float] = {}
    if len(reach_travel_seconds) > 10_000:
        raise ValueError("plan recipe reach travel map exceeds the item limit")
    for key, raw in reach_travel_seconds.items():
        reach_id = _text(key, "reach_id", 256)
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("reach travel seconds must be finite and nonnegative")
        travel[reach_id] = value
    return HydrologyDerivationRecipe(
        owner_id,
        project_id,
        "retrieval_plan",
        logical_id,
        artifact_fingerprint,
        {
            "topology_id": _text(topology_id, "topology_id", 500),
            "topology_fingerprint": _digest(topology_fingerprint, "topology_fingerprint"),
            "package_id": _text(package_id, "package_id", 500),
            "package_fingerprint": _digest(package_fingerprint, "package_fingerprint"),
        },
        {"spec": dict(_mapping(spec, "spec")), "reach_travel_seconds": travel, "limit": limit},
    )


def projection_recipe(
    owner_id: str,
    project_id: str,
    *,
    logical_id: str,
    artifact_fingerprint: str,
    package_id: str,
    package_fingerprint: str,
    plan_id: str,
    plan_fingerprint: str,
) -> HydrologyDerivationRecipe:
    return HydrologyDerivationRecipe(
        owner_id,
        project_id,
        "evidence_projection",
        logical_id,
        artifact_fingerprint,
        {
            "package_id": _text(package_id, "package_id", 500),
            "package_fingerprint": _digest(package_fingerprint, "package_fingerprint"),
            "plan_id": _text(plan_id, "plan_id", 500),
            "plan_fingerprint": _digest(plan_fingerprint, "plan_fingerprint"),
        },
        {},
    )


def report_recipe(
    owner_id: str,
    project_id: str,
    *,
    logical_id: str,
    artifact_fingerprint: str,
    projection_id: str,
    projection_fingerprint: str,
    title: str,
    research_question: str,
) -> HydrologyDerivationRecipe:
    return HydrologyDerivationRecipe(
        owner_id,
        project_id,
        "evidence_report",
        logical_id,
        artifact_fingerprint,
        {
            "projection_id": _text(projection_id, "projection_id", 500),
            "projection_fingerprint": _digest(projection_fingerprint, "projection_fingerprint"),
        },
        {
            "title": _text(title, "title", 1000),
            "research_question": str(research_question)[:20_000],
        },
    )


class HydrologyDerivationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hydrology_derivation_recipes (
                    owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    logical_id TEXT NOT NULL,
                    artifact_fingerprint CHAR(64) NOT NULL,
                    recipe_sha256 CHAR(64) NOT NULL,
                    inputs_json TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, recipe_sha256)
                );
                CREATE INDEX IF NOT EXISTS hydrology_recipe_artifact_idx
                  ON hydrology_derivation_recipes(owner_id,artifact_kind,artifact_fingerprint,created_at,recipe_sha256);
                CREATE INDEX IF NOT EXISTS hydrology_recipe_project_idx
                  ON hydrology_derivation_recipes(owner_id,project_id,created_at DESC,recipe_sha256);
                """
            )

    def put(self, recipe: HydrologyDerivationRecipe) -> HydrologyDerivationRecipe:
        if not isinstance(recipe, HydrologyDerivationRecipe):
            raise TypeError("recipe must be HydrologyDerivationRecipe")
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO hydrology_derivation_recipes
                   (owner_id,project_id,artifact_kind,logical_id,artifact_fingerprint,recipe_sha256,inputs_json,parameters_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    recipe.owner_id,
                    recipe.project_id,
                    recipe.artifact_kind,
                    recipe.logical_id,
                    recipe.artifact_fingerprint,
                    recipe.recipe_sha256,
                    _canonical(recipe.inputs).decode("utf-8"),
                    _canonical(recipe.parameters).decode("utf-8"),
                    recipe.created_at,
                ),
            )
        return recipe

    def for_artifact(self, owner_id: str, artifact_kind: str, artifact_fingerprint: str) -> HydrologyDerivationRecipe:
        owner = normalize_owner_id(owner_id)
        kind = _text(artifact_kind, "artifact_kind", 64).lower()
        if kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported hydrology derivation artifact kind")
        fingerprint = _digest(artifact_fingerprint, "artifact_fingerprint")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM hydrology_derivation_recipes
                   WHERE owner_id=? AND artifact_kind=? AND artifact_fingerprint=?
                   ORDER BY created_at,recipe_sha256 LIMIT 3""",
                (owner, kind, fingerprint),
            ).fetchall()
        if not rows:
            raise KeyError(fingerprint)
        distinct = {str(row["recipe_sha256"]) for row in rows}
        if len(distinct) != 1:
            raise RuntimeError("hydrology artifact has multiple derivation recipes; automatic recompute is ambiguous")
        return self._from_row(rows[0])

    def list_project(self, owner_id: str, project_id: str, *, limit: int = 1000) -> tuple[HydrologyDerivationRecipe, ...]:
        owner = normalize_owner_id(owner_id)
        project = _text(project_id, "project_id", 256)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM hydrology_derivation_recipes WHERE owner_id=? AND project_id=?
                   ORDER BY created_at DESC,recipe_sha256 LIMIT ?""",
                (owner, project, limit),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> HydrologyDerivationRecipe:
        recipe = HydrologyDerivationRecipe(
            str(row["owner_id"]),
            str(row["project_id"]),
            str(row["artifact_kind"]),
            str(row["logical_id"]),
            str(row["artifact_fingerprint"]),
            json.loads(str(row["inputs_json"])),
            json.loads(str(row["parameters_json"])),
            float(row["created_at"]),
        )
        if recipe.recipe_sha256 != str(row["recipe_sha256"]):
            raise RuntimeError("stored hydrology derivation recipe failed integrity verification")
        return recipe


__all__ = [
    "HydrologyDerivationRecipe",
    "HydrologyDerivationStore",
    "plan_recipe",
    "projection_recipe",
    "report_recipe",
]
