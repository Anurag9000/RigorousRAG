"""Worker-local SQLite lookup adapters for production dynamic supervision materialization.

Historical provider classes open a connection per lookup, which is correct but expensive at
corpus scale.  These subclasses preserve the exact sidecar receipt/provider contract while
caching one immutable read-only SQLite connection per Python thread/process.  Construction still
performs the full restart verifier; first use in each worker re-hashes the sealed index once.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Mapping

from training.sqlite_dynamic_supervision_sidecars import (
    SqliteCounterfactualActionProvider,
    SqliteInformationNeedAnnotationProvider,
    SqliteLoggedValueProvider,
    SqliteRealizedRetrievalGainProvider,
)


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class _WorkerLocalLookupMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._local = threading.local()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_local", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            database = Path(self.database_path)
            if database.is_symlink() or not database.is_file():
                raise ValueError("dynamic supervision SQLite authority is missing or unsafe")
            if _stream_sha(database) != self.receipt.index_sha256:
                raise ValueError("dynamic supervision SQLite authority changed after verification")
            connection = sqlite3.connect(
                f"file:{database}?mode=ro&immutable=1",
                uri=True,
                timeout=30.0,
            )
            self._local.connection = connection
        return connection

    def _payload(self, episode_id: str, step_id: str) -> Mapping[str, Any]:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ValueError("step_id must be a non-empty string")
        row = self._connection().execute(
            "SELECT payload_json FROM entries WHERE episode_id=? AND step_id=?",
            (episode_id.strip(), step_id.strip()),
        ).fetchone()
        if row is None:
            raise ValueError(f"dynamic supervision sidecar lacks step {(episode_id.strip(), step_id.strip())!r}")
        try:
            value = json.loads(
                str(row[0]),
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except Exception as exc:
            raise ValueError("dynamic supervision indexed payload is not strict JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("dynamic supervision indexed payload is malformed")
        return value

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None


class WorkerLocalInformationNeedAnnotationProvider(
    _WorkerLocalLookupMixin,
    SqliteInformationNeedAnnotationProvider,
):
    pass


class WorkerLocalRealizedRetrievalGainProvider(
    _WorkerLocalLookupMixin,
    SqliteRealizedRetrievalGainProvider,
):
    pass


class WorkerLocalLoggedValueProvider(
    _WorkerLocalLookupMixin,
    SqliteLoggedValueProvider,
):
    pass


class WorkerLocalCounterfactualActionProvider(
    _WorkerLocalLookupMixin,
    SqliteCounterfactualActionProvider,
):
    pass


__all__ = [
    "WorkerLocalCounterfactualActionProvider",
    "WorkerLocalInformationNeedAnnotationProvider",
    "WorkerLocalLoggedValueProvider",
    "WorkerLocalRealizedRetrievalGainProvider",
]
