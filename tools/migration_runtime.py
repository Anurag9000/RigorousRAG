"""Process-local factory for the durable migration journal."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.migration_journal import MigrationJournal

_LOCK = threading.RLock()
_JOURNALS: dict[str, MigrationJournal] = {}


def get_migration_journal(
    path: str | os.PathLike[str] | None = None,
) -> MigrationJournal:
    selected = path if path is not None else os.getenv(
        "INDEX_MIGRATION_DB_PATH",
        "data/index_migrations.sqlite3",
    )
    rendered = os.fspath(selected)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = MigrationJournal(key)
            _JOURNALS[key] = journal
        return journal


def clear_migration_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = ["clear_migration_journal_cache", "get_migration_journal"]
