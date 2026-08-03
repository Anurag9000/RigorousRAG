"""Query-only listing of signed signer-administration reservations."""

from __future__ import annotations

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _identifier,
    _integer,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_readonly import (
    ReadOnlyCustodySignerAdminUseStore,
)
from tools.security import normalize_owner_id

_STATES = frozenset({"reserved", "committed"})
_ACTIONS = frozenset({"register", "retire"})
_MAX_LIMIT = 10_000
_TABLE = "evidence_graph_restore_custody_signer_admin_uses"


class ReadOnlyCustodySignerAdminUseAuditStore(
    ReadOnlyCustodySignerAdminUseStore
):
    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ):
        owner = normalize_owner_id(owner_id)
        selected_state = None if state is None else _identifier(state, "state", 30)
        selected_action = None if action is None else _identifier(action, "action", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("signer admin-use state is unsupported.")
        if selected_action is not None and selected_action not in _ACTIONS:
            raise ValueError("signer admin-use action is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[object] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        if selected_action is not None:
            query += " AND action=?"
            parameters.append(selected_action)
        query += " ORDER BY reserved_at DESC, use_id DESC LIMIT ?"
        parameters.append(count)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._base_value(row) for row in rows)

    @staticmethod
    def _base_value(row):
        from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use import (
            CustodySignerAdminUseStore,
        )

        return CustodySignerAdminUseStore._value(row)


__all__ = ["ReadOnlyCustodySignerAdminUseAuditStore"]
