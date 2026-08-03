"""Protected-key boundary for external restore custody authentication."""

from __future__ import annotations

import os
import stat

from tools import evidence_graph_set_signed_retirement_restore_custody_export as _base
from tools.evidence_graph_set_signed_retirement_snapshot import _path, _redirecting

_ORIGINAL_KEY_READER = getattr(
    _base,
    "_unprotected_hmac_key_reader",
    _base._read_hmac_key,
)
_base._unprotected_hmac_key_reader = _ORIGINAL_KEY_READER


def _read_hmac_key(path):
    selected = _path(path, label="hmac_key_path")
    info = selected.lstat()
    if _redirecting(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError("HMAC key must be a regular non-redirecting file.")
    if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("HMAC key permissions are too broad.")
    return _ORIGINAL_KEY_READER(selected)


_base._read_hmac_key = _read_hmac_key

AuthenticatedCustodyEnvelope = _base.AuthenticatedCustodyEnvelope
CustodyArtifactEvidence = _base.CustodyArtifactEvidence
RestoreChainOfCustodyManifest = _base.RestoreChainOfCustodyManifest
authenticate_restore_chain_of_custody = _base.authenticate_restore_chain_of_custody
build_restore_chain_of_custody = _base.build_restore_chain_of_custody
export_restore_chain_of_custody = _base.export_restore_chain_of_custody
verify_authenticated_restore_chain_of_custody = (
    _base.verify_authenticated_restore_chain_of_custody
)
verify_restore_chain_of_custody = _base.verify_restore_chain_of_custody


__all__ = [
    "AuthenticatedCustodyEnvelope",
    "CustodyArtifactEvidence",
    "RestoreChainOfCustodyManifest",
    "authenticate_restore_chain_of_custody",
    "build_restore_chain_of_custody",
    "export_restore_chain_of_custody",
    "verify_authenticated_restore_chain_of_custody",
    "verify_restore_chain_of_custody",
]
