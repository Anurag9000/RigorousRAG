from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller as base
import universal_training_controller_current as current
import universal_training_controller_opf_mechanism_audit as mechanism
import universal_training_controller_opf_reference_v2 as opf_reference


def _install_reference() -> None:
    opf_reference.install()


def test_mechanism_certificate_is_offline_when_cache_absent(monkeypatch, tmp_path: Path) -> None:
    _install_reference()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("coverage audit attempted to materialize the private OPF runtime")

    monkeypatch.setattr(base, "_prepare_opf_runtime", forbidden)
    certificate = mechanism._mechanism_certificate(tmp_path)

    assert certificate["pass"] is True
    assert certificate["reference_synchronized"] is True
    assert certificate["runtime_validation_status"] == "deferred_to_execution"
    assert certificate["runtime_validation_deferred_to_execution"] is True
    assert certificate["execution_integrity_gate"]["pass"] is True
    assert certificate["only_job_catalog_builder_is_replaced"] is True


def test_partial_local_reference_cache_fails_closed(tmp_path: Path) -> None:
    _install_reference()
    cache = tmp_path / ".training_control" / "opf_reference" / current.OPF_REFERENCE_COMMIT
    relative, _expected = next(iter(current.OPF_RUNTIME_BLOBS.items()))
    poisoned = cache / relative
    poisoned.parent.mkdir(parents=True, exist_ok=True)
    poisoned.write_text("not the pinned OPF blob\n", encoding="utf-8")

    certificate = mechanism._mechanism_certificate(tmp_path)

    assert certificate["pass"] is False
    assert certificate["runtime_validation_status"] == "invalid_local_cache"
    assert certificate["runtime_validation_deferred_to_execution"] is False
    assert certificate["runtime_blob_errors"]


def test_complete_local_cache_is_hash_verified_before_optional_scheduler_introspection(tmp_path: Path) -> None:
    _install_reference()
    cache = tmp_path / ".training_control" / "opf_reference" / current.OPF_REFERENCE_COMMIT
    # Deliberately create every expected path with invalid bytes.  The certificate
    # must fail on the first integrity layer and must not treat file presence as
    # equivalent to a verified literal runtime.
    for relative in current.OPF_RUNTIME_BLOBS:
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invalid\n")

    certificate = mechanism._mechanism_certificate(tmp_path)
    assert certificate["pass"] is False
    assert certificate["runtime_validation_status"] == "invalid_local_cache"
    assert len(certificate["runtime_blob_errors"]) >= len(current.OPF_RUNTIME_BLOBS)
