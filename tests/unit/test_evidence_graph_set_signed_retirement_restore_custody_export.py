from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_custody_export as export
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export_boundary as protected,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export_integrity as integrity,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    deterministic_signed_retirement_restore_id,
)


class RestoreStore:
    def __init__(self, value):
        self.value = value

    def get(self, restore_id):
        assert restore_id == self.value.restore_id
        return self.value


class Values:
    def __init__(self, values):
        self.values = tuple(values)

    def list(self, **kwargs):
        values = self.values
        if kwargs.get("state") is not None:
            values = tuple(value for value in values if value.state == kwargs["state"])
        return values[: kwargs["limit"]]


class Holds:
    def __init__(self, values=()):
        self.values = frozenset(values)

    def active_restore_ids(self, **kwargs):
        return self.values


def install_chain(monkeypatch, *, completed=True, custody_state="post_bound", artifacts=True):
    owner = "alice"
    snapshot_digest = "1" * 64
    target_digest = "2" * 64
    verification = "3" * 64
    restore_id = deterministic_signed_retirement_restore_id(
        owner_id=owner,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
    )
    snapshot = SimpleNamespace(
        owner_id=owner,
        snapshot_digest=snapshot_digest,
        record_count=2,
    )
    restore = SimpleNamespace(
        restore_id=restore_id,
        owner_id=owner,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
        snapshot_record_count=2,
        state="completed" if completed else "failed",
        phase="verified" if completed else "planned",
        target_verification_digest=verification if completed else None,
        completed_at=20.0 if completed else None,
    )
    pre = SimpleNamespace(
        owner_id=owner,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
        backup_sha256="4" * 64,
        backup_size_bytes=100,
        receipt_digest="5" * 64,
        actor_id="pre-actor",
        binding_method="process_environment",
        binding_digest="6" * 64,
    )
    post = SimpleNamespace(
        owner_id=owner,
        restore_id=restore_id,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
        pre_restore_receipt_digest=pre.receipt_digest,
        backup_sha256=pre.backup_sha256,
        target_verification_digest=verification,
        target_record_count=2,
        receipt_digest="7" * 64,
        actor_id="post-actor",
        binding_method="descriptor_file",
        binding_digest="8" * 64,
        compared_at=30.0,
    )
    custody = SimpleNamespace(
        custody_id="9" * 64,
        owner_id=owner,
        restore_id=restore_id,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
        pre_receipt_digest=pre.receipt_digest,
        backup_sha256=pre.backup_sha256,
        backup_size_bytes=pre.backup_size_bytes,
        pre_bound_actor_id=pre.actor_id,
        pre_bound_method=pre.binding_method,
        pre_bound_binding_digest=pre.binding_digest,
        pre_bound_at=10.0,
        post_receipt_digest=post.receipt_digest if custody_state == "post_bound" else None,
        target_verification_digest=(
            verification if custody_state == "post_bound" else None
        ),
        post_bound_actor_id=post.actor_id if custody_state == "post_bound" else None,
        post_bound_method=post.binding_method if custody_state == "post_bound" else None,
        post_bound_binding_digest=(
            post.binding_digest if custody_state == "post_bound" else None
        ),
        post_bound_at=post.compared_at if custody_state == "post_bound" else None,
        manifest_digest="a" * 64,
        state=custody_state,
    )
    artifact = SimpleNamespace(
        artifact_id="b" * 64,
        owner_id=owner,
        snapshot_digest=snapshot_digest,
        target_path_digest=target_digest,
        backup_path_digest="c" * 64,
        receipt_path_digest="d" * 64,
        state="completed",
        phase="verified",
        disposition="paired",
        backup_sha256=pre.backup_sha256,
        backup_size_bytes=pre.backup_size_bytes,
        receipt_digest=pre.receipt_digest,
        receipt_actor_id=pre.actor_id,
        receipt_binding_method=pre.binding_method,
        receipt_binding_digest=pre.binding_digest,
        completed_at=9.0,
    )

    monkeypatch.setattr(export, "verify_signed_retirement_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(export, "validate_terminal_snapshot", lambda _value: None)
    monkeypatch.setattr(export, "target_path_digest", lambda _path: target_digest)
    monkeypatch.setattr(
        export,
        "inspect_restored_target",
        lambda **kwargs: ("exact", verification),
    )
    monkeypatch.setattr(
        export,
        "verify_pre_restore_backup_receipt",
        lambda **kwargs: pre,
    )
    monkeypatch.setattr(
        export,
        "verify_post_restore_comparison_receipt",
        lambda _path: post,
    )
    monkeypatch.setattr(
        integrity,
        "artifact_path_digest",
        lambda _path, *, label: "c" * 64 if "backup" in label else "d" * 64,
    )
    return {
        "restore_id": restore_id,
        "restore_journal": RestoreStore(restore),
        "custody_store": Values((custody,)),
        "artifact_journal": Values((artifact,) if artifacts else ()),
        "hold_store": Holds((restore_id,)),
        "snapshot_path": "snapshot.json",
        "target_db_path": "target.sqlite3",
        "backup_path": "backup.sqlite3",
        "pre_receipt_path": "pre.json",
        "post_receipt_path": "post.json",
        "now": 40.0,
        "limit": 100,
    }


def test_complete_chain_is_deterministic_strict_and_privacy_reduced(monkeypatch):
    kwargs = install_chain(monkeypatch)
    first = export.build_restore_chain_of_custody(**kwargs)
    second = export.build_restore_chain_of_custody(**kwargs)

    assert first == second
    assert first.legal_hold_status == "active"
    assert len(first.artifacts) == 1
    assert first.restore_target_verification_digest == "3" * 64
    rendered = json.dumps(first.public_payload(), sort_keys=True)
    assert "pre-actor" not in rendered
    assert "post-actor" not in rendered
    assert "snapshot.json" not in rendered
    assert first.contains_raw_paths is False
    assert first.mutation_performed is False
    with pytest.raises(ValueError, match="chain_digest"):
        replace(first, chain_digest="0" * 64)


def test_incomplete_divergent_or_path_unbound_chain_refuses(monkeypatch):
    with pytest.raises(RuntimeError, match="completed exact"):
        export.build_restore_chain_of_custody(
            **install_chain(monkeypatch, completed=False)
        )
    with pytest.raises(RuntimeError, match="post-bound"):
        export.build_restore_chain_of_custody(
            **install_chain(monkeypatch, custody_state="pre_bound")
        )
    with pytest.raises(RuntimeError, match="completed artifact pair"):
        export.build_restore_chain_of_custody(
            **install_chain(monkeypatch, artifacts=False)
        )

    kwargs = install_chain(monkeypatch)
    monkeypatch.setattr(
        export,
        "inspect_restored_target",
        lambda **values: ("exact", "f" * 64),
    )
    with pytest.raises(RuntimeError, match="live restored target"):
        export.build_restore_chain_of_custody(**kwargs)

    kwargs = install_chain(monkeypatch)
    monkeypatch.setattr(
        integrity,
        "artifact_path_digest",
        lambda _path, *, label: "e" * 64,
    )
    with pytest.raises(RuntimeError, match="live paths"):
        export.build_restore_chain_of_custody(**kwargs)


def test_manifest_export_is_atomic_no_overwrite_and_detects_tampering(
    tmp_path,
    monkeypatch,
):
    kwargs = install_chain(monkeypatch)
    output = tmp_path / "chain.json"
    manifest = export.export_restore_chain_of_custody(
        output_path=output,
        **kwargs,
    )
    verified = export.verify_restore_chain_of_custody(output)
    assert verified == manifest
    assert output.read_bytes().endswith(b"\n")
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        export.export_restore_chain_of_custody(
            output_path=output,
            **kwargs,
        )

    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["generated_at"] = 41.0
    output.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="chain_digest"):
        export.verify_restore_chain_of_custody(output)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        export.verify_restore_chain_of_custody(duplicate)


def test_hmac_envelope_round_trip_wrong_key_tamper_and_permissions(
    tmp_path,
    monkeypatch,
):
    kwargs = install_chain(monkeypatch)
    manifest_path = tmp_path / "chain.json"
    export.export_restore_chain_of_custody(
        output_path=manifest_path,
        **kwargs,
    )
    key = tmp_path / "key.bin"
    key.write_bytes(b"k" * 32)
    if os.name != "nt":
        key.chmod(0o600)
    envelope_path = tmp_path / "chain.auth.json"
    envelope = protected.authenticate_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=envelope_path,
        key_id="custody-key-1",
        key_path=key,
    )
    verified = protected.verify_authenticated_restore_chain_of_custody(
        envelope_path=envelope_path,
        key_path=key,
        expected_key_id="custody-key-1",
    )
    assert verified == envelope
    assert verified.contains_key_material is False

    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"z" * 32)
    if os.name != "nt":
        wrong.chmod(0o600)
    with pytest.raises(PermissionError, match="verification"):
        protected.verify_authenticated_restore_chain_of_custody(
            envelope_path=envelope_path,
            key_path=wrong,
        )
    with pytest.raises(PermissionError, match="key ID"):
        protected.verify_authenticated_restore_chain_of_custody(
            envelope_path=envelope_path,
            key_path=key,
            expected_key_id="other-key",
        )

    raw = json.loads(envelope_path.read_text(encoding="utf-8"))
    raw["authentication_tag"] = "0" * 64
    envelope_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PermissionError, match="verification"):
        protected.verify_authenticated_restore_chain_of_custody(
            envelope_path=envelope_path,
            key_path=key,
        )

    weak = tmp_path / "weak.bin"
    weak.write_bytes(b"short")
    if os.name != "nt":
        weak.chmod(0o600)
    with pytest.raises(ValueError, match="at least 32"):
        protected.authenticate_restore_chain_of_custody(
            manifest_path=manifest_path,
            output_path=tmp_path / "weak.json",
            key_id="weak",
            key_path=weak,
        )

    if os.name != "nt":
        broad = tmp_path / "broad.bin"
        broad.write_bytes(b"b" * 32)
        broad.chmod(0o644)
        with pytest.raises(PermissionError, match="permissions"):
            protected.authenticate_restore_chain_of_custody(
                manifest_path=manifest_path,
                output_path=tmp_path / "broad.json",
                key_id="broad",
                key_path=broad,
            )
