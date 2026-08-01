import json

import pytest

from tools.migration_rollback_artifact import (
    RollbackEncryptionKey,
    capture_rollback_payload,
)
from tools.migration_rollback_runtime import (
    clear_migration_rollback_store_cache,
    get_migration_rollback_store,
)
from tools.migration_rollback_store import MigrationRollbackStore
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def key(name="key-1", byte=b"k"):
    return RollbackEncryptionKey(name, byte * 32)


def test_encrypts_round_trips_and_does_not_store_plaintext(tmp_path):
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    manifest = store.write(
        preflight=preflight,
        payload=payload,
        key=key(),
        now=1,
    )
    directory = store.root / preflight.task_id / preflight.preflight_digest
    ciphertext = (directory / "ciphertext.bin").read_bytes()
    assert b'"document":"one"' not in ciphertext
    raw_manifest = (directory / "manifest.json").read_text()
    assert "one" not in raw_manifest
    assert "source_path" not in raw_manifest
    loaded, verified = store.load(preflight=preflight, key=key())
    assert loaded == payload
    assert verified == manifest
    assert verified.algorithm == "AES-256-GCM"


def test_identical_capture_reuses_existing_random_nonce_artifact(tmp_path):
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    first = store.write(preflight=preflight, payload=payload, key=key(), now=1)
    second = store.write(preflight=preflight, payload=payload, key=key(), now=2)
    assert second == first
    assert second.artifact_digest == first.artifact_digest


def test_wrong_key_id_key_material_and_ciphertext_tamper_fail_closed(tmp_path):
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    store.write(preflight=preflight, payload=payload, key=key(), now=1)
    with pytest.raises(RuntimeError, match="key ID"):
        store.load(preflight=preflight, key=key("key-2"))
    with pytest.raises(RuntimeError, match="authentication"):
        store.load(preflight=preflight, key=key("key-1", b"z"))
    path = store.root / preflight.task_id / preflight.preflight_digest / "ciphertext.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="ciphertext digest"):
        store.load(preflight=preflight, key=key())


def test_manifest_tamper_and_key_id_tamper_are_detected(tmp_path):
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    store.write(preflight=preflight, payload=payload, key=key(), now=1)
    path = store.root / preflight.task_id / preflight.preflight_digest / "manifest.json"
    raw = json.loads(path.read_text())
    raw["doc_id"] = "other"
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="manifest does not match"):
        store.load(preflight=preflight, key=key())


def test_remove_runtime_cache_and_root_identity_defenses(tmp_path):
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    store.write(preflight=preflight, payload=payload, key=key(), now=1)
    assert store.remove(preflight.task_id, preflight.preflight_digest) is True
    assert store.remove(preflight.task_id, preflight.preflight_digest) is False
    clear_migration_rollback_store_cache()
    one = get_migration_rollback_store(tmp_path / "one")
    again = get_migration_rollback_store(tmp_path / "one")
    two = get_migration_rollback_store(tmp_path / "two")
    assert one is again and one is not two
    root = tmp_path / "identity"
    guarded = MigrationRollbackStore(root)
    root.rename(tmp_path / "old")
    root.mkdir()
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.read_manifest(preflight.task_id, preflight.preflight_digest)


def test_symlink_root_is_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="redirects"):
        MigrationRollbackStore(link)
