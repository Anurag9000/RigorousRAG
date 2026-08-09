import hashlib

import pytest

from evaluation.dataset_manifest import (
    DatasetAcquisitionManifest,
    DatasetFileManifest,
    verify_dataset_manifest,
)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest(payload: bytes, *, path="corpus.jsonl") -> DatasetAcquisitionManifest:
    return DatasetAcquisitionManifest(
        dataset_name="scifact",
        version="1.0",
        revision="beir-release-2024-01",
        source_uri="https://example.invalid/scifact-1.0.tar.gz",
        license_id="cc-by-nc-2.0",
        license_sha256=hashlib.sha256(b"license text").hexdigest(),
        files=(
            DatasetFileManifest(
                path=path,
                sha256=digest_bytes(payload),
                bytes=len(payload),
                records=2,
            ),
        ),
    )


def test_manifest_digest_is_deterministic_and_verification_binds_local_bytes(tmp_path):
    payload = b'{"id":1}\n{"id":2}\n'
    selected = manifest(payload)
    assert selected.manifest_digest == manifest(payload).manifest_digest
    (tmp_path / "corpus.jsonl").write_bytes(payload)
    report = verify_dataset_manifest(tmp_path, selected)
    assert report.verified is True
    assert report.verified_files == 1
    assert report.verified_bytes == len(payload)
    assert report.manifest_digest == selected.manifest_digest


def test_tampered_same_size_dataset_member_fails_digest_verification(tmp_path):
    payload = b"member-a"
    selected = manifest(payload)
    (tmp_path / "corpus.jsonl").write_bytes(b"member-b")
    assert len(payload) == len(b"member-b")
    with pytest.raises(RuntimeError, match="digest"):
        verify_dataset_manifest(tmp_path, selected)


def test_manifest_rejects_unknown_dataset_and_unsafe_paths():
    payload = b"x"
    with pytest.raises(KeyError):
        DatasetAcquisitionManifest(
            dataset_name="not-in-registry",
            version="1",
            revision="r1",
            source_uri="source",
            license_id="license",
            license_sha256=digest_bytes(b"license"),
            files=(DatasetFileManifest("file", digest_bytes(payload), 1),),
        )
    with pytest.raises(ValueError, match="safe relative"):
        DatasetFileManifest("../escape.json", digest_bytes(payload), 1)


def test_verification_refuses_symlinked_members(tmp_path):
    payload = b"safe"
    target = tmp_path / "target.jsonl"
    target.write_bytes(payload)
    link = tmp_path / "corpus.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(RuntimeError, match="links or reparse"):
        verify_dataset_manifest(tmp_path, manifest(payload))


def test_manifest_fingerprint_changes_with_revision_or_license():
    payload = b"x"
    first = manifest(payload)
    changed_revision = DatasetAcquisitionManifest(
        **{**first.__dict__, "revision": "beir-release-2024-02"}
    )
    changed_license = DatasetAcquisitionManifest(
        **{**first.__dict__, "license_sha256": digest_bytes(b"changed license")}
    )
    assert len(
        {first.manifest_digest, changed_revision.manifest_digest, changed_license.manifest_digest}
    ) == 3
