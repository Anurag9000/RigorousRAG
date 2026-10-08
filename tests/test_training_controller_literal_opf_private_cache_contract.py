"""Truthful private-source boundary for literal OPF scheduler tests.

These tests do not assert the OPF scheduler passed when private bytes are
unavailable; a skip means unverified, not successful behavior certification.
"""
from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import pytest

from tests import test_training_controller_literal_opf_scheduler_behavior as literal


def test_uncredentialed_private_404_reports_unverified_skip(tmp_path, monkeypatch):
    monkeypatch.delenv("OPF_LITERAL_VERIFIED_CACHE", raising=False)
    monkeypatch.delenv("OPF_LITERAL_REQUIRE_PRIVATE", raising=False)
    monkeypatch.setattr(literal.base, "_prepare_opf_runtime", lambda _root: (
        (_ for _ in ()).throw(HTTPError("https://raw.githubusercontent.com/private", 404,
                                     "not accessible", None, None))
    ))
    with pytest.raises(pytest.skip.Exception, match="no scheduler behavior was verified"):
        literal._literal_scheduler(tmp_path)


def test_strict_mode_does_not_hide_private_access_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPF_LITERAL_VERIFIED_CACHE", raising=False)
    monkeypatch.setenv("OPF_LITERAL_REQUIRE_PRIVATE", "1")
    monkeypatch.setattr(literal.base, "_prepare_opf_runtime", lambda _root: (
        (_ for _ in ()).throw(HTTPError("https://raw.githubusercontent.com/private", 404,
                                     "not accessible", None, None))
    ))
    with pytest.raises(HTTPError) as exc:
        literal._literal_scheduler(tmp_path)
    assert exc.value.code == 404


def test_server_errors_are_failures_not_unverified_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("OPF_LITERAL_VERIFIED_CACHE", raising=False)
    monkeypatch.delenv("OPF_LITERAL_REQUIRE_PRIVATE", raising=False)
    monkeypatch.setattr(literal.base, "_prepare_opf_runtime", lambda _root: (
        (_ for _ in ()).throw(HTTPError("https://raw.githubusercontent.com/private", 500,
                                     "server error", None, None))
    ))
    with pytest.raises(HTTPError) as exc:
        literal._literal_scheduler(tmp_path)
    assert exc.value.code == 500


def test_explicit_private_cache_is_blob_verified_before_import(tmp_path, monkeypatch):
    relative = "utils/opf_massive_suite_runner.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    source = b"# pinned private scheduler fixture\n"
    path.write_bytes(source)
    monkeypatch.setenv("OPF_LITERAL_VERIFIED_CACHE", str(tmp_path))
    monkeypatch.setattr(literal.reference, "OPF_RUNTIME_BLOBS", {
        relative: literal.base._git_blob_sha(source),
    })
    assert literal._verified_private_cache() == tmp_path.resolve()
    path.write_bytes(b"# altered, not pinned\n")
    with pytest.raises(RuntimeError, match="blob mismatch"):
        literal._verified_private_cache()


def test_symlink_private_cache_file_is_rejected(tmp_path, monkeypatch):
    relative = "utils/opf_massive_suite_runner.py"
    real = tmp_path / "real.py"
    real.write_bytes(b"# real\n")
    symlink = tmp_path / relative
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(real)
    monkeypatch.setenv("OPF_LITERAL_VERIFIED_CACHE", str(tmp_path))
    monkeypatch.setattr(literal.reference, "OPF_RUNTIME_BLOBS", {
        relative: literal.base._git_blob_sha(real.read_bytes()),
    })
    with pytest.raises(RuntimeError, match="regular pinned file"):
        literal._verified_private_cache()
