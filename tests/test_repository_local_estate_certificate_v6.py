"""Current v41.1 estate certificate keeps immutable v4 historical authority."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "repository_local_estate_certificate_v6.py"


def load():
    spec = importlib.util.spec_from_file_location("estate_cert_v6_under_test", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_pins_exact_current_bootstrap_and_preserves_v4():
    v6 = load()
    assert v6.SCHEMA_VERSION == 6
    assert v6.CANONICAL_BOOTSTRAP_COMMIT == "e2acf14bb06c4d72a343024d6e3146a82756c1df"
    assert v6.CANONICAL_BOOTSTRAP_BLOB == "2f34cf0a1319c00d04bcfa99972f6e209534a096"
    assert v6.CANONICAL_OPF_COMMIT == "a361b49ac24ffc7de87440538c88994faed017c4"
    assert (ROOT / "tools" / "repository_local_estate_certificate.py").is_file()


def test_v6_accepts_current_pin_and_rejects_stale_release(monkeypatch, tmp_path):
    v6 = load()
    launcher = tmp_path / "run_all_training.py"
    def check(commit, blob):
        launcher.write_text(
            'scientific_authority strict_coverage '
            'require_literal_opf_mechanism_parity '
            'require_all_retained_trainable_source_reachability '
            + commit + " " + blob,
            encoding="utf-8",
        )
        monkeypatch.setattr(v6, "_remote_topology", lambda root: ("main", ["main"]))
        monkeypatch.setattr(v6, "_compile_retained_python", lambda root: (1, []))
        monkeypatch.setattr(v6, "_public_bytes", lambda url: b"pinned-source")
        # Exercise the status contract without external network dependencies.
        monkeypatch.setattr(v6, "_git_blob_sha", lambda data: v6.CANONICAL_BOOTSTRAP_BLOB)
        monkeypatch.setattr(sys, "argv", ["certificate", "--repository", "Anurag9000/Gram-Connect",
                                         "--root", str(tmp_path), "--output", "out.json"])
        rc = v6.main()
        return rc, json.loads((tmp_path / "out.json").read_text())
    ok, certificate = check(v6.CANONICAL_BOOTSTRAP_COMMIT, v6.CANONICAL_BOOTSTRAP_BLOB)
    assert ok == 0
    assert certificate["pass"] is True
    assert certificate["schema_version"] == 6
    assert certificate["canonical_v41_blob"] == v6.CANONICAL_BOOTSTRAP_BLOB
    stale, certificate = check("fd34a95d18892df7fb14d1efbb99076a7810fb91",
                                "05ef472b29933f18e956c69dfb7e543921ddaff5")
    assert stale == 2
    assert not certificate["pass"]
    assert any("canonical v41" in error for error in certificate["errors"])


def test_historical_v5_certificate_remains_byte_unchanged():
    import hashlib
    path = ROOT / "tools" / "repository_local_estate_certificate_v5.py"
    data = path.read_bytes()
    observed = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert observed == "a998473b2683dfec0f3b7cb6d097d53a539b7a0a"
