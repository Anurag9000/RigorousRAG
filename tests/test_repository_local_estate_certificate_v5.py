"""Current v38.1 estate certificate keeps immutable v4 historical authority."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "repository_local_estate_certificate_v5.py"


def load():
    spec = importlib.util.spec_from_file_location("estate_cert_v5_under_test", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_pins_exact_current_bootstrap_and_preserves_v4():
    v5 = load()
    assert v5.SCHEMA_VERSION == 5
    assert v5.CANONICAL_BOOTSTRAP_COMMIT == "4443fea72e77b78e43bd9851ccb6354a6a9558e7"
    assert v5.CANONICAL_BOOTSTRAP_BLOB == "a341f0039139a9e9984476c370123df92da896b3"
    assert v5.CANONICAL_OPF_COMMIT == "1d3dea055736b1261a621436626ce89d69242a14"
    assert (ROOT / "tools" / "repository_local_estate_certificate.py").is_file()


def test_v5_accepts_current_pin_and_rejects_stale_release(monkeypatch, tmp_path):
    v5 = load()
    launcher = tmp_path / "run_all_training.py"
    def check(commit, blob):
        launcher.write_text(
            'scientific_authority strict_coverage '
            'require_literal_opf_mechanism_parity '
            'require_all_retained_trainable_source_reachability '
            + commit + " " + blob,
            encoding="utf-8",
        )
        monkeypatch.setattr(v5, "_remote_topology", lambda root: ("main", ["main"]))
        monkeypatch.setattr(v5, "_compile_retained_python", lambda root: (1, []))
        monkeypatch.setattr(v5, "_public_bytes", lambda url: b"pinned-source")
        # Exercise the status contract without external network dependencies.
        monkeypatch.setattr(v5, "_git_blob_sha", lambda data: v5.CANONICAL_BOOTSTRAP_BLOB)
        monkeypatch.setattr(sys, "argv", ["certificate", "--repository", "Anurag9000/Gram-Connect",
                                         "--root", str(tmp_path), "--output", "out.json"])
        rc = v5.main()
        return rc, json.loads((tmp_path / "out.json").read_text())
    ok, certificate = check(v5.CANONICAL_BOOTSTRAP_COMMIT, v5.CANONICAL_BOOTSTRAP_BLOB)
    assert ok == 0
    assert certificate["pass"] is True
    assert certificate["schema_version"] == 5
    assert certificate["canonical_v38_blob"] == v5.CANONICAL_BOOTSTRAP_BLOB
    stale, certificate = check("fd34a95d18892df7fb14d1efbb99076a7810fb91",
                                "05ef472b29933f18e956c69dfb7e543921ddaff5")
    assert stale == 2
    assert not certificate["pass"]
    assert any("canonical v38" in error for error in certificate["errors"])
