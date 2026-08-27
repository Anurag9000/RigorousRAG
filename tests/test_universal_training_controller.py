from __future__ import annotations

import importlib.util
from pathlib import Path


CONTROLLER = Path(__file__).resolve().parents[1] / "tools" / "universal_training_controller.py"


def _load():
    spec = importlib.util.spec_from_file_location("training_controller_under_test", CONTROLLER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_opf_reference_is_immutable():
    module = _load()
    assert module.OPF_REFERENCE_REPOSITORY == "Anurag9000/OPF_ADP"
    assert module.OPF_REFERENCE_COMMIT == "a3c41f7c25f21977f1ff33e94a65b6450afabee9"
    assert module.OPF_RUNTIME_BLOBS["utils/opf_massive_suite_runner.py"] == "314dc390955e54c7ca35589e3008068155f9fb44"


def test_git_blob_verification():
    module = _load()
    data = b"hello"
    import hashlib
    expected = hashlib.sha1(b"blob 5\0hello").hexdigest()
    assert module._git_blob_sha(data) == expected


def test_explicit_catalog_covers_detected_trainer(tmp_path):
    module = _load()
    trainer = tmp_path / "train.py"
    trainer.write_text(
        "if __name__ == '__main__':\n"
        "    optimizer = object()\n",
        encoding="utf-8",
    )
    profile = {
        "repository": "owner/repo",
        "jobs": [{"id": "train", "entrypoint": "train.py"}],
        "ignore_entrypoints": ["run_all_training.py"],
        "strict_coverage": True,
    }
    jobs = module._job_records(tmp_path, profile)
    report = module._coverage_report(tmp_path, profile, jobs)
    assert report["coverage_ok"] is True
    assert report["uncovered_training_candidates"] == []


def test_uncovered_executable_trainer_fails_audit(tmp_path):
    module = _load()
    (tmp_path / "run_experiment.py").write_text(
        "if __name__ == '__main__':\n"
        "    optimizer = object()\n",
        encoding="utf-8",
    )
    profile = {"repository": "owner/repo", "jobs": [], "strict_coverage": True}
    jobs = module._job_records(tmp_path, profile)
    report = module._coverage_report(tmp_path, profile, jobs)
    assert report["coverage_ok"] is False
    assert "run_experiment.py" in report["uncovered_training_candidates"]
