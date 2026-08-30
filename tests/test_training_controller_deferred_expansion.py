from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_deferred as deferred
import universal_training_controller_deferred_v3 as deferred_v3


def _restart_job(job_id: str, *, depends_on=()) -> dict:
    return {
        "id": job_id,
        "command": ["python", f"{job_id}.py"],
        "phase": "features",
        "family": "dataset",
        "device_capable": False,
        "depends_on": list(depends_on),
        "resume_strategy": "restart_exact",
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": True,
        "checkpoint_contract": {"exact_resume": True},
    }


def test_producer_closure_is_transitive_and_order_preserving() -> None:
    records = [
        _restart_job("raw"),
        _restart_job("features", depends_on=("raw",)),
        _restart_job("domains", depends_on=("features",)),
        {"id": "train", "command": ["python", "train.py"], "phase": "training"},
    ]
    descriptors = [{"id": "fanout", "depends_on": ["domains"]}]
    assert deferred._producer_closure(records, descriptors) == ["raw", "features", "domains"]
    deferred_v3._strict_validate_producers(records, ["raw", "features", "domains"])


def test_training_job_cannot_be_deferred_enumeration_prerequisite() -> None:
    records = [
        {
            "id": "trainer",
            "command": ["python", "train.py"],
            "phase": "training",
            "resume_strategy": "exact_checkpoint",
            "checkpoint_contract": {"exact_resume": True},
        }
    ]
    with pytest.raises(SystemExit, match="training job cannot"):
        deferred_v3._strict_validate_producers(records, ["trainer"])


def test_restart_exact_producer_requires_atomic_idempotent_determinism() -> None:
    broken = _restart_job("producer")
    broken["atomic_outputs"] = False
    with pytest.raises(SystemExit, match="restart_exact producer lacks"):
        deferred_v3._strict_validate_producers([broken], ["producer"])


@pytest.mark.parametrize("strategy", ["exact_checkpoint", "framework_exact_checkpoint"])
def test_exact_producer_requires_explicit_exact_checkpoint_contract(strategy: str) -> None:
    producer = {
        "id": "producer",
        "command": ["python", "producer.py"],
        "phase": "features",
        "family": "dataset",
        "device_capable": False,
        "resume_strategy": strategy,
    }
    with pytest.raises(SystemExit, match="lacks explicit checkpoint_contract.exact_resume=true"):
        deferred_v3._strict_validate_producers([producer], ["producer"])

    producer["checkpoint_contract"] = {"exact_resume": True}
    deferred_v3._strict_validate_producers([producer], ["producer"])


def _write_expander(root: Path) -> None:
    path = root / "catalog.py"
    path.write_text(
        "def iter_jobs(artifact):\n"
        "    values=[line.strip() for line in open(artifact, encoding='utf-8') if line.strip()]\n"
        "    for value in values:\n"
        "        yield {\n"
        "            'id': f'train:{value}',\n"
        "            'command': ['python', 'train.py', '--domain', value],\n"
        "            'phase': 'training',\n"
        "            'resume_strategy': 'exact_checkpoint',\n"
        "            'checkpoint_contract': {'exact_resume': True},\n"
        "            'early_stopping': True,\n"
        "        }\n",
        encoding="utf-8",
    )


def _descriptor() -> dict:
    return {
        "id": "domains",
        "path": "catalog.py",
        "function": "iter_jobs",
        "args": ["domains.txt"],
        "depends_on": ["data:domains"],
        "artifact_inputs": ["domains.txt"],
    }


def test_materialization_freezes_artifact_source_descriptor_and_job_set(tmp_path: Path) -> None:
    _write_expander(tmp_path)
    (tmp_path / "domains.txt").write_text("a\nb\n", encoding="utf-8")
    records, rows = deferred._freeze_materializations(
        tmp_path,
        {"repository": "owner/repo"},
        [_descriptor()],
    )
    assert [record["id"] for record in records] == ["train:a", "train:b"]
    assert all("data:domains" in record["depends_on"] for record in records)
    state = json.loads((tmp_path / ".training_control" / deferred.STATE_NAME).read_text())
    assert state["expanders"][0]["generated_job_count"] == 2
    assert rows[0]["materialization_sha256"] == state["expanders"][0]["materialization_sha256"]

    # A different producer artifact implies a different experiment universe and
    # must not be silently mixed with the frozen downstream run.
    (tmp_path / "domains.txt").write_text("a\nc\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="materialization drifted"):
        deferred._freeze_materializations(
            tmp_path,
            {"repository": "owner/repo"},
            [_descriptor()],
        )


def test_generated_job_ids_and_commands_must_be_unique(tmp_path: Path) -> None:
    (tmp_path / "domains.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "catalog.py").write_text(
        "def iter_jobs():\n"
        "    yield {'id':'same','command':['python','train.py'],'phase':'training'}\n"
        "    yield {'id':'same','command':['python','other.py'],'phase':'training'}\n",
        encoding="utf-8",
    )
    descriptor = {
        "id": "bad",
        "path": "catalog.py",
        "function": "iter_jobs",
        "depends_on": [],
        "artifact_inputs": ["domains.txt"],
    }
    with pytest.raises(SystemExit, match="duplicate id"):
        deferred._materialize_one(tmp_path, descriptor)


def test_completed_producers_are_removed_from_remaining_dependencies() -> None:
    records = [
        _restart_job("raw"),
        _restart_job("domains", depends_on=("raw",)),
        {
            "id": "train:a",
            "command": ["python", "train.py", "a"],
            "phase": "training",
            "depends_on": ["domains"],
        },
        {
            "id": "aggregate",
            "command": ["python", "aggregate.py"],
            "phase": "preprocess",
            "depends_on": ["train:a", "domains"],
        },
    ]
    remaining = deferred._without_completed(records, ["raw", "domains"])
    by_id = {record["id"]: record for record in remaining}
    assert set(by_id) == {"train:a", "aggregate"}
    assert by_id["train:a"]["depends_on"] == []
    assert by_id["aggregate"]["depends_on"] == ["train:a"]
