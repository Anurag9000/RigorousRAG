from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_exact_resume as exact_resume


def _base_contract(_root: Path, _paths) -> dict:
    return {
        "native_resume_detected": True,
        "progress_state_save": True,
        "progress_state_load": True,
        "exact_resume_detected": True,
    }


def _negative_base_contract(_root: Path, _paths) -> dict:
    return {
        "native_resume_detected": False,
        "progress_state_save": False,
        "progress_state_load": False,
        "exact_resume_detected": False,
    }


def _evaluate(tmp_path: Path, source: str, *, base=_base_contract) -> dict:
    trainer = tmp_path / "trainer.py"
    trainer.write_text(source, encoding="utf-8")
    original = exact_resume._ORIGINAL_CHECKPOINT_CONTRACT
    try:
        exact_resume._ORIGINAL_CHECKPOINT_CONTRACT = base
        return exact_resume._exact_checkpoint_contract(tmp_path, ["trainer.py"])
    finally:
        exact_resume._ORIGINAL_CHECKPOINT_CONTRACT = original


def test_epoch_only_checkpoint_is_not_interruption_exact_for_batch_loop(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
for batch_idx, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
checkpoint = {"epoch": epoch, "model": model.state_dict()}
torch.save(checkpoint, "last.pt")
start_epoch = torch.load("last.pt").get("epoch", 0)
""",
    )
    assert report["batch_training_loop_detected"] is True
    assert report["epoch_resume_detected"] is True
    assert report["interruption_exact_resume_detected"] is False
    assert report["exact_resume_detected"] is False


def test_batch_cursor_without_data_order_state_is_not_exact(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
for batch_idx, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
    if batch_idx % checkpoint_interval_steps == 0:
        torch.save({"batch_idx": batch_idx, "model": model.state_dict()}, "last.pt")
resume_batch = torch.load("last.pt").get("batch_idx", 0)
""",
    )
    assert report["step_cursor_save"] is True
    assert report["step_cursor_load"] is True
    assert report["data_order_state_save"] is False
    assert report["data_order_state_load"] is False
    assert report["exact_resume_detected"] is False


def test_full_batch_cursor_order_and_safe_checkpoint_is_interruption_exact(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
for batch_idx, batch in enumerate(train_loader):
    loss = model(batch)
    loss.backward()
    if batch_idx % checkpoint_interval_steps == 0:
        torch.save({
            "batch_idx": batch_idx,
            "sampler_state": sampler.state_dict(),
            "model": model.state_dict(),
        }, "last.pt")
state = torch.load("last.pt")
resume_batch = state.get("batch_idx", 0)
sampler.load_state_dict(state["sampler_state"])
""",
    )
    assert report["step_cursor_save"] is True
    assert report["step_cursor_load"] is True
    assert report["data_order_state_save"] is True
    assert report["data_order_state_load"] is True
    assert report["step_checkpoint_policy"] is True
    assert report["interruption_exact_resume_detected"] is True
    assert report["exact_resume_detected"] is True


def test_framework_exact_resume_evidence_allows_batch_loop(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
from torchdata.stateful_dataloader import StatefulDataLoader
loader = StatefulDataLoader(dataset)
for batch_idx, batch in enumerate(loader):
    loss = model(batch)
    loss.backward()
torch.save({"model": model.state_dict()}, "last.pt")
state = torch.load("last.pt")
""",
    )
    assert report["batch_training_loop_detected"] is True
    assert report["framework_exact_resume_evidence"] is True
    assert report["exact_resume_detected"] is True


def test_transactional_optimizer_free_state_store_is_exact(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
import os
from dataclasses import asdict
class ContentAddressedStateStore:
    def save(self, name, payload):
        handle.flush(); os.fsync(handle.fileno()); os.replace(tmp, destination)
    def load_latest(self, name):
        return payload

def _epoch_order(count, *, seed, epoch):
    return list(range(count))
state = FusionWeightTrainingState(epoch=0, batch_index=0, theta=(0.0,), best_theta=(0.0,), stale_epochs=0)
spec_sha256 = "x"; train_examples_sha256 = "y"; validation_examples_sha256 = "z"
store = ContentAddressedStateStore()
if store.exists("fusion"):
    state = store.load_latest("fusion")
while not state.completed:
    state = advance_training(spec, state, train, validation, max_batches=1)
    store.save("fusion", asdict(state))
""",
        base=_negative_base_contract,
    )
    proof = report["transactional_state_resume_evidence"]
    assert proof["proven"] is True
    assert proof["deterministic_data_order"] is True
    assert report["native_resume_detected"] is True
    assert report["exact_resume_detected"] is True


def test_transactional_state_store_without_atomic_write_is_not_exact(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        """
from dataclasses import asdict
class ResumeStateStore:
    def save(self, name, payload):
        destination.write_text(str(payload))
    def load_latest(self, name):
        return payload
state_store = ResumeStateStore()
training_manifest_digest = "x"; train_sha256 = "y"; validation_sha256 = "z"
epoch = 0; next_batch_start = 0; permutation = [0]; random_state = rng.getstate()
weights = [0.0]; best_weights = [0.0]; bad_epochs = 0
if resume:
    state = state_store.load_latest("plan")
state_store.save("plan", asdict(state))
checkpoint_every_batches=1
""",
        base=_negative_base_contract,
    )
    assert report["transactional_state_resume_evidence"]["atomic_write"] is False
    assert report["transactional_state_resume_evidence"]["proven"] is False
    assert report["exact_resume_detected"] is False
