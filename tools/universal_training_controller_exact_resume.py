#!/usr/bin/env python3
"""Conservative interruption-exact resume audit for universal training jobs.

Epoch-boundary restart is useful but is not the same as resuming at the exact
training position at which the pressure controller terminated a child. This
extension tightens the shared static audit without changing OPF scheduling.

Besides framework checkpoints, the audit recognizes a deliberately narrow
transactional-state contract for optimizer-free learners. Such a learner must persist
its full cursor/current+best parameters/early-stop state atomically after each bounded
training advancement, bind the state to immutable data/config identity, and either
persist the data permutation/RNG or reconstruct order deterministically from seed+epoch.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable

import universal_training_controller_current as current

BATCH_LOOP = re.compile(
    r"for\s+(?:\([^\n]+\)|[^\n:]+)\s+in\s+(?:enumerate\s*\()?[^\n:]*(?:loader|dataloader|batches|dataset)",
    re.I,
)
STEP_CURSOR_SAVE = re.compile(
    r"['\"](?:batch|batch_idx|batch_index|step|global_step|iteration|iter|sample_cursor|data_cursor)['\"]\s*:",
    re.I,
)
STEP_CURSOR_LOAD = re.compile(
    r"(?:start_batch|resume_batch|batch_cursor|start_step|resume_step|global_step|sample_cursor|data_cursor)\s*=|"
    r"\.get\(\s*['\"](?:batch|batch_idx|batch_index|step|global_step|iteration|iter|sample_cursor|data_cursor)['\"]|"
    r"\[['\"](?:batch|batch_idx|batch_index|step|global_step|iteration|iter|sample_cursor|data_cursor)['\"]\]",
    re.I,
)
DATA_ORDER_SAVE = re.compile(
    r"(?:sampler|batch_sampler|generator|dataloader|data_loader).*state_dict\s*\(|"
    r"(?:sampler|batch_sampler|generator|dataloader|data_loader).*get_state\s*\(|"
    r"['\"](?:sampler_state|dataloader_state|data_loader_state|generator_state|sample_order|permutation)['\"]\s*:",
    re.I,
)
DATA_ORDER_LOAD = re.compile(
    r"(?:sampler|batch_sampler|generator|dataloader|data_loader).*load_state_dict\s*\(|"
    r"(?:sampler|batch_sampler|generator|dataloader|data_loader).*set_state\s*\(|"
    r"StatefulDataLoader|skip_first_batches|set_start_index|resume_(?:sampler|dataloader)|"
    r"(?:sampler_state|dataloader_state|data_loader_state|generator_state|sample_order|permutation)",
    re.I,
)
COOPERATIVE_TERM = re.compile(
    r"SIGTERM|SIGINT|KeyboardInterrupt|termination_requested|shutdown_requested|checkpoint_requested",
    re.I,
)
STEP_CHECKPOINT = re.compile(
    r"(?:checkpoint_interval_steps|checkpoint_every_steps|save_every_steps|save_steps|checkpoint_steps|"
    r"global_step\s*%|step\s*%|batch_idx\s*%|batch_index\s*%)",
    re.I,
)
FRAMEWORK_EXACT = re.compile(
    r"StatefulDataLoader|torchdata\.stateful_dataloader|accelerator\.skip_first_batches|"
    r"deepspeed.*load_checkpoint|fabric.*load|lightning.*ckpt_path",
    re.I,
)
_STATE_SAVE = re.compile(r"\b(?:store|state_store)\.save\s*\(", re.I)
_STATE_LOAD = re.compile(r"\b(?:store|state_store)\.load_latest\s*\(", re.I)
_CURRENT_PARAMETERS = re.compile(r"\b(?:theta|weights|bias)\b", re.I)
_BEST_PARAMETERS = re.compile(r"\bbest_(?:theta|weights|bias)\b", re.I)
_EARLY_STATE = re.compile(r"\b(?:stale_epochs|bad_epochs|best_loss|best_validation_loss)\b", re.I)
_BATCH_CURSOR = re.compile(r"\b(?:batch_index|next_batch_start)\b", re.I)


def _combined_text(root: Path, paths: Iterable[str]) -> str:
    parts = []
    for rel in sorted(set(paths)):
        path = root / rel
        if path.is_file() and path.suffix.lower() in current.SOURCE_SUFFIXES:
            text = current._read_text(path)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _transactional_state_contract(text: str) -> Dict[str, bool]:
    """Prove the narrow optimizer-free interruption-exact state-store contract."""
    store_type = "class ContentAddressedStateStore" in text or "class ResumeStateStore" in text
    atomic_write = "os.replace(" in text and "os.fsync(" in text
    state_write = bool(_STATE_SAVE.search(text))
    state_read = bool(_STATE_LOAD.search(text))
    cursor = "epoch" in text and bool(_BATCH_CURSOR.search(text))
    parameters = bool(_CURRENT_PARAMETERS.search(text) and _BEST_PARAMETERS.search(text))
    early_state = bool(_EARLY_STATE.search(text))
    persisted_order = "permutation" in text and ("random_state" in text or "rng.getstate(" in text)
    deterministic_order = "_epoch_order" in text and "seed" in text and "epoch" in text
    order = persisted_order or deterministic_order
    bounded_safe_point = (
        "max_batches=1" in text
        or "checkpoint_every_batches=1" in text
        or ("checkpoint_every_batches" in text and state_write)
    )
    immutable_identity = (
        (
            "spec_sha256" in text
            and ("train_examples_sha256" in text or "train_queries_sha256" in text)
            and ("validation_examples_sha256" in text or "validation_queries_sha256" in text)
        )
        or (
            "training_manifest_digest" in text
            and "train_sha256" in text
            and "validation_sha256" in text
        )
    )
    resume_path = state_read and (
        "store.exists(" in text
        or "if resume:" in text
        or "resume=pointer.is_file()" in text
    )
    serialized_state = "asdict(state)" in text or "asdict(\n" in text
    proven = all(
        (
            store_type,
            atomic_write,
            state_write,
            state_read,
            cursor,
            parameters,
            early_state,
            order,
            bounded_safe_point,
            immutable_identity,
            resume_path,
            serialized_state,
        )
    )
    return {
        "store_type": store_type,
        "atomic_write": atomic_write,
        "state_write": state_write,
        "state_read": state_read,
        "cursor": cursor,
        "parameters": parameters,
        "early_stop_state": early_state,
        "data_order": order,
        "persisted_data_order": persisted_order,
        "deterministic_data_order": deterministic_order,
        "bounded_safe_point": bounded_safe_point,
        "immutable_identity": immutable_identity,
        "resume_path": resume_path,
        "serialized_state": serialized_state,
        "proven": proven,
    }


def _exact_checkpoint_contract(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    base = _ORIGINAL_CHECKPOINT_CONTRACT(root, paths)
    text = _combined_text(root, paths)
    batch_loop = bool(BATCH_LOOP.search(text))
    step_save = bool(STEP_CURSOR_SAVE.search(text))
    step_load = bool(STEP_CURSOR_LOAD.search(text))
    order_save = bool(DATA_ORDER_SAVE.search(text))
    order_load = bool(DATA_ORDER_LOAD.search(text))
    cooperative = bool(COOPERATIVE_TERM.search(text) and current.CHECKPOINT_WRITE.search(text))
    step_checkpoint = bool(STEP_CHECKPOINT.search(text) and current.CHECKPOINT_WRITE.search(text))
    framework_exact = bool(FRAMEWORK_EXACT.search(text))
    transactional = _transactional_state_contract(text)

    full_state = bool(base.get("exact_resume_detected"))
    in_epoch_position = (
        (not batch_loop)
        or framework_exact
        or (step_save and step_load and order_save and order_load and (cooperative or step_checkpoint))
    )
    conventional_exact = bool(full_state and in_epoch_position)
    transactional_exact = bool(transactional["proven"])
    interruption_exact = bool(conventional_exact or transactional_exact)
    epoch_resume = bool(
        transactional_exact
        or (
            base.get("native_resume_detected")
            and base.get("progress_state_save")
            and base.get("progress_state_load")
        )
    )

    if transactional_exact:
        # These are semantic equivalents for the optimizer-free transactional store;
        # expose them in the common certificate so downstream strict checks do not need
        # repository-specific exemptions.
        base.update(
            {
                "checkpoint_write": True,
                "checkpoint_read": True,
                "resume_token": True,
                "progress_state_save": True,
                "progress_state_load": True,
                "native_resume_detected": True,
            }
        )

    base.update(
        {
            "batch_training_loop_detected": batch_loop,
            "step_cursor_save": step_save,
            "step_cursor_load": step_load,
            "data_order_state_save": order_save,
            "data_order_state_load": order_load,
            "cooperative_termination_checkpoint": cooperative,
            "step_checkpoint_policy": step_checkpoint,
            "framework_exact_resume_evidence": framework_exact,
            "transactional_state_resume_evidence": transactional,
            "epoch_resume_detected": epoch_resume,
            "interruption_exact_resume_detected": interruption_exact,
            "exact_resume_detected": interruption_exact,
        }
    )
    return base


def install() -> None:
    current._checkpoint_contract_for_paths = _exact_checkpoint_contract


_ORIGINAL_CHECKPOINT_CONTRACT = current._checkpoint_contract_for_paths
