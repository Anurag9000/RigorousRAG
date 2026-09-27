#!/usr/bin/env python3
"""Conservative interruption-exact resume audit for universal training jobs.

Epoch-boundary restart is useful but is not the same as resuming at the exact
training position at which the pressure controller terminated a child.  This
extension tightens the shared static audit without changing OPF scheduling.
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


def _combined_text(root: Path, paths: Iterable[str]) -> str:
    parts = []
    for rel in sorted(set(paths)):
        path = root / rel
        if path.is_file() and path.suffix.lower() in current.SOURCE_SUFFIXES:
            text = current._read_text(path)
            if text:
                parts.append(text)
    return "\n".join(parts)


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

    # Preserve the lower-level full-state checks from audit v5, but do not call
    # an epoch-only checkpoint interruption-exact when a batch training loop is
    # present. Exactness inside an epoch needs a cursor plus data-order recovery
    # and a safe point at which pressure termination can persist state.
    full_state = bool(base.get("exact_resume_detected"))
    in_epoch_position = (
        (not batch_loop)
        or framework_exact
        or (step_save and step_load and order_save and order_load and (cooperative or step_checkpoint))
    )
    interruption_exact = bool(full_state and in_epoch_position)
    epoch_resume = bool(base.get("native_resume_detected") and base.get("progress_state_save") and base.get("progress_state_load"))

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
            "epoch_resume_detected": epoch_resume,
            "interruption_exact_resume_detected": interruption_exact,
            # v6 defines exact resume as interruption-exact, not merely an
            # epoch-boundary restart with complete optimizer state.
            "exact_resume_detected": interruption_exact,
        }
    )
    return base


def install() -> None:
    current._checkpoint_contract_for_paths = _exact_checkpoint_contract


_ORIGINAL_CHECKPOINT_CONTRACT = current._checkpoint_contract_for_paths
