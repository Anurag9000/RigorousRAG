"""Strict production verification for recorded dynamic-RAG training episodes.

The compatibility verifier proves receipt/file SHA, row parsing, order and count.  This layer also
proves that every row is *raw runtime logging* produced under the identities claimed by the
receipt.  In particular it rejects pre-filled reward/GAE/cache targets and verifies the exact
deterministic behavior-policy contract used by the server-owned runtime selector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from orchestration.dynamic_training_episode_recording import (
    RecordedDynamicEpisodeReceipt,
    verify_recorded_dynamic_episode,
)
from training.advanced_rag_authoritative_data import parse_authoritative_dynamic_step
from training.dynamic_retrieval_policy import DynamicRetrievalAction

_HEX = frozenset("0123456789abcdef")
_MAX_LINE_BYTES = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _behavior_sha(receipt: RecordedDynamicEpisodeReceipt) -> str:
    return _digest({
        "schema": "rigorousrag-dynamic-runtime-behavior-policy/v1",
        "runtime_policy_sha256": receipt.runtime_policy_sha256,
        "policy_artifact_sha256": receipt.policy_artifact_sha256,
        "policy_contract_sha256": receipt.policy_contract_sha256,
        "selection": "server_argmax_then_action_value_tiebreak",
        "selected_action_probability": 1.0,
    })


def verify_recorded_dynamic_episode_strict(path: str | Path) -> RecordedDynamicEpisodeReceipt:
    receipt = verify_recorded_dynamic_episode(path)
    if _behavior_sha(receipt) != receipt.behavior_policy_sha256:
        raise ValueError("recorded dynamic episode behavior-policy identity is not canonical")
    output = Path(receipt.output_path)
    count = 0
    with output.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError("recorded dynamic episode row exceeds byte safety bound")
            try:
                payload = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                step = parse_authoritative_dynamic_step(payload)
            except Exception as exc:
                raise ValueError(f"recorded dynamic episode row {line_number} is invalid") from exc
            metadata = step.metadata
            expected_metadata = {
                "request_sha256": receipt.request_sha256,
                "snapshot_sha256": None,
                "runtime_policy_sha256": receipt.runtime_policy_sha256,
                "feature_provider_sha256": receipt.feature_provider_sha256,
                "policy_artifact_sha256": receipt.policy_artifact_sha256,
                "policy_contract_sha256": receipt.policy_contract_sha256,
                "behavior_policy_sha256": receipt.behavior_policy_sha256,
                "context_provider_sha256": receipt.context_provider_sha256,
                "action_scores_sha256": None,
                "behavior_selection": "server_argmax_then_action_value_tiebreak",
            }
            for key, expected in expected_metadata.items():
                actual = metadata.get(key)
                if expected is None:
                    _sha(actual, f"recorded row {key}")
                elif actual != expected:
                    raise ValueError(f"recorded dynamic episode row {line_number} {key} differs from receipt")
            allowed_metadata = set(expected_metadata)
            is_last = count == receipt.record_count - 1
            terminal_marker = metadata.get("terminal_utility_provider_sha256")
            if receipt.terminal_utility_provider_sha256 is None:
                if terminal_marker is not None or step.terminal_utility is not None:
                    raise ValueError("recorded dynamic episode has terminal utility without provider authority")
            else:
                if is_last:
                    if terminal_marker != receipt.terminal_utility_provider_sha256 or step.terminal_utility is None:
                        raise ValueError("recorded dynamic episode final terminal utility lacks exact provider authority")
                    allowed_metadata.add("terminal_utility_provider_sha256")
                elif terminal_marker is not None or step.terminal_utility is not None:
                    raise ValueError("recorded dynamic episode terminal utility may appear only on final step")
            unknown_metadata = set(metadata) - allowed_metadata
            if unknown_metadata:
                raise ValueError(f"recorded dynamic episode row contains unsupported raw-runtime metadata: {sorted(unknown_metadata)}")
            if step.behavior_action_probability != 1.0:
                raise ValueError("server-deterministic recorded behavior probability must equal 1.0")
            if step.action not in step.valid_actions or not step.valid_actions:
                raise ValueError("recorded action must be legal under the exact logged action set")
            if step.realized_retrieval_gain != 0.0:
                raise ValueError("raw recorded runtime episode may not contain realized-gain supervision")
            if step.advantage is not None or step.value_target is not None:
                raise ValueError("raw recorded runtime episode may not contain GAE/value targets")
            if step.hidden_state_cache_key is not None:
                raise ValueError("raw recorded runtime episode may not contain hidden-cache keys")
            if step.need_spans:
                raise ValueError("raw recorded runtime episode may not contain information-need annotation spans")
            count += 1
    if count != receipt.record_count:
        raise ValueError("strict recorded dynamic episode count differs from receipt")
    return receipt


__all__ = ["verify_recorded_dynamic_episode_strict"]
