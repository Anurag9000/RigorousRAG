"""Production atomic entry for runtime-to-training dynamic episode recording.

The underlying recorder primitives live in ``dynamic_training_episode_recording``. This entry
preserves their exact observation semantics, binds structural feature fractions to the exact
runtime policy, retains the exact finite action-score map used by deterministic server selection,
writes the final-path receipt while staging, renames the closed directory, and only then verifies
the final publication.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from orchestration.dynamic_feature_authority import BoundDynamicModelContextProvider
from orchestration.dynamic_rag_runtime import (
    DynamicEvidenceAdmission,
    DynamicFeatureProvider,
    DynamicGenerationProvider,
    DynamicPolicyProvider,
    DynamicRagRuntimePolicy,
    DynamicRagRuntimeResult,
    DynamicRetrievalProvider,
    DynamicRetrievedEvidence,
    DynamicVerificationProvider,
    InformationNeedQueryProvider,
    RetrievalQueryRelease,
    run_dynamic_rag,
)
from orchestration.dynamic_training_episode_recording import (
    DynamicTerminalUtilityProvider,
    RecordedDynamicEpisodeReceipt,
    _EpisodeBuffer,
    _RecordingFeatureProvider,
    _RecordingPolicyProvider,
    _canonical,
    _finite,
    _sha,
    _step_payload,
)
from orchestration.runtime_bound_dynamic_features import RuntimeBoundDynamicFeatureProvider
from orchestration.strict_dynamic_training_episode_io import verify_recorded_dynamic_episode_strict
from training.advanced_path_authority import safe_advanced_path
from training.dynamic_retrieval_policy import DynamicRetrievalAction

_MAX_LINE_BYTES = 64 * 1024 * 1024


class _ScoreProvenEpisodeBuffer(_EpisodeBuffer):
    """Retain the exact score map consumed by the server selector for restart proof."""

    def observe_scores(
        self,
        snapshot_sha256: str,
        scores: Mapping[DynamicRetrievalAction, float],
    ) -> None:
        normalized = {
            (
                raw_action
                if isinstance(raw_action, DynamicRetrievalAction)
                else DynamicRetrievalAction(raw_action)
            ).value: _finite(raw_score, "action score")
            for raw_action, raw_score in scores.items()
        }
        super().observe_scores(snapshot_sha256, scores)
        if not self.steps:
            raise RuntimeError("recording buffer did not append a step after policy scores")
        last = self.steps[-1]
        metadata = dict(last.metadata)
        metadata["action_scores_json"] = json.dumps(
            dict(sorted(normalized.items())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self.steps[-1] = replace(last, metadata=metadata)


def run_authoritative_recorded_dynamic_rag_episode(
    *,
    episode_id: str,
    request_sha256: str,
    runtime_policy: DynamicRagRuntimePolicy,
    feature_provider: DynamicFeatureProvider,
    policy_provider: DynamicPolicyProvider,
    context_provider: BoundDynamicModelContextProvider,
    generation_provider: DynamicGenerationProvider,
    query_provider: InformationNeedQueryProvider,
    query_release: RetrievalQueryRelease,
    retrieval_provider: DynamicRetrievalProvider,
    evidence_admission: DynamicEvidenceAdmission,
    verification_provider: DynamicVerificationProvider,
    output_dir: str | Path,
    initial_evidence: Sequence[DynamicRetrievedEvidence] = (),
    terminal_utility_provider: DynamicTerminalUtilityProvider | None = None,
) -> tuple[DynamicRagRuntimeResult, RecordedDynamicEpisodeReceipt]:
    if not isinstance(runtime_policy, DynamicRagRuntimePolicy):
        raise ValueError("runtime_policy must be DynamicRagRuntimePolicy")
    selected_request = _sha(request_sha256, "request_sha256")
    context_request = getattr(context_provider, "request_sha256", None)
    if context_request is not None and _sha(context_request, "context provider request_sha256") != selected_request:
        raise ValueError("context-provider request identity differs from runtime request")
    bound_features = RuntimeBoundDynamicFeatureProvider(feature_provider, runtime_policy)
    feature_sha = bound_features.contract_sha256
    policy_artifact_sha = _sha(getattr(policy_provider, "artifact_sha256", None), "policy artifact_sha256")
    policy_contract_sha = _sha(getattr(policy_provider, "contract_sha256", None), "policy contract_sha256")
    context_sha = _sha(getattr(context_provider, "contract_sha256", None), "context provider contract_sha256")
    terminal_sha = None if terminal_utility_provider is None else _sha(getattr(terminal_utility_provider, "contract_sha256", None), "terminal utility provider contract_sha256")
    buffer = _ScoreProvenEpisodeBuffer(
        episode_id=episode_id,
        request_sha256=selected_request,
        runtime_policy=runtime_policy,
        context_provider=context_provider,
        feature_sha=feature_sha,
        policy_artifact_sha=policy_artifact_sha,
        policy_contract_sha=policy_contract_sha,
    )
    result = run_dynamic_rag(
        request_sha256=selected_request,
        runtime_policy=runtime_policy,
        feature_provider=_RecordingFeatureProvider(bound_features, buffer),
        policy_provider=_RecordingPolicyProvider(policy_provider, buffer),
        generation_provider=generation_provider,
        query_provider=query_provider,
        query_release=query_release,
        retrieval_provider=retrieval_provider,
        evidence_admission=evidence_admission,
        verification_provider=verification_provider,
        initial_evidence=initial_evidence,
    )
    if buffer.pending:
        raise RuntimeError("dynamic recording ended with unmatched feature observations")
    if len(buffer.steps) != result.iterations or not buffer.steps:
        raise RuntimeError("dynamic recording step count differs from runtime iterations")
    steps = list(buffer.steps)
    if terminal_utility_provider is not None:
        utility = _finite(terminal_utility_provider.utility(result), "terminal utility")
        last = steps[-1]
        metadata = dict(last.metadata)
        metadata["terminal_utility_provider_sha256"] = str(terminal_sha)
        steps[-1] = replace(last, terminal_utility=utility, metadata=metadata)

    root = safe_advanced_path(output_dir, label="authoritative recorded dynamic episode output", must_exist=False)
    if root.exists():
        raise ValueError("authoritative recorded dynamic episode output must not already exist")
    parent = safe_advanced_path(root.parent, label="authoritative recorded dynamic episode parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'episode'}-stage-", dir=parent))
    published = False
    try:
        episode_path = stage / "episode.jsonl"
        output_digest, step_digest = hashlib.sha256(), hashlib.sha256()
        with episode_path.open("xb") as handle:
            for step in steps:
                encoded = _canonical(_step_payload(step)) + b"\n"
                if len(encoded) > _MAX_LINE_BYTES:
                    raise ValueError("recorded dynamic episode row exceeds byte safety bound")
                handle.write(encoded)
                output_digest.update(encoded)
                step_digest.update(f"{step.episode_id}\n{step.step_id}\n".encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        final_episode = root / "episode.jsonl"
        unsigned = {
            "schema": "rigorousrag-recorded-dynamic-episode-receipt/v1",
            "episode_id": buffer.episode_id,
            "request_sha256": selected_request,
            "runtime_policy_sha256": runtime_policy.policy_sha256,
            "feature_provider_sha256": feature_sha,
            "policy_artifact_sha256": policy_artifact_sha,
            "policy_contract_sha256": policy_contract_sha,
            "behavior_policy_sha256": buffer.behavior_policy_sha256,
            "context_provider_sha256": context_sha,
            "terminal_utility_provider_sha256": terminal_sha,
            "runtime_provider_contract_sha256": result.provider_contract_sha256,
            "runtime_result_sha256": result.result_sha256,
            "output_path": str(final_episode),
            "output_sha256": output_digest.hexdigest(),
            "record_count": len(steps),
            "step_sequence_sha256": step_digest.hexdigest(),
        }
        receipt_payload = {**unsigned, "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest()}
        receipt_path = stage / "episode_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_canonical(receipt_payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"episode.jsonl", "episode_receipt.json"}:
            raise RuntimeError("authoritative recorded dynamic episode directory is not closed")
        os.replace(stage, root)
        published = True
        receipt = verify_recorded_dynamic_episode_strict(root / "episode_receipt.json")
        if receipt.runtime_result_sha256 != result.result_sha256:
            raise RuntimeError("recorded dynamic episode runtime result identity changed")
        return result, receipt
    except Exception:
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "run_authoritative_recorded_dynamic_rag_episode",
    "RecordedDynamicEpisodeReceipt",
    "verify_recorded_dynamic_episode_strict",
]
