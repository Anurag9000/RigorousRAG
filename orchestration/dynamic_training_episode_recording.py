"""Authoritative runtime-to-training episode recording for dynamic RAG.

This module wraps the existing server-owned ``run_dynamic_rag`` loop; it does not reimplement
retrieval/generation/tool authority.  Transparent feature/policy proxies observe the exact
snapshot and scores already consumed by that loop, reproduce the same closed deterministic
selector for logging, and publish ``LegalDynamicRagEpisodeStep`` JSONL only after the underlying
runtime succeeds.

The runtime selector is deterministic argmax with a stable tie-break, so the behavior action
probability of the selected action is exactly 1.0.  Offline stochastic/exploratory logs remain
supported by the separate governed import/sidecar paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

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
    DynamicRuntimeSnapshot,
    DynamicVerificationProvider,
    InformationNeedQueryProvider,
    RetrievalQueryRelease,
    _choose_action,
    run_dynamic_rag,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import (
    LegalDynamicRagEpisodeStep,
    parse_authoritative_dynamic_step,
)
from training.dynamic_retrieval_policy import DynamicRetrievalAction, DynamicRetrievalFeatures, allowed_actions

_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, maximum: int = 10_000_000) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is invalid")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class DynamicTerminalUtilityProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def utility(self, result: DynamicRagRuntimeResult) -> float: ...


@dataclass(frozen=True)
class CanonicalRequestTrainingContextProvider:
    """Deterministic request/generated/evidence-fingerprint context for need selection."""

    request_text: str
    include_retrieval_scores: bool = True
    include_verification: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_text", _text(self.request_text, "request_text"))
        if not isinstance(self.include_retrieval_scores, bool) or not isinstance(self.include_verification, bool):
            raise ValueError("context inclusion flags must be boolean")

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-canonical-dynamic-training-context-provider/v1",
            "request_text_sha256": hashlib.sha256(self.request_text.encode("utf-8")).hexdigest(),
            "include_retrieval_scores": self.include_retrieval_scores,
            "include_verification": self.include_verification,
            "template": "request+generated+evidence_fingerprints+optional_verification",
        })

    def model_text(self, snapshot: DynamicRuntimeSnapshot) -> str:
        lines = ["Request:", self.request_text, "", "Generated:", snapshot.generated_text, "", "Evidence:"]
        for item in snapshot.evidence:
            record = f"{item.evidence_id}\tsha256={item.evidence_sha256}\tsource_group={item.source_group_sha256}"
            if self.include_retrieval_scores:
                record += f"\tscore={format(item.retrieval_score, '.17g')}"
            lines.append(record)
        if self.include_verification and snapshot.verification is not None:
            lines.extend([
                "",
                "Verification:",
                f"support={format(snapshot.verification.support_score, '.17g')}",
                f"contradiction={format(snapshot.verification.contradiction_score, '.17g')}",
                f"verifier_sha256={snapshot.verification.verifier_sha256}",
            ])
        result = "\n".join(lines)
        return _text(result, "dynamic training context")


@dataclass(frozen=True)
class RecordedDynamicEpisodeReceipt:
    episode_id: str
    request_sha256: str
    runtime_policy_sha256: str
    feature_provider_sha256: str
    policy_artifact_sha256: str
    policy_contract_sha256: str
    behavior_policy_sha256: str
    context_provider_sha256: str
    terminal_utility_provider_sha256: str | None
    runtime_provider_contract_sha256: str
    runtime_result_sha256: str
    output_path: str
    output_sha256: str
    record_count: int
    step_sequence_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        for name in (
            "request_sha256", "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
            "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
            "runtime_provider_contract_sha256", "runtime_result_sha256", "output_sha256",
            "step_sequence_sha256", "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.terminal_utility_provider_sha256 is not None:
            object.__setattr__(self, "terminal_utility_provider_sha256", _sha(self.terminal_utility_provider_sha256, "terminal_utility_provider_sha256"))
        output = safe_advanced_path(self.output_path, label="recorded dynamic episode output", must_exist=True, require_file=True)
        object.__setattr__(self, "output_path", str(output))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("recorded dynamic episode receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-recorded-dynamic-episode-receipt/v1",
            **{name: getattr(self, name) for name in (
                "episode_id", "request_sha256", "runtime_policy_sha256", "feature_provider_sha256",
                "policy_artifact_sha256", "policy_contract_sha256", "behavior_policy_sha256",
                "context_provider_sha256", "terminal_utility_provider_sha256",
                "runtime_provider_contract_sha256", "runtime_result_sha256", "output_path",
                "output_sha256", "record_count", "step_sequence_sha256",
            )},
        }


class _EpisodeBuffer:
    def __init__(self, *, episode_id: str, request_sha256: str, runtime_policy: DynamicRagRuntimePolicy, context_provider: BoundDynamicModelContextProvider, feature_sha: str, policy_artifact_sha: str, policy_contract_sha: str) -> None:
        self.episode_id = _identifier(episode_id, "episode_id")
        self.request_sha256 = _sha(request_sha256, "request_sha256")
        self.runtime_policy = runtime_policy
        self.context_provider = context_provider
        self.context_sha = _sha(getattr(context_provider, "contract_sha256", None), "context provider contract_sha256")
        self.feature_sha = _sha(feature_sha, "feature provider sha256")
        self.policy_artifact_sha = _sha(policy_artifact_sha, "policy artifact sha256")
        self.policy_contract_sha = _sha(policy_contract_sha, "policy contract sha256")
        self.pending: dict[str, tuple[DynamicRuntimeSnapshot, DynamicRetrievalFeatures, str]] = {}
        self.steps: list[LegalDynamicRagEpisodeStep] = []

    @property
    def behavior_policy_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-runtime-behavior-policy/v1",
            "runtime_policy_sha256": self.runtime_policy.policy_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha,
            "policy_contract_sha256": self.policy_contract_sha,
            "selection": "server_argmax_then_action_value_tiebreak",
            "selected_action_probability": 1.0,
        })

    def observe_features(self, snapshot: DynamicRuntimeSnapshot, features: DynamicRetrievalFeatures) -> None:
        key = snapshot.snapshot_sha256
        if key in self.pending:
            raise RuntimeError("dynamic runtime requested features twice for one snapshot")
        if snapshot.request_sha256 != self.request_sha256:
            raise RuntimeError("dynamic runtime snapshot request differs from recording request")
        context = self.context_provider.model_text(snapshot)
        self.pending[key] = (snapshot, features, _text(context, "recorded dynamic context"))

    def observe_scores(self, snapshot_sha256: str, scores: Mapping[DynamicRetrievalAction, float]) -> None:
        key = _sha(snapshot_sha256, "snapshot_sha256")
        pending = self.pending.pop(key, None)
        if pending is None:
            raise RuntimeError("dynamic policy scores were observed without matching features")
        snapshot, features, context = pending
        permitted = allowed_actions(snapshot.state, self.runtime_policy.budget, verification_enabled=True)
        action = _choose_action(scores, permitted)
        normalized_scores = {
            (raw_action if isinstance(raw_action, DynamicRetrievalAction) else DynamicRetrievalAction(raw_action)).value: _finite(raw_score, "action score")
            for raw_action, raw_score in scores.items()
        }
        score_sha = _digest({"schema": "rigorousrag-recorded-dynamic-action-scores/v1", "scores": dict(sorted(normalized_scores.items()))})
        metadata = {
            "request_sha256": self.request_sha256,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "runtime_policy_sha256": self.runtime_policy.policy_sha256,
            "feature_provider_sha256": self.feature_sha,
            "policy_artifact_sha256": self.policy_artifact_sha,
            "policy_contract_sha256": self.policy_contract_sha,
            "behavior_policy_sha256": self.behavior_policy_sha256,
            "context_provider_sha256": self.context_sha,
            "action_scores_sha256": score_sha,
            "behavior_selection": "server_argmax_then_action_value_tiebreak",
        }
        self.steps.append(LegalDynamicRagEpisodeStep(
            episode_id=self.episode_id,
            step_id=f"step-{snapshot.iteration:09d}",
            context=context,
            features=features.as_mapping(),
            action=action,
            realized_retrieval_gain=0.0,
            behavior_action_probability=1.0,
            advantage=None,
            need_spans=(),
            hidden_state_cache_key=None,
            terminal_utility=None,
            metadata=metadata,
            valid_actions=tuple(permitted),
            value_target=None,
        ))


class _RecordingFeatureProvider:
    def __init__(self, inner: DynamicFeatureProvider, buffer: _EpisodeBuffer) -> None:
        self.inner, self.buffer = inner, buffer

    @property
    def contract_sha256(self) -> str:
        return self.inner.contract_sha256

    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures:
        features = self.inner.features(snapshot)
        if not isinstance(features, DynamicRetrievalFeatures):
            raise ValueError("inner feature provider returned invalid features")
        self.buffer.observe_features(snapshot, features)
        return features


class _RecordingPolicyProvider:
    def __init__(self, inner: DynamicPolicyProvider, buffer: _EpisodeBuffer) -> None:
        self.inner, self.buffer = inner, buffer

    @property
    def artifact_sha256(self) -> str:
        return self.inner.artifact_sha256

    @property
    def contract_sha256(self) -> str:
        return self.inner.contract_sha256

    def action_scores(self, features: DynamicRetrievalFeatures, *, snapshot_sha256: str) -> Mapping[DynamicRetrievalAction, float]:
        scores = self.inner.action_scores(features, snapshot_sha256=snapshot_sha256)
        if not isinstance(scores, Mapping):
            raise ValueError("inner policy provider returned invalid scores")
        self.buffer.observe_scores(snapshot_sha256, scores)
        return scores


def _step_payload(step: LegalDynamicRagEpisodeStep) -> Mapping[str, Any]:
    return {
        "episode_id": step.episode_id,
        "step_id": step.step_id,
        "context": step.context,
        "features": dict(step.features),
        "action": step.action.value,
        "realized_retrieval_gain": step.realized_retrieval_gain,
        "behavior_action_probability": step.behavior_action_probability,
        "advantage": step.advantage,
        "need_spans": [asdict(span) for span in step.need_spans],
        "hidden_state_cache_key": step.hidden_state_cache_key,
        "terminal_utility": step.terminal_utility,
        "metadata": dict(step.metadata),
        "valid_actions": [action.value for action in step.valid_actions],
        "value_target": step.value_target,
    }


def _verify_episode_file(path: Path, *, episode_id: str, expected_sha256: str, expected_count: int, expected_step_sha256: str) -> None:
    if _stream_sha(path) != expected_sha256:
        raise ValueError("recorded dynamic episode bytes differ from receipt")
    count = 0; step_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError("recorded dynamic episode line exceeds byte safety bound")
            try:
                payload = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
                step = parse_authoritative_dynamic_step(payload)
            except Exception as exc:
                raise ValueError("recorded dynamic episode contains invalid authoritative JSON") from exc
            if step.episode_id != episode_id or step.step_id != f"step-{count:09d}":
                raise ValueError("recorded dynamic episode order/identity differs from receipt")
            step_digest.update(f"{step.episode_id}\n{step.step_id}\n".encode("utf-8"))
            count += 1
    if count != expected_count or step_digest.hexdigest() != expected_step_sha256:
        raise ValueError("recorded dynamic episode count/step sequence differs from receipt")


def run_recorded_dynamic_rag_episode(
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
    """Run the existing bounded runtime and atomically publish its training episode."""
    if not isinstance(runtime_policy, DynamicRagRuntimePolicy):
        raise ValueError("runtime_policy must be DynamicRagRuntimePolicy")
    feature_sha = _sha(getattr(feature_provider, "contract_sha256", None), "feature provider contract_sha256")
    policy_artifact_sha = _sha(getattr(policy_provider, "artifact_sha256", None), "policy artifact_sha256")
    policy_contract_sha = _sha(getattr(policy_provider, "contract_sha256", None), "policy contract_sha256")
    context_sha = _sha(getattr(context_provider, "contract_sha256", None), "context provider contract_sha256")
    terminal_sha = None if terminal_utility_provider is None else _sha(getattr(terminal_utility_provider, "contract_sha256", None), "terminal utility provider contract_sha256")
    buffer = _EpisodeBuffer(
        episode_id=episode_id,
        request_sha256=request_sha256,
        runtime_policy=runtime_policy,
        context_provider=context_provider,
        feature_sha=feature_sha,
        policy_artifact_sha=policy_artifact_sha,
        policy_contract_sha=policy_contract_sha,
    )
    result = run_dynamic_rag(
        request_sha256=request_sha256,
        runtime_policy=runtime_policy,
        feature_provider=_RecordingFeatureProvider(feature_provider, buffer),
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
    if len(buffer.steps) != result.iterations:
        raise RuntimeError("dynamic recording step count differs from runtime iterations")
    steps = list(buffer.steps)
    if terminal_utility_provider is not None:
        utility = _finite(terminal_utility_provider.utility(result), "terminal utility")
        last = steps[-1]
        metadata = dict(last.metadata); metadata["terminal_utility_provider_sha256"] = terminal_sha
        steps[-1] = replace(last, terminal_utility=utility, metadata=metadata)

    root = safe_advanced_path(output_dir, label="recorded dynamic episode output", must_exist=False)
    if root.exists():
        raise ValueError("recorded dynamic episode output must not already exist")
    parent = safe_advanced_path(root.parent, label="recorded dynamic episode parent", must_exist=True, require_directory=True)
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
                handle.write(encoded); output_digest.update(encoded)
                step_digest.update(f"{step.episode_id}\n{step.step_id}\n".encode("utf-8"))
            handle.flush(); os.fsync(handle.fileno())
        final_episode = root / "episode.jsonl"
        unsigned = {
            "schema": "rigorousrag-recorded-dynamic-episode-receipt/v1",
            "episode_id": buffer.episode_id,
            "request_sha256": _sha(request_sha256, "request_sha256"),
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
        receipt = RecordedDynamicEpisodeReceipt(**{key: value for key, value in {**unsigned, "receipt_sha256": _digest(unsigned)}.items() if key != "schema"})
        receipt_path = stage / "episode_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"episode.jsonl", "episode_receipt.json"}:
            raise RuntimeError("recorded dynamic episode directory is not closed")
        os.replace(stage, root); published = True
        verified = verify_recorded_dynamic_episode(root / "episode_receipt.json")
        if verified.receipt_sha256 != receipt.receipt_sha256:
            raise RuntimeError("recorded dynamic episode changed after publication")
        return result, verified
    except Exception:
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_recorded_dynamic_episode(path: str | Path) -> RecordedDynamicEpisodeReceipt:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("recorded dynamic episode receipt may not be a symlink")
    receipt_path = safe_advanced_path(raw_path, label="recorded dynamic episode receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / "episode_receipt.json":
        raise ValueError("recorded dynamic episode receipt must use canonical filename")
    if {item.name for item in root.iterdir()} != {"episode.jsonl", "episode_receipt.json"}:
        raise ValueError("recorded dynamic episode directory is not closed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("recorded dynamic episode directory contains a non-regular child")
    if receipt_path.stat().st_size <= 0 or receipt_path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("recorded dynamic episode receipt exceeds byte safety bound")
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("recorded dynamic episode receipt is not strict JSON") from exc
    expected = {
        "schema", "episode_id", "request_sha256", "runtime_policy_sha256", "feature_provider_sha256",
        "policy_artifact_sha256", "policy_contract_sha256", "behavior_policy_sha256",
        "context_provider_sha256", "terminal_utility_provider_sha256", "runtime_provider_contract_sha256",
        "runtime_result_sha256", "output_path", "output_sha256", "record_count", "step_sequence_sha256",
        "receipt_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected or raw.get("schema") != "rigorousrag-recorded-dynamic-episode-receipt/v1":
        raise ValueError("unsupported recorded dynamic episode receipt schema")
    receipt = RecordedDynamicEpisodeReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    output = safe_advanced_path(receipt.output_path, label="recorded dynamic episode JSONL", must_exist=True, require_file=True)
    if output != root / "episode.jsonl":
        raise ValueError("recorded dynamic episode output must be canonical root child")
    _verify_episode_file(
        output,
        episode_id=receipt.episode_id,
        expected_sha256=receipt.output_sha256,
        expected_count=receipt.record_count,
        expected_step_sha256=receipt.step_sequence_sha256,
    )
    return receipt


__all__ = [
    "CanonicalRequestTrainingContextProvider",
    "DynamicTerminalUtilityProvider",
    "RecordedDynamicEpisodeReceipt",
    "run_recorded_dynamic_rag_episode",
    "verify_recorded_dynamic_episode",
]
