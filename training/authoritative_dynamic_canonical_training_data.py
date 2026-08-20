"""Atomic, episode-streamed canonical dynamic-RAG training-data authority v2.

The mathematical workflow is unchanged: deterministic hidden-key/need-span planning, governed
realized retrieval gain binding, episode-level value/GAE and legal counterfactual targets,
episode-isolated final dataset publication, then final-manifest-bound hidden-state caching.
Only one bounded episode is resident while GAE/value targets are computed. Corpus identity,
episode accounting and cache-key authority are SQLite-backed. The retained planned and
materialized JSONL artifacts make pre/post supervision lineage independently restart-verifiable,
and the complete publication is atomically renamed into place before a final strict read-back.
Importing this module executes no model, provider, download, retrieval or training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import (
    LegalDynamicRagEpisodeStep,
    parse_authoritative_dynamic_step,
)
from training.advanced_rag_data import TextSpan
from training.advanced_rag_supervision import (
    CounterfactualActionProvider,
    DynamicRewardConfig,
    SupervisionCacheIdentity,
    generalized_advantage_estimate,
    trajectory_rewards,
)
from training.disk_backed_supervision_cache import DiskBackedAuthoritativeSafetensorCache
from training.dynamic_canonical_training_data_pipeline import DynamicRuntimeTrainingLineage
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication, verify_dynamic_dataset_publication
from training.dynamic_dataset_publication import (
    DynamicDatasetGovernance,
    DynamicDatasetPublicationReceipt,
    DynamicTrajectorySource,
    EpisodeSplitPolicy,
    PublishedDynamicSplit,
    publish_dynamic_training_dataset,
)
from training.dynamic_record_identity import dynamic_hidden_cache_key, dynamic_step_identity
from training.dynamic_reward_supervision import RealizedRetrievalGainProvider
from training.dynamic_retrieval_policy import DynamicRetrievalAction
from training.dynamic_trajectory_materialization import LoggedValueProvider
from training.dynamic_trajectory_preparation import (
    BoundGeneratorHiddenStateProvider,
    InformationNeedAnnotationProvider,
)
from training.sqlite_identity_ledger import SqliteIdentityLedger

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

_HEX = frozenset("0123456789abcdef")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_EPISODE_STEPS = 1_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _commit(value: Any) -> str:
    selected = str(value).strip().lower()
    if len(selected) not in {40, 64} or any(ch not in _HEX for ch in selected):
        raise ValueError("source_commit must be a full Git object id")
    return selected


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


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _strict_step_line(raw: bytes, label: str) -> LegalDynamicRagEpisodeStep:
    if not raw.strip() or len(raw) > _MAX_LINE_BYTES:
        raise ValueError(f"{label} is empty or exceeds line safety bound")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    return parse_authoritative_dynamic_step(value)


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


def _normalized_hidden(encoded: Mapping[str, Any]) -> Mapping[str, Any]:
    if torch is None:
        raise RuntimeError("dynamic hidden-state cache materialization requires optional PyTorch")
    required = {"token_hidden", "state_hidden", "attention_mask"}
    if not isinstance(encoded, Mapping) or not required.issubset(encoded):
        raise ValueError("hidden provider must return token_hidden/state_hidden/attention_mask")
    token_hidden, state_hidden, attention = encoded["token_hidden"], encoded["state_hidden"], encoded["attention_mask"]
    if not all(torch.is_tensor(value) for value in (token_hidden, state_hidden, attention)):
        raise ValueError("hidden provider outputs must be tensors")
    token_hidden, state_hidden, attention = token_hidden.detach().cpu(), state_hidden.detach().cpu(), attention.detach().cpu()
    if token_hidden.ndim != 3 or token_hidden.size(0) != 1 or state_hidden.ndim != 2 or state_hidden.size(0) != 1 or attention.ndim != 2 or attention.size(0) != 1:
        raise ValueError("hidden provider must return batch-one [1,T,H]/[1,H]/[1,T]")
    if token_hidden.size(1) != attention.size(1) or token_hidden.size(2) != state_hidden.size(1):
        raise ValueError("hidden provider tensor shapes are inconsistent")
    if not bool(attention[0].to(dtype=torch.bool).any().item()):
        raise ValueError("hidden provider returned no visible token")
    return {"token_hidden": token_hidden[0].contiguous(), "state_hidden": state_hidden[0].contiguous(), "attention_mask": attention[0].contiguous()}


def _provider_sha(provider: Any, label: str) -> str:
    return _sha(getattr(provider, "contract_sha256", None), f"{label} contract_sha256")


def _legal_utilities(step: LegalDynamicRagEpisodeStep, raw: Mapping[Any, float]) -> Mapping[DynamicRetrievalAction, float]:
    normalized: dict[DynamicRetrievalAction, float] = {}
    for raw_action, raw_value in raw.items():
        action = raw_action if isinstance(raw_action, DynamicRetrievalAction) else DynamicRetrievalAction(raw_action)
        if action in normalized:
            raise ValueError(f"counterfactual provider duplicated action {action.value}")
        normalized[action] = _finite(raw_value, f"counterfactual utility {action.value}")
    legal = set(step.valid_actions)
    normalized = {action: value for action, value in normalized.items() if action in legal}
    if not normalized or step.action not in normalized:
        raise ValueError("counterfactual provider must score the logged action and a legal action set")
    return normalized


def _counterfactual_target(step: LegalDynamicRagEpisodeStep, utilities: Mapping[DynamicRetrievalAction, float], reward: DynamicRewardConfig) -> tuple[DynamicRetrievalAction, float]:
    adjusted = {action: _finite(value, f"counterfactual utility {action.value}") - reward.action_cost(action) for action, value in utilities.items()}
    baseline = adjusted[step.action]
    best = min(adjusted, key=lambda action: (-adjusted[action], action.value))
    return best, _finite(adjusted[best] - baseline, "counterfactual gain over logged action")


def _runtime_payload(lineage: DynamicRuntimeTrainingLineage) -> Mapping[str, Any]:
    return {
        "source_dataset_sha256": lineage.source_dataset_sha256,
        "source_dataset_manifest_sha256": lineage.source_dataset_manifest_sha256,
        "runtime_stack_sha256": lineage.runtime_stack_sha256,
        "feature_provider_sha256": lineage.feature_provider_sha256,
        "behavior_policy_sha256": lineage.behavior_policy_sha256,
        "source_commit": lineage.source_commit,
        "reward_config": asdict(lineage.reward_config),
    }


def _runtime_lineage(raw: Any) -> DynamicRuntimeTrainingLineage:
    required = {"source_dataset_sha256", "source_dataset_manifest_sha256", "runtime_stack_sha256", "feature_provider_sha256", "behavior_policy_sha256", "source_commit", "reward_config"}
    if not isinstance(raw, Mapping) or set(raw) != required or not isinstance(raw["reward_config"], Mapping):
        raise ValueError("dynamic runtime lineage fields are invalid")
    return DynamicRuntimeTrainingLineage(
        source_dataset_sha256=raw["source_dataset_sha256"],
        source_dataset_manifest_sha256=raw["source_dataset_manifest_sha256"],
        runtime_stack_sha256=raw["runtime_stack_sha256"],
        feature_provider_sha256=raw["feature_provider_sha256"],
        behavior_policy_sha256=raw["behavior_policy_sha256"],
        source_commit=raw["source_commit"],
        reward_config=DynamicRewardConfig(**dict(raw["reward_config"])),
    )


def _materialization_identity(lineage: DynamicRuntimeTrainingLineage, value_sha: str, counterfactual_sha: str | None) -> str:
    return _digest({
        "schema": "rigorousrag-dynamic-trajectory-materialization-identity/v2",
        "runtime_lineage_sha256": lineage.lineage_sha256,
        "value_provider_sha256": value_sha,
        "counterfactual_provider_sha256": counterfactual_sha,
        "reward_config": asdict(lineage.reward_config),
        "episode_processing": "input_order_within_episode+first_seen_episode_order",
    })


def _cache_key_digest(root: Path) -> tuple[int, str]:
    database = root / "authority.sqlite"
    if database.is_symlink() or not database.is_file():
        raise ValueError("dynamic hidden cache authority index is missing or unsafe")
    digest = hashlib.sha256(); count = 0
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0) as connection:
        for (key,) in connection.execute("SELECT key FROM entries ORDER BY key"):
            digest.update(str(key).encode("utf-8")); digest.update(b"\n"); count += 1
    return count, digest.hexdigest()


def _rebase_publication(receipt: DynamicDatasetPublicationReceipt, final_root: Path) -> DynamicDatasetPublicationReceipt:
    splits = tuple(PublishedDynamicSplit(item.name, str(final_root / Path(item.path).name), item.sha256, item.record_count, item.record_id_sha256, item.episode_id_sha256) for item in receipt.splits)
    unsigned = {
        "schema": "rigorousrag-dynamic-dataset-publication-receipt/v1",
        "dataset_manifest_sha256": receipt.dataset_manifest_sha256,
        "source_set_sha256": receipt.source_set_sha256,
        "transformation_sha256": receipt.transformation_sha256,
        "split_policy_sha256": receipt.split_policy_sha256,
        "manifest_path": str(final_root / "dataset_manifest.json"),
        "splits": [asdict(item) for item in splits],
    }
    return DynamicDatasetPublicationReceipt(receipt.dataset_manifest_sha256, receipt.source_set_sha256, receipt.transformation_sha256, receipt.split_policy_sha256, unsigned["manifest_path"], splits, _digest(unsigned))


def _write_publication_receipt(path: Path, receipt: DynamicDatasetPublicationReceipt) -> None:
    _atomic(path, _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")


@dataclass(frozen=True)
class AuthoritativeDynamicCanonicalReceipt:
    runtime_lineage: Mapping[str, Any]
    runtime_lineage_sha256: str
    hidden_provider_sha256: str
    annotation_provider_sha256: str | None
    gain_provider_sha256: str | None
    value_provider_sha256: str
    counterfactual_provider_sha256: str | None
    planned_sha256: str
    planned_record_count: int
    planned_keyset_sha256: str
    gain_provenance_sha256: str
    materialization_identity_sha256: str
    materialized_sha256: str
    materialized_record_count: int
    materialized_episode_count: int
    dataset_publication_receipt_sha256: str
    dataset_manifest_sha256: str
    hidden_cache_producer_sha256: str
    hidden_cache_tokenizer_sha256: str
    hidden_cache_source_commit: str
    hidden_cache_config_sha256: str
    hidden_cache_identity_sha256: str
    hidden_cache_contract_sha256: str
    hidden_cache_authority_json_sha256: str
    hidden_cache_authority_db_sha256: str
    hidden_cache_entry_count: int
    hidden_cache_keyset_sha256: str
    require_promotable: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        lineage = _runtime_lineage(self.runtime_lineage)
        object.__setattr__(self, "runtime_lineage", _runtime_payload(lineage))
        for field in ("runtime_lineage_sha256", "hidden_provider_sha256", "value_provider_sha256", "planned_sha256", "planned_keyset_sha256", "gain_provenance_sha256", "materialization_identity_sha256", "materialized_sha256", "dataset_publication_receipt_sha256", "dataset_manifest_sha256", "hidden_cache_producer_sha256", "hidden_cache_tokenizer_sha256", "hidden_cache_config_sha256", "hidden_cache_identity_sha256", "hidden_cache_contract_sha256", "hidden_cache_authority_json_sha256", "hidden_cache_authority_db_sha256", "hidden_cache_keyset_sha256", "receipt_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        for field in ("annotation_provider_sha256", "gain_provider_sha256", "counterfactual_provider_sha256"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _sha(value, field))
        object.__setattr__(self, "hidden_cache_source_commit", _commit(self.hidden_cache_source_commit))
        for field in ("planned_record_count", "materialized_record_count", "materialized_episode_count", "hidden_cache_entry_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.require_promotable, bool):
            raise ValueError("require_promotable must be boolean")
        if lineage.lineage_sha256 != self.runtime_lineage_sha256:
            raise ValueError("dynamic canonical runtime lineage digest mismatch")
        if self.planned_record_count != self.materialized_record_count or self.materialized_record_count != self.hidden_cache_entry_count:
            raise ValueError("dynamic canonical record/cache counts differ")
        if self.planned_keyset_sha256 != self.hidden_cache_keyset_sha256:
            raise ValueError("dynamic canonical planned/final hidden-key universes differ")
        identity = self.hidden_identity()
        if identity.digest != self.hidden_cache_identity_sha256:
            raise ValueError("dynamic canonical hidden-cache identity digest mismatch")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative dynamic canonical receipt digest mismatch")

    def hidden_identity(self) -> SupervisionCacheIdentity:
        return SupervisionCacheIdentity(
            cache_kind="generator_hidden_states",
            producer_sha256=self.hidden_cache_producer_sha256,
            tokenizer_sha256=self.hidden_cache_tokenizer_sha256,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            source_commit=self.hidden_cache_source_commit,
            config_sha256=self.hidden_cache_config_sha256,
        )

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-dynamic-canonical-receipt/v2",
            **{field: getattr(self, field) for field in (
                "runtime_lineage", "runtime_lineage_sha256", "hidden_provider_sha256", "annotation_provider_sha256", "gain_provider_sha256", "value_provider_sha256", "counterfactual_provider_sha256", "planned_sha256", "planned_record_count", "planned_keyset_sha256", "gain_provenance_sha256", "materialization_identity_sha256", "materialized_sha256", "materialized_record_count", "materialized_episode_count", "dataset_publication_receipt_sha256", "dataset_manifest_sha256", "hidden_cache_producer_sha256", "hidden_cache_tokenizer_sha256", "hidden_cache_source_commit", "hidden_cache_config_sha256", "hidden_cache_identity_sha256", "hidden_cache_contract_sha256", "hidden_cache_authority_json_sha256", "hidden_cache_authority_db_sha256", "hidden_cache_entry_count", "hidden_cache_keyset_sha256", "require_promotable"
            )},
        }


@dataclass(frozen=True)
class VerifiedAuthoritativeDynamicCanonicalData:
    root: str
    dataset: VerifiedDynamicDatasetPublication
    hidden_cache: DiskBackedAuthoritativeSafetensorCache
    receipt: AuthoritativeDynamicCanonicalReceipt


def _spool(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    connection: sqlite3.Connection,
    ledger: SqliteIdentityLedger,
    planned_path: Path,
    hidden_provider_sha: str,
    annotation_provider: InformationNeedAnnotationProvider | None,
    annotation_sha: str | None,
    require_need_annotations: bool,
) -> tuple[int, str, str]:
    if not isinstance(steps, Sequence) or len(steps) <= 0:
        raise ValueError("dynamic canonical input must be a non-empty Sequence")
    connection.execute("CREATE TABLE episodes(episode_id TEXT PRIMARY KEY,first_ordinal INTEGER NOT NULL) WITHOUT ROWID")
    connection.execute("CREATE TABLE steps(episode_id TEXT NOT NULL,step_id TEXT NOT NULL,ordinal INTEGER NOT NULL UNIQUE,payload_json TEXT NOT NULL,PRIMARY KEY(episode_id,step_id)) WITHOUT ROWID")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".planned-", suffix=".tmp", dir=planned_path.parent)
    count = 0; digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for ordinal in range(len(steps)):
                step = steps[ordinal]
                if not isinstance(step, LegalDynamicRagEpisodeStep):
                    raise ValueError("dynamic canonical input must contain LegalDynamicRagEpisodeStep values")
                if count >= _MAX_RECORDS:
                    raise ValueError("dynamic canonical input exceeds record safety bound")
                spans = tuple(annotation_provider.spans(step)) if annotation_provider is not None else tuple(step.need_spans)
                if annotation_provider is None and require_need_annotations:
                    raise ValueError("dynamic need planning requires an explicit annotation provider")
                if any(not isinstance(span, TextSpan) or span.end > len(step.context) for span in spans):
                    raise ValueError(f"invalid information-need spans for {step.episode_id}:{step.step_id}")
                metadata = dict(step.metadata); metadata["hidden_provider_sha256"] = hidden_provider_sha
                if annotation_sha is not None:
                    metadata["need_annotation_provider_sha256"] = annotation_sha
                key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
                planned = replace(step, hidden_state_cache_key=key, need_spans=spans, metadata=metadata)
                encoded = _canonical(_step_payload(planned)); line = encoded + b"\n"
                if len(line) > _MAX_LINE_BYTES:
                    raise ValueError("dynamic planned record exceeds line safety bound")
                pair = dynamic_step_identity(planned.episode_id, planned.step_id)
                ledger.add_unique("dynamic-step", planned.episode_id, pair, payload_sha256=hashlib.sha256(encoded).hexdigest())
                ledger.add_unique("dynamic-hidden-key", planned.episode_id, key)
                try:
                    connection.execute("INSERT INTO steps(episode_id,step_id,ordinal,payload_json) VALUES(?,?,?,?)", (planned.episode_id, planned.step_id, ordinal, encoded.decode("utf-8")))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate dynamic step {planned.episode_id}:{planned.step_id}") from exc
                connection.execute("INSERT OR IGNORE INTO episodes(episode_id,first_ordinal) VALUES(?,?)", (planned.episode_id, ordinal))
                handle.write(line); digest.update(line); count += 1
                if count % 10_000 == 0:
                    connection.commit(); ledger.commit()
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_name, planned_path)
    finally:
        if os.path.exists(temporary_name): os.unlink(temporary_name)
    connection.commit(); ledger.commit()
    if ledger.count_unique("dynamic-step") != count:
        raise RuntimeError("dynamic planned identity count differs from spool")
    if _stream_sha(planned_path) != digest.hexdigest():
        raise RuntimeError("dynamic planned bytes changed during publication")
    return count, digest.hexdigest(), ledger.digest_unique("dynamic-hidden-key")


def _episode(connection: sqlite3.Connection, episode_id: str) -> list[LegalDynamicRagEpisodeStep]:
    rows = connection.execute("SELECT payload_json FROM steps WHERE episode_id=? ORDER BY ordinal", (episode_id,)).fetchall()
    if not rows or len(rows) > _MAX_EPISODE_STEPS:
        raise ValueError(f"episode {episode_id!r} must contain 1..{_MAX_EPISODE_STEPS} steps")
    result = []
    for row in rows:
        result.append(parse_authoritative_dynamic_step(json.loads(str(row[0]), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))))
    return result


def build_authoritative_dynamic_canonical_training_data(
    steps: Sequence[LegalDynamicRagEpisodeStep], *, hidden_provider: BoundGeneratorHiddenStateProvider,
    annotation_provider: InformationNeedAnnotationProvider | None, realized_gain_provider: RealizedRetrievalGainProvider | None,
    value_provider: LoggedValueProvider, counterfactual_provider: CounterfactualActionProvider | None,
    runtime_lineage: DynamicRuntimeTrainingLineage, governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy, output_dir: str | Path, require_need_annotations: bool = True,
) -> VerifiedAuthoritativeDynamicCanonicalData:
    if not isinstance(runtime_lineage, DynamicRuntimeTrainingLineage) or not isinstance(governance, DynamicDatasetGovernance) or not isinstance(split_policy, EpisodeSplitPolicy):
        raise ValueError("runtime_lineage/governance/split_policy have invalid types")
    if not isinstance(require_need_annotations, bool):
        raise ValueError("require_need_annotations must be boolean")
    hidden_sha = _provider_sha(hidden_provider, "hidden provider")
    generator_sha = _sha(getattr(hidden_provider, "generator_sha256", None), "hidden generator_sha256")
    tokenizer_sha = _sha(getattr(hidden_provider, "tokenizer_sha256", None), "hidden tokenizer_sha256")
    annotation_sha = None if annotation_provider is None else _provider_sha(annotation_provider, "annotation provider")
    gain_sha = None if realized_gain_provider is None else _provider_sha(realized_gain_provider, "gain provider")
    value_sha = _provider_sha(value_provider, "value provider")
    counterfactual_sha = None if counterfactual_provider is None else _provider_sha(counterfactual_provider, "counterfactual provider")
    materialization_sha = _materialization_identity(runtime_lineage, value_sha, counterfactual_sha)

    root = safe_advanced_path(output_dir, label="authoritative dynamic canonical output", must_exist=False)
    if root.exists(): raise ValueError("authoritative dynamic canonical output must not already exist")
    parent = safe_advanced_path(root.parent, label="authoritative dynamic canonical parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'dynamic'}-stage-", dir=parent))
    spool_path, ledger_path = stage / ".spool.sqlite", stage / ".identity.sqlite"
    spool: sqlite3.Connection | None = None; ledger: SqliteIdentityLedger | None = None; published = False
    try:
        spool = sqlite3.connect(str(spool_path), timeout=30.0); spool.execute("PRAGMA journal_mode=WAL"); spool.execute("PRAGMA synchronous=FULL"); spool.execute("PRAGMA temp_store=FILE")
        ledger = SqliteIdentityLedger(ledger_path)
        planned_path = stage / "planned.dynamic.jsonl"
        planned_count, planned_sha, planned_key_sha = _spool(steps, connection=spool, ledger=ledger, planned_path=planned_path, hidden_provider_sha=hidden_sha, annotation_provider=annotation_provider, annotation_sha=annotation_sha, require_need_annotations=require_need_annotations)

        materialized_path = stage / "materialized.dynamic.jsonl"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".materialized-", suffix=".tmp", dir=stage)
        materialized_digest, gain_digest = hashlib.sha256(), hashlib.sha256(); record_count = 0; episode_count = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for (episode_id_raw,) in spool.execute("SELECT episode_id FROM episodes ORDER BY first_ordinal"):
                    episode = _episode(spool, str(episode_id_raw))
                    if realized_gain_provider is not None:
                        gains = tuple(_finite(value, "realized retrieval gain") for value in realized_gain_provider.gains(episode))
                        if len(gains) != len(episode): raise ValueError("gain provider returned wrong episode length")
                        gain_bound = []
                        for step, gain in zip(episode, gains):
                            metadata = dict(step.metadata); metadata["realized_retrieval_gain_provider_sha256"] = gain_sha
                            gain_bound.append(replace(step, realized_retrieval_gain=gain, metadata=metadata))
                    else:
                        gain_bound = episode
                    for step in gain_bound:
                        marker = _sha(step.metadata.get("realized_retrieval_gain_provider_sha256"), "realized-gain provider sha256")
                        gain_digest.update(_canonical({"episode_id": step.episode_id, "step_id": step.step_id, "provider_sha256": marker, "gain": step.realized_retrieval_gain}) + b"\n")
                    values = tuple(_finite(value, "logged state value") for value in value_provider.values(gain_bound))
                    if len(values) != len(gain_bound): raise ValueError("value provider returned wrong episode length")
                    rewards = trajectory_rewards(gain_bound, runtime_lineage.reward_config)
                    targets = generalized_advantage_estimate(rewards, values, discount=runtime_lineage.reward_config.discount, gae_lambda=runtime_lineage.reward_config.gae_lambda, bootstrap_value=0.0)
                    for index, step in enumerate(gain_bound):
                        metadata = dict(step.metadata); metadata["trajectory_identity_sha256"] = materialization_sha
                        if counterfactual_provider is not None:
                            action, improvement = _counterfactual_target(step, _legal_utilities(step, counterfactual_provider.action_utilities(step)), runtime_lineage.reward_config)
                            metadata["counterfactual_best_action"] = action.value; metadata["counterfactual_logged_action"] = step.action.value; metadata["counterfactual_gain_over_logged_action"] = format(improvement, ".17g")
                        output = replace(step, advantage=_finite(targets.advantages[index], "advantage"), value_target=_finite(targets.returns[index], "value target"), metadata=metadata)
                        line = _canonical(_step_payload(output)) + b"\n"
                        if len(line) > _MAX_LINE_BYTES: raise ValueError("materialized dynamic record exceeds line safety bound")
                        handle.write(line); materialized_digest.update(line); record_count += 1
                    episode_count += 1
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary_name, materialized_path)
        finally:
            if os.path.exists(temporary_name): os.unlink(temporary_name)
        if record_count != planned_count or episode_count <= 0: raise RuntimeError("dynamic materialization counts differ from plan")
        materialized_sha = materialized_digest.hexdigest(); gain_provenance_sha = gain_digest.hexdigest()
        if _stream_sha(materialized_path) != materialized_sha: raise RuntimeError("materialized dynamic bytes changed")

        source_lineage_sha = _digest({"schema": "rigorousrag-authoritative-dynamic-prepublication/v2", "runtime_lineage_sha256": runtime_lineage.lineage_sha256, "hidden_provider_sha256": hidden_sha, "annotation_provider_sha256": annotation_sha, "gain_provider_sha256": gain_sha, "value_provider_sha256": value_sha, "counterfactual_provider_sha256": counterfactual_sha, "planned_sha256": planned_sha, "planned_record_count": planned_count, "planned_keyset_sha256": planned_key_sha, "gain_provenance_sha256": gain_provenance_sha, "materialization_identity_sha256": materialization_sha, "materialized_sha256": materialized_sha, "materialized_record_count": record_count, "materialized_episode_count": episode_count})
        source = DynamicTrajectorySource(str(materialized_path), materialized_sha, source_lineage_sha)
        manifest, publication = publish_dynamic_training_dataset((source,), governance=governance, split_policy=split_policy, output_dir=stage / "published")
        verified_stage = verify_dynamic_dataset_publication(stage / "published" / "publication_receipt.json", sources=(source,), require_promotable=governance.require_promotable)
        if verified_stage.manifest.manifest_digest != manifest.manifest_digest or verified_stage.receipt.receipt_sha256 != publication.receipt_sha256: raise RuntimeError("dynamic dataset changed before rebasing")
        rebased = _rebase_publication(publication, root / "published"); _write_publication_receipt(stage / "published" / "publication_receipt.json", rebased)

        hidden_config_sha = _digest({"schema": "rigorousrag-authoritative-dynamic-hidden-cache-config/v2", "hidden_provider_sha256": hidden_sha, "dataset_manifest_sha256": manifest.manifest_digest, "prepublication_lineage_sha256": source_lineage_sha, "dataset_publication_receipt_sha256": rebased.receipt_sha256})
        hidden_identity = SupervisionCacheIdentity("generator_hidden_states", generator_sha, tokenizer_sha, manifest.manifest_digest, runtime_lineage.source_commit, hidden_config_sha)
        hidden_cache = DiskBackedAuthoritativeSafetensorCache(stage / "hidden_cache", hidden_identity)
        hidden_ledger_path = stage / ".hidden.sqlite"; hidden_ledger = SqliteIdentityLedger(hidden_ledger_path)
        try:
            hidden_count = 0
            for split in verified_stage.manifest.splits:
                dataset = verified_stage.split(split.name)
                for index in range(len(dataset)):
                    step = dataset[index]; expected = dynamic_hidden_cache_key(step.episode_id, step.step_id)
                    if step.hidden_state_cache_key != expected or step.metadata.get("hidden_provider_sha256") != hidden_sha: raise ValueError("published hidden-key/provider lineage differs")
                    hidden_ledger.add_unique("dynamic-hidden-key", step.episode_id, expected)
                    hidden_cache.put(expected, _normalized_hidden(hidden_provider.encode([step.context]))); hidden_count += 1
                    if hidden_count % 10_000 == 0: hidden_ledger.commit()
            hidden_ledger.commit(); hidden_key_sha = hidden_ledger.digest_unique("dynamic-hidden-key")
        finally:
            hidden_ledger.close()
            for suffix in ("", "-wal", "-shm"): Path(str(hidden_ledger_path) + suffix).unlink(missing_ok=True)
        if hidden_count != record_count or hidden_key_sha != planned_key_sha: raise ValueError("dynamic hidden-key universe differs from planned corpus")
        hidden_cache.seal(); cache_count, cache_key_sha = _cache_key_digest(hidden_cache.root)
        if cache_count != record_count or cache_key_sha != hidden_key_sha: raise ValueError("dynamic hidden-cache key authority differs")
        authority_json_sha, authority_db_sha = hidden_cache.authority_file_sha256s

        unsigned = {"schema": "rigorousrag-authoritative-dynamic-canonical-receipt/v2", "runtime_lineage": _runtime_payload(runtime_lineage), "runtime_lineage_sha256": runtime_lineage.lineage_sha256, "hidden_provider_sha256": hidden_sha, "annotation_provider_sha256": annotation_sha, "gain_provider_sha256": gain_sha, "value_provider_sha256": value_sha, "counterfactual_provider_sha256": counterfactual_sha, "planned_sha256": planned_sha, "planned_record_count": planned_count, "planned_keyset_sha256": planned_key_sha, "gain_provenance_sha256": gain_provenance_sha, "materialization_identity_sha256": materialization_sha, "materialized_sha256": materialized_sha, "materialized_record_count": record_count, "materialized_episode_count": episode_count, "dataset_publication_receipt_sha256": rebased.receipt_sha256, "dataset_manifest_sha256": manifest.manifest_digest, "hidden_cache_producer_sha256": generator_sha, "hidden_cache_tokenizer_sha256": tokenizer_sha, "hidden_cache_source_commit": runtime_lineage.source_commit, "hidden_cache_config_sha256": hidden_config_sha, "hidden_cache_identity_sha256": hidden_identity.digest, "hidden_cache_contract_sha256": hidden_cache.contract_sha256, "hidden_cache_authority_json_sha256": authority_json_sha, "hidden_cache_authority_db_sha256": authority_db_sha, "hidden_cache_entry_count": hidden_cache.entry_count, "hidden_cache_keyset_sha256": hidden_key_sha, "require_promotable": governance.require_promotable}
        receipt = AuthoritativeDynamicCanonicalReceipt(**{key: value for key, value in unsigned.items() if key != "schema"}, receipt_sha256=_digest(unsigned)); _atomic(stage / "canonical_receipt.json", _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")
        spool.close(); spool = None; ledger.close(); ledger = None
        for base in (spool_path, ledger_path):
            for suffix in ("", "-wal", "-shm"): Path(str(base) + suffix).unlink(missing_ok=True)
        if {item.name for item in stage.iterdir()} != {"planned.dynamic.jsonl", "materialized.dynamic.jsonl", "published", "hidden_cache", "canonical_receipt.json"}: raise RuntimeError("dynamic canonical staging directory is not closed")
        os.replace(stage, root); published = True
        try: verified = verify_authoritative_dynamic_canonical_training_data(root / "canonical_receipt.json")
        except Exception:
            shutil.rmtree(root, ignore_errors=True); published = False; raise
        if verified.receipt.receipt_sha256 != receipt.receipt_sha256:
            shutil.rmtree(root, ignore_errors=True); published = False; raise RuntimeError("dynamic canonical identity changed after publication")
        return verified
    finally:
        if spool is not None: spool.close()
        if ledger is not None: ledger.close()
        if not published: shutil.rmtree(stage, ignore_errors=True)


def _parse_receipt(raw: Mapping[str, Any]) -> AuthoritativeDynamicCanonicalReceipt:
    expected = set(AuthoritativeDynamicCanonicalReceipt.__dataclass_fields__) | {"schema"}
    if set(raw) != expected or raw.get("schema") != "rigorousrag-authoritative-dynamic-canonical-receipt/v2": raise ValueError("unsupported authoritative dynamic canonical receipt schema")
    return AuthoritativeDynamicCanonicalReceipt(**{key: value for key, value in raw.items() if key != "schema"})


def _scan_planned(path: Path, receipt: AuthoritativeDynamicCanonicalReceipt, ledger: SqliteIdentityLedger) -> int:
    count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            step = _strict_step_line(raw, f"planned dynamic line {line_number}"); expected = dynamic_hidden_cache_key(step.episode_id, step.step_id)
            if step.hidden_state_cache_key != expected or step.metadata.get("hidden_provider_sha256") != receipt.hidden_provider_sha256: raise ValueError("planned dynamic hidden-key/provider lineage differs")
            if receipt.annotation_provider_sha256 is not None and step.metadata.get("need_annotation_provider_sha256") != receipt.annotation_provider_sha256: raise ValueError("planned annotation-provider lineage differs")
            ledger.add_unique("dynamic-hidden-key", step.episode_id, expected); count += 1
            if count % 10_000 == 0: ledger.commit()
    ledger.commit(); return count


def _scan_materialized(path: Path, receipt: AuthoritativeDynamicCanonicalReceipt, ledger: SqliteIdentityLedger) -> tuple[int, int, str]:
    count = 0; gain_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            step = _strict_step_line(raw, f"materialized dynamic line {line_number}"); expected = dynamic_hidden_cache_key(step.episode_id, step.step_id)
            if step.hidden_state_cache_key != expected or step.metadata.get("hidden_provider_sha256") != receipt.hidden_provider_sha256: raise ValueError("materialized hidden-key/provider lineage differs")
            if step.metadata.get("trajectory_identity_sha256") != receipt.materialization_identity_sha256: raise ValueError("materialized trajectory identity differs")
            marker = _sha(step.metadata.get("realized_retrieval_gain_provider_sha256"), "materialized gain provider sha256")
            if receipt.gain_provider_sha256 is not None and marker != receipt.gain_provider_sha256: raise ValueError("materialized gain-provider lineage differs")
            gain_digest.update(_canonical({"episode_id": step.episode_id, "step_id": step.step_id, "provider_sha256": marker, "gain": step.realized_retrieval_gain}) + b"\n")
            ledger.add_unique("materialized-step", step.episode_id, dynamic_step_identity(step.episode_id, step.step_id)); ledger.add_set("dynamic-episode", "all", step.episode_id); count += 1
            if count % 10_000 == 0: ledger.commit()
    ledger.commit(); return count, ledger.count_set("dynamic-episode", scope="all"), gain_digest.hexdigest()


def verify_authoritative_dynamic_canonical_training_data(receipt_path: str | Path) -> VerifiedAuthoritativeDynamicCanonicalData:
    raw_path = Path(receipt_path).expanduser()
    if raw_path.is_symlink(): raise ValueError("dynamic canonical receipt may not be a symlink")
    selected = safe_advanced_path(raw_path, label="authoritative dynamic canonical receipt", must_exist=True, require_file=True); root = selected.parent
    if selected != root / "canonical_receipt.json": raise ValueError("dynamic canonical receipt must use canonical filename")
    expected_top = {"planned.dynamic.jsonl", "materialized.dynamic.jsonl", "published", "hidden_cache", "canonical_receipt.json"}
    if {item.name for item in root.iterdir()} != expected_top: raise ValueError("dynamic canonical publication root is not closed")
    if any(item.is_symlink() for item in root.iterdir()): raise ValueError("dynamic canonical publication contains symlinked top-level children")
    receipt = _parse_receipt(_strict_json(selected, "authoritative dynamic canonical receipt")); lineage = _runtime_lineage(receipt.runtime_lineage)
    if _materialization_identity(lineage, receipt.value_provider_sha256, receipt.counterfactual_provider_sha256) != receipt.materialization_identity_sha256: raise ValueError("dynamic materialization identity cannot be reconstructed")
    planned = safe_advanced_path(root / "planned.dynamic.jsonl", label="planned dynamic trajectory", must_exist=True, require_file=True); materialized = safe_advanced_path(root / "materialized.dynamic.jsonl", label="materialized dynamic trajectory", must_exist=True, require_file=True)
    if _stream_sha(planned) != receipt.planned_sha256 or _stream_sha(materialized) != receipt.materialized_sha256: raise ValueError("dynamic planned/materialized bytes differ from receipt")

    verify_ledger_path = root.parent / f".{root.name}.dynamic-verify.sqlite"; verify_ledger = SqliteIdentityLedger(verify_ledger_path)
    try:
        planned_count = _scan_planned(planned, receipt, verify_ledger)
        planned_key_sha = verify_ledger.digest_unique("dynamic-hidden-key")
        materialized_count, episode_count, gain_sha = _scan_materialized(materialized, receipt, verify_ledger)
        if planned_count != receipt.planned_record_count or materialized_count != receipt.materialized_record_count or episode_count != receipt.materialized_episode_count or planned_key_sha != receipt.planned_keyset_sha256 or gain_sha != receipt.gain_provenance_sha256: raise ValueError("dynamic canonical retained trajectory lineage differs from receipt")
        source_lineage_sha = _digest({"schema": "rigorousrag-authoritative-dynamic-prepublication/v2", "runtime_lineage_sha256": receipt.runtime_lineage_sha256, "hidden_provider_sha256": receipt.hidden_provider_sha256, "annotation_provider_sha256": receipt.annotation_provider_sha256, "gain_provider_sha256": receipt.gain_provider_sha256, "value_provider_sha256": receipt.value_provider_sha256, "counterfactual_provider_sha256": receipt.counterfactual_provider_sha256, "planned_sha256": receipt.planned_sha256, "planned_record_count": receipt.planned_record_count, "planned_keyset_sha256": receipt.planned_keyset_sha256, "gain_provenance_sha256": receipt.gain_provenance_sha256, "materialization_identity_sha256": receipt.materialization_identity_sha256, "materialized_sha256": receipt.materialized_sha256, "materialized_record_count": receipt.materialized_record_count, "materialized_episode_count": receipt.materialized_episode_count})
        source = DynamicTrajectorySource(str(materialized), receipt.materialized_sha256, source_lineage_sha)
        dataset = verify_dynamic_dataset_publication(root / "published" / "publication_receipt.json", sources=(source,), require_promotable=receipt.require_promotable)
        if dataset.receipt.receipt_sha256 != receipt.dataset_publication_receipt_sha256 or dataset.manifest.manifest_digest != receipt.dataset_manifest_sha256: raise ValueError("dynamic final dataset identity differs from canonical receipt")
        final_ledger_path = root.parent / f".{root.name}.dynamic-final.sqlite"; final_ledger = SqliteIdentityLedger(final_ledger_path)
        try:
            final_count = 0
            for split in dataset.manifest.splits:
                records = dataset.split(split.name)
                for index in range(len(records)):
                    step = records[index]; key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
                    if step.hidden_state_cache_key != key or step.metadata.get("hidden_provider_sha256") != receipt.hidden_provider_sha256 or step.metadata.get("trajectory_identity_sha256") != receipt.materialization_identity_sha256: raise ValueError("dynamic published training lineage differs")
                    final_ledger.add_unique("dynamic-hidden-key", step.episode_id, key); final_count += 1
                    if final_count % 10_000 == 0: final_ledger.commit()
            final_ledger.commit(); final_key_sha = final_ledger.digest_unique("dynamic-hidden-key")
        finally:
            final_ledger.close()
            for suffix in ("", "-wal", "-shm"): Path(str(final_ledger_path) + suffix).unlink(missing_ok=True)
        if final_count != receipt.materialized_record_count or final_key_sha != receipt.hidden_cache_keyset_sha256: raise ValueError("dynamic final dataset hidden-key universe differs")

        cache = DiskBackedAuthoritativeSafetensorCache(root / "hidden_cache", receipt.hidden_identity())
        if cache.assert_sealed_integrity() != receipt.hidden_cache_contract_sha256: raise ValueError("dynamic hidden-cache contract differs")
        authority_json_sha, authority_db_sha = cache.authority_file_sha256s; cache_count, cache_key_sha = _cache_key_digest(cache.root)
        if authority_json_sha != receipt.hidden_cache_authority_json_sha256 or authority_db_sha != receipt.hidden_cache_authority_db_sha256 or cache.entry_count != receipt.hidden_cache_entry_count or cache_count != final_count or cache_key_sha != final_key_sha: raise ValueError("dynamic hidden-cache authority differs from receipt/final dataset")
        return VerifiedAuthoritativeDynamicCanonicalData(str(root), dataset, cache, receipt)
    finally:
        verify_ledger.close()
        for suffix in ("", "-wal", "-shm"): Path(str(verify_ledger_path) + suffix).unlink(missing_ok=True)


__all__ = ["AuthoritativeDynamicCanonicalReceipt", "VerifiedAuthoritativeDynamicCanonicalData", "build_authoritative_dynamic_canonical_training_data", "verify_authoritative_dynamic_canonical_training_data"]
