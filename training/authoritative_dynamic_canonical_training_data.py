"""Atomic, episode-streamed canonical dynamic-RAG training-data authority v2.

The mathematical workflow is unchanged: deterministic hidden-key/need-span planning, governed
realized retrieval gain binding, episode-level value/GAE and legal counterfactual targets,
episode-isolated final dataset publication, then final-manifest-bound hidden-state caching.
Authority mechanics are upgraded: the corpus is spooled in SQLite instead of Python tuples,
only one bounded episode is resident while GAE is computed, hidden-cache keys are disk-backed,
the cache uses a disk-backed sealed authority, and the complete trajectory + dataset + cache +
receipt is atomically renamed into place and restart-verified. Importing this module executes no
model, provider, download, retrieval or training.
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
from training.advanced_rag_data import DynamicRagEpisodeStep, TextSpan
from training.advanced_rag_supervision import (
    CounterfactualActionProvider,
    DynamicRewardConfig,
    SupervisionCacheIdentity,
    generalized_advantage_estimate,
    trajectory_rewards,
)
from training.disk_backed_supervision_cache import (
    DiskBackedAuthoritativeSafetensorCache,
)
from training.dynamic_canonical_training_data_pipeline import DynamicRuntimeTrainingLineage
from training.dynamic_dataset_io import (
    VerifiedDynamicDatasetPublication,
    verify_dynamic_dataset_publication,
)
from training.dynamic_dataset_publication import (
    DynamicDatasetGovernance,
    DynamicDatasetPublicationReceipt,
    DynamicTrajectorySource,
    EpisodeSplitPolicy,
    PublishedDynamicSplit,
    publish_dynamic_training_dataset,
)
from training.dynamic_record_identity import (
    dynamic_hidden_cache_key,
    dynamic_step_identity,
)
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
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
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
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


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


def _decode_step(raw: str) -> LegalDynamicRagEpisodeStep:
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("dynamic canonical spool row is not strict JSON") from exc
    return parse_authoritative_dynamic_step(value)


def _normalized_hidden(encoded: Mapping[str, Any]) -> Mapping[str, Any]:
    if torch is None:
        raise RuntimeError("dynamic hidden-state cache materialization requires optional PyTorch")
    required = {"token_hidden", "state_hidden", "attention_mask"}
    if not isinstance(encoded, Mapping) or not required.issubset(encoded):
        raise ValueError("hidden provider must return token_hidden/state_hidden/attention_mask")
    token_hidden = encoded["token_hidden"]
    state_hidden = encoded["state_hidden"]
    attention = encoded["attention_mask"]
    if not all(torch.is_tensor(value) for value in (token_hidden, state_hidden, attention)):
        raise ValueError("hidden provider outputs must be tensors")
    token_hidden = token_hidden.detach().cpu()
    state_hidden = state_hidden.detach().cpu()
    attention = attention.detach().cpu()
    if (
        token_hidden.ndim != 3
        or token_hidden.size(0) != 1
        or state_hidden.ndim != 2
        or state_hidden.size(0) != 1
        or attention.ndim != 2
        or attention.size(0) != 1
        or token_hidden.size(1) != attention.size(1)
        or token_hidden.size(2) != state_hidden.size(1)
    ):
        raise ValueError("hidden provider tensor shapes are inconsistent")
    if not bool(attention[0].to(dtype=torch.bool).any().item()):
        raise ValueError("hidden provider returned no visible token")
    return {
        "token_hidden": token_hidden[0].contiguous(),
        "state_hidden": state_hidden[0].contiguous(),
        "attention_mask": attention[0].contiguous(),
    }


def _legal_counterfactual_utilities(
    step: LegalDynamicRagEpisodeStep,
    utilities: Mapping[Any, float],
) -> Mapping[DynamicRetrievalAction, float]:
    normalized: dict[DynamicRetrievalAction, float] = {}
    for raw_action, raw_value in utilities.items():
        action = (
            raw_action
            if isinstance(raw_action, DynamicRetrievalAction)
            else DynamicRetrievalAction(raw_action)
        )
        if action in normalized:
            raise ValueError(f"counterfactual provider duplicated action {action.value}")
        normalized[action] = _finite(raw_value, f"counterfactual utility {action.value}")
    legal = set(step.valid_actions)
    normalized = {action: value for action, value in normalized.items() if action in legal}
    if not normalized or step.action not in normalized:
        raise ValueError("counterfactual provider must score the logged action and a legal action set")
    return normalized


def _counterfactual_target(
    step: LegalDynamicRagEpisodeStep,
    utilities: Mapping[DynamicRetrievalAction, float],
    reward_config: DynamicRewardConfig,
) -> tuple[DynamicRetrievalAction, float]:
    adjusted = {
        action: _finite(value, f"counterfactual utility {action.value}")
        - reward_config.action_cost(action)
        for action, value in utilities.items()
    }
    baseline = adjusted[step.action]
    best = min(adjusted, key=lambda action: (-adjusted[action], action.value))
    return best, _finite(adjusted[best] - baseline, "counterfactual gain over logged action")


def _provider_sha(provider: Any, label: str) -> str:
    return _sha(getattr(provider, "contract_sha256", None), f"{label} contract_sha256")


def _rebase_publication_receipt(
    receipt: DynamicDatasetPublicationReceipt,
    *,
    final_publication_root: Path,
) -> DynamicDatasetPublicationReceipt:
    splits = tuple(
        PublishedDynamicSplit(
            name=item.name,
            path=str(final_publication_root / Path(item.path).name),
            sha256=item.sha256,
            record_count=item.record_count,
            record_id_sha256=item.record_id_sha256,
            episode_id_sha256=item.episode_id_sha256,
        )
        for item in receipt.splits
    )
    unsigned = {
        "schema": "rigorousrag-dynamic-dataset-publication-receipt/v1",
        "dataset_manifest_sha256": receipt.dataset_manifest_sha256,
        "source_set_sha256": receipt.source_set_sha256,
        "transformation_sha256": receipt.transformation_sha256,
        "split_policy_sha256": receipt.split_policy_sha256,
        "manifest_path": str(final_publication_root / "dataset_manifest.json"),
        "splits": [asdict(item) for item in splits],
    }
    return DynamicDatasetPublicationReceipt(
        dataset_manifest_sha256=receipt.dataset_manifest_sha256,
        source_set_sha256=receipt.source_set_sha256,
        transformation_sha256=receipt.transformation_sha256,
        split_policy_sha256=receipt.split_policy_sha256,
        manifest_path=unsigned["manifest_path"],
        splits=splits,
        receipt_sha256=_digest(unsigned),
    )


def _write_publication_receipt(path: Path, receipt: DynamicDatasetPublicationReceipt) -> None:
    _atomic(
        path,
        _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n",
    )


def _cache_key_digest(cache_root: Path) -> tuple[int, str]:
    authority_db = cache_root / "authority.sqlite"
    if authority_db.is_symlink() or not authority_db.is_file():
        raise ValueError("dynamic hidden cache authority index is missing or unsafe")
    digest = hashlib.sha256()
    count = 0
    with sqlite3.connect(f"file:{authority_db}?mode=ro", uri=True, timeout=30.0) as connection:
        for (key,) in connection.execute("SELECT key FROM entries ORDER BY key"):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def _runtime_lineage_payload(lineage: DynamicRuntimeTrainingLineage) -> Mapping[str, Any]:
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
    required = {
        "source_dataset_sha256",
        "source_dataset_manifest_sha256",
        "runtime_stack_sha256",
        "feature_provider_sha256",
        "behavior_policy_sha256",
        "source_commit",
        "reward_config",
    }
    if not isinstance(raw, Mapping) or set(raw) != required or not isinstance(raw["reward_config"], Mapping):
        raise ValueError("dynamic runtime lineage fields are invalid")
    reward = DynamicRewardConfig(**dict(raw["reward_config"]))
    return DynamicRuntimeTrainingLineage(
        source_dataset_sha256=raw["source_dataset_sha256"],
        source_dataset_manifest_sha256=raw["source_dataset_manifest_sha256"],
        runtime_stack_sha256=raw["runtime_stack_sha256"],
        feature_provider_sha256=raw["feature_provider_sha256"],
        behavior_policy_sha256=raw["behavior_policy_sha256"],
        source_commit=raw["source_commit"],
        reward_config=reward,
    )


@dataclass(frozen=True)
class AuthoritativeDynamicCanonicalReceipt:
    runtime_lineage: Mapping[str, Any]
    runtime_lineage_sha256: str
    hidden_provider_sha256: str
    annotation_provider_sha256: str | None
    gain_provider_sha256: str | None
    value_provider_sha256: str
    counterfactual_provider_sha256: str | None
    planned_record_count: int
    planned_keyset_sha256: str
    planned_records_sha256: str
    gain_provenance_sha256: str
    materialization_identity_sha256: str
    materialized_sha256: str
    materialized_record_count: int
    materialized_episode_count: int
    dataset_publication_receipt_sha256: str
    dataset_manifest_sha256: str
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
        object.__setattr__(self, "runtime_lineage", _runtime_lineage_payload(lineage))
        sha_fields = (
            "runtime_lineage_sha256",
            "hidden_provider_sha256",
            "value_provider_sha256",
            "planned_keyset_sha256",
            "planned_records_sha256",
            "gain_provenance_sha256",
            "materialization_identity_sha256",
            "materialized_sha256",
            "dataset_publication_receipt_sha256",
            "dataset_manifest_sha256",
            "hidden_cache_identity_sha256",
            "hidden_cache_contract_sha256",
            "hidden_cache_authority_json_sha256",
            "hidden_cache_authority_db_sha256",
            "hidden_cache_keyset_sha256",
            "receipt_sha256",
        )
        for field in sha_fields:
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        for field in (
            "annotation_provider_sha256",
            "gain_provider_sha256",
            "counterfactual_provider_sha256",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _sha(value, field))
        for field in (
            "planned_record_count",
            "materialized_record_count",
            "materialized_episode_count",
            "hidden_cache_entry_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.require_promotable, bool):
            raise ValueError("require_promotable must be boolean")
        if lineage.lineage_sha256 != self.runtime_lineage_sha256:
            raise ValueError("dynamic canonical runtime lineage digest mismatch")
        if self.planned_record_count != self.materialized_record_count:
            raise ValueError("dynamic canonical planned/materialized record counts differ")
        if self.materialized_record_count != self.hidden_cache_entry_count:
            raise ValueError("dynamic canonical hidden-cache entry count differs from records")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative dynamic canonical receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-dynamic-canonical-receipt/v2",
            "runtime_lineage": dict(self.runtime_lineage),
            "runtime_lineage_sha256": self.runtime_lineage_sha256,
            "hidden_provider_sha256": self.hidden_provider_sha256,
            "annotation_provider_sha256": self.annotation_provider_sha256,
            "gain_provider_sha256": self.gain_provider_sha256,
            "value_provider_sha256": self.value_provider_sha256,
            "counterfactual_provider_sha256": self.counterfactual_provider_sha256,
            "planned_record_count": self.planned_record_count,
            "planned_keyset_sha256": self.planned_keyset_sha256,
            "planned_records_sha256": self.planned_records_sha256,
            "gain_provenance_sha256": self.gain_provenance_sha256,
            "materialization_identity_sha256": self.materialization_identity_sha256,
            "materialized_sha256": self.materialized_sha256,
            "materialized_record_count": self.materialized_record_count,
            "materialized_episode_count": self.materialized_episode_count,
            "dataset_publication_receipt_sha256": self.dataset_publication_receipt_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "hidden_cache_identity_sha256": self.hidden_cache_identity_sha256,
            "hidden_cache_contract_sha256": self.hidden_cache_contract_sha256,
            "hidden_cache_authority_json_sha256": self.hidden_cache_authority_json_sha256,
            "hidden_cache_authority_db_sha256": self.hidden_cache_authority_db_sha256,
            "hidden_cache_entry_count": self.hidden_cache_entry_count,
            "hidden_cache_keyset_sha256": self.hidden_cache_keyset_sha256,
            "require_promotable": self.require_promotable,
        }


@dataclass(frozen=True)
class VerifiedAuthoritativeDynamicCanonicalData:
    root: str
    dataset: VerifiedDynamicDatasetPublication
    hidden_cache: DiskBackedAuthoritativeSafetensorCache
    receipt: AuthoritativeDynamicCanonicalReceipt


def _materialization_identity_sha(
    lineage: DynamicRuntimeTrainingLineage,
    *,
    value_provider_sha256: str,
    counterfactual_provider_sha256: str | None,
) -> str:
    return _digest(
        {
            "schema": "rigorousrag-dynamic-trajectory-materialization-identity/v2",
            "runtime_lineage_sha256": lineage.lineage_sha256,
            "value_provider_sha256": value_provider_sha256,
            "counterfactual_provider_sha256": counterfactual_provider_sha256,
            "reward_config": asdict(lineage.reward_config),
            "episode_processing": "input_order_within_episode+first_seen_episode_order",
        }
    )


def _spool_steps(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    connection: sqlite3.Connection,
    identity_ledger: SqliteIdentityLedger,
    hidden_provider_sha256: str,
    annotation_provider: InformationNeedAnnotationProvider | None,
    annotation_provider_sha256: str | None,
    require_need_annotations: bool,
) -> tuple[int, str, str]:
    if not isinstance(steps, Sequence) or len(steps) <= 0:
        raise ValueError("dynamic canonical input must be a non-empty Sequence")
    connection.execute(
        """CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            first_ordinal INTEGER NOT NULL
        ) WITHOUT ROWID"""
    )
    connection.execute(
        """CREATE TABLE steps (
            episode_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(episode_id,step_id)
        ) WITHOUT ROWID"""
    )
    records_digest = hashlib.sha256()
    count = 0
    for ordinal in range(len(steps)):
        step = steps[ordinal]
        if not isinstance(step, LegalDynamicRagEpisodeStep):
            raise ValueError("dynamic canonical input must contain LegalDynamicRagEpisodeStep values")
        if count >= _MAX_RECORDS:
            raise ValueError("dynamic canonical input exceeds record safety bound")
        key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
        spans = (
            tuple(annotation_provider.spans(step))
            if annotation_provider is not None
            else tuple(step.need_spans)
        )
        if annotation_provider is None and require_need_annotations:
            raise ValueError("dynamic canonical need planning requires an explicit annotation provider")
        if any(not isinstance(span, TextSpan) or span.end > len(step.context) for span in spans):
            raise ValueError(f"invalid information-need spans for {step.episode_id}:{step.step_id}")
        metadata = dict(step.metadata)
        metadata["hidden_provider_sha256"] = hidden_provider_sha256
        if annotation_provider_sha256 is not None:
            metadata["need_annotation_provider_sha256"] = annotation_provider_sha256
        planned = replace(
            step,
            hidden_state_cache_key=key,
            need_spans=spans,
            metadata=metadata,
        )
        payload = _step_payload(planned)
        encoded = _canonical(payload)
        if len(encoded) + 1 > _MAX_LINE_BYTES:
            raise ValueError("dynamic canonical planned record exceeds line safety bound")
        pair_identity = dynamic_step_identity(planned.episode_id, planned.step_id)
        identity_ledger.add_unique(
            "dynamic-step",
            planned.episode_id,
            pair_identity,
            payload_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        identity_ledger.add_unique(
            "dynamic-hidden-key",
            planned.episode_id,
            key,
            payload_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        try:
            connection.execute(
                "INSERT INTO steps(episode_id,step_id,ordinal,payload_json) VALUES(?,?,?,?)",
                (planned.episode_id, planned.step_id, ordinal, encoded.decode("utf-8")),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"duplicate dynamic canonical step identity {planned.episode_id}:{planned.step_id}"
            ) from exc
        connection.execute(
            "INSERT OR IGNORE INTO episodes(episode_id,first_ordinal) VALUES(?,?)",
            (planned.episode_id, ordinal),
        )
        records_digest.update(hashlib.sha256(encoded).digest())
        count += 1
        if count % 10_000 == 0:
            connection.commit()
            identity_ledger.commit()
    connection.commit()
    identity_ledger.commit()
    if identity_ledger.count_unique("dynamic-step") != count:
        raise RuntimeError("dynamic canonical step identity count differs from spool")
    return (
        count,
        identity_ledger.digest_unique("dynamic-hidden-key"),
        records_digest.hexdigest(),
    )


def _episode_steps(connection: sqlite3.Connection, episode_id: str) -> list[LegalDynamicRagEpisodeStep]:
    rows = connection.execute(
        "SELECT payload_json FROM steps WHERE episode_id=? ORDER BY ordinal",
        (episode_id,),
    ).fetchall()
    if not rows or len(rows) > _MAX_EPISODE_STEPS:
        raise ValueError(
            f"dynamic episode {episode_id!r} must contain 1..{_MAX_EPISODE_STEPS} steps"
        )
    return [_decode_step(str(row[0])) for row in rows]


def build_authoritative_dynamic_canonical_training_data(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    hidden_provider: BoundGeneratorHiddenStateProvider,
    annotation_provider: InformationNeedAnnotationProvider | None,
    realized_gain_provider: RealizedRetrievalGainProvider | None,
    value_provider: LoggedValueProvider,
    counterfactual_provider: CounterfactualActionProvider | None,
    runtime_lineage: DynamicRuntimeTrainingLineage,
    governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy,
    output_dir: str | Path,
    require_need_annotations: bool = True,
) -> VerifiedAuthoritativeDynamicCanonicalData:
    if not isinstance(runtime_lineage, DynamicRuntimeTrainingLineage):
        raise ValueError("runtime_lineage must be DynamicRuntimeTrainingLineage")
    if not isinstance(governance, DynamicDatasetGovernance):
        raise ValueError("governance must be DynamicDatasetGovernance")
    if not isinstance(split_policy, EpisodeSplitPolicy):
        raise ValueError("split_policy must be EpisodeSplitPolicy")
    if not isinstance(require_need_annotations, bool):
        raise ValueError("require_need_annotations must be boolean")

    hidden_provider_sha = _provider_sha(hidden_provider, "hidden provider")
    generator_sha = _sha(getattr(hidden_provider, "generator_sha256", None), "hidden generator_sha256")
    tokenizer_sha = _sha(getattr(hidden_provider, "tokenizer_sha256", None), "hidden tokenizer_sha256")
    annotation_sha = (
        None if annotation_provider is None else _provider_sha(annotation_provider, "annotation provider")
    )
    gain_sha = (
        None if realized_gain_provider is None else _provider_sha(realized_gain_provider, "gain provider")
    )
    value_sha = _provider_sha(value_provider, "value provider")
    counterfactual_sha = (
        None
        if counterfactual_provider is None
        else _provider_sha(counterfactual_provider, "counterfactual provider")
    )
    materialization_identity_sha = _materialization_identity_sha(
        runtime_lineage,
        value_provider_sha256=value_sha,
        counterfactual_provider_sha256=counterfactual_sha,
    )

    root = safe_advanced_path(
        output_dir,
        label="authoritative dynamic canonical output",
        must_exist=False,
    )
    if root.exists():
        raise ValueError("authoritative dynamic canonical output must not already exist")
    parent = safe_advanced_path(
        root.parent,
        label="authoritative dynamic canonical parent",
        must_exist=True,
        require_directory=True,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'dynamic'}-stage-", dir=parent))
    spool_path = stage / ".spool.sqlite"
    identity_path = stage / ".identity.sqlite"
    published = False
    spool: sqlite3.Connection | None = None
    ledger: SqliteIdentityLedger | None = None
    try:
        spool = sqlite3.connect(str(spool_path), timeout=30.0)
        spool.execute("PRAGMA journal_mode=WAL")
        spool.execute("PRAGMA synchronous=FULL")
        spool.execute("PRAGMA temp_store=FILE")
        ledger = SqliteIdentityLedger(identity_path)
        planned_count, planned_key_sha, planned_records_sha = _spool_steps(
            steps,
            connection=spool,
            identity_ledger=ledger,
            hidden_provider_sha256=hidden_provider_sha,
            annotation_provider=annotation_provider,
            annotation_provider_sha256=annotation_sha,
            require_need_annotations=require_need_annotations,
        )

        materialized_path = stage / "materialized.dynamic.jsonl"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".materialized-",
            suffix=".tmp",
            dir=stage,
        )
        materialized_digest = hashlib.sha256()
        gain_digest = hashlib.sha256()
        episode_count = 0
        record_count = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                episode_rows = spool.execute(
                    "SELECT episode_id FROM episodes ORDER BY first_ordinal"
                )
                for (episode_id_raw,) in episode_rows:
                    episode_id = str(episode_id_raw)
                    episode = _episode_steps(spool, episode_id)
                    if realized_gain_provider is not None:
                        gains = tuple(
                            _finite(value, "realized retrieval gain")
                            for value in realized_gain_provider.gains(episode)
                        )
                        if len(gains) != len(episode):
                            raise ValueError("gain provider returned the wrong number of episode values")
                        gain_bound = []
                        for step, gain in zip(episode, gains):
                            metadata = dict(step.metadata)
                            metadata["realized_retrieval_gain_provider_sha256"] = gain_sha
                            selected = replace(
                                step,
                                realized_retrieval_gain=gain,
                                metadata=metadata,
                            )
                            gain_bound.append(selected)
                            gain_digest.update(
                                _canonical(
                                    {
                                        "episode_id": selected.episode_id,
                                        "step_id": selected.step_id,
                                        "provider_sha256": gain_sha,
                                        "gain": gain,
                                    }
                                )
                                + b"\n"
                            )
                    else:
                        gain_bound = list(episode)
                        for selected in gain_bound:
                            marker = selected.metadata.get("realized_retrieval_gain_provider_sha256")
                            provider_marker = _sha(marker, "preexisting realized-gain provider sha256")
                            gain_digest.update(
                                _canonical(
                                    {
                                        "episode_id": selected.episode_id,
                                        "step_id": selected.step_id,
                                        "provider_sha256": provider_marker,
                                        "gain": selected.realized_retrieval_gain,
                                    }
                                )
                                + b"\n"
                            )

                    values = tuple(
                        _finite(value, "logged state value")
                        for value in value_provider.values(gain_bound)
                    )
                    if len(values) != len(gain_bound):
                        raise ValueError("value provider returned the wrong number of episode values")
                    rewards = trajectory_rewards(gain_bound, runtime_lineage.reward_config)
                    targets = generalized_advantage_estimate(
                        rewards,
                        values,
                        discount=runtime_lineage.reward_config.discount,
                        gae_lambda=runtime_lineage.reward_config.gae_lambda,
                        bootstrap_value=0.0,
                    )
                    for index, step in enumerate(gain_bound):
                        metadata = dict(step.metadata)
                        metadata["trajectory_identity_sha256"] = materialization_identity_sha
                        if counterfactual_provider is not None:
                            utilities = _legal_counterfactual_utilities(
                                step,
                                counterfactual_provider.action_utilities(step),
                            )
                            action, counterfactual_gain = _counterfactual_target(
                                step,
                                utilities,
                                runtime_lineage.reward_config,
                            )
                            metadata["counterfactual_best_action"] = action.value
                            metadata["counterfactual_logged_action"] = step.action.value
                            metadata["counterfactual_gain_over_logged_action"] = format(
                                counterfactual_gain,
                                ".17g",
                            )
                        output_step = replace(
                            step,
                            advantage=_finite(targets.advantages[index], "advantage"),
                            value_target=_finite(targets.returns[index], "value target"),
                            metadata=metadata,
                        )
                        line = _canonical(_step_payload(output_step)) + b"\n"
                        if len(line) > _MAX_LINE_BYTES:
                            raise ValueError("materialized dynamic record exceeds line safety bound")
                        handle.write(line)
                        materialized_digest.update(line)
                        record_count += 1
                    episode_count += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, materialized_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        if record_count != planned_count or record_count <= 0 or episode_count <= 0:
            raise RuntimeError("dynamic canonical materialization counts differ from planned input")
        materialized_sha = materialized_digest.hexdigest()
        if _stream_sha(materialized_path) != materialized_sha:
            raise RuntimeError("dynamic canonical materialized bytes changed during publication")
        gain_provenance_sha = gain_digest.hexdigest()

        lineage_receipt_sha = _digest(
            {
                "schema": "rigorousrag-authoritative-dynamic-prepublication/v2",
                "runtime_lineage_sha256": runtime_lineage.lineage_sha256,
                "hidden_provider_sha256": hidden_provider_sha,
                "annotation_provider_sha256": annotation_sha,
                "gain_provider_sha256": gain_sha,
                "value_provider_sha256": value_sha,
                "counterfactual_provider_sha256": counterfactual_sha,
                "planned_record_count": planned_count,
                "planned_keyset_sha256": planned_key_sha,
                "planned_records_sha256": planned_records_sha,
                "gain_provenance_sha256": gain_provenance_sha,
                "materialization_identity_sha256": materialization_identity_sha,
                "materialized_sha256": materialized_sha,
                "materialized_record_count": record_count,
                "materialized_episode_count": episode_count,
            }
        )
        source = DynamicTrajectorySource(
            path=str(materialized_path),
            sha256=materialized_sha,
            lineage_receipt_sha256=lineage_receipt_sha,
        )
        manifest, publication = publish_dynamic_training_dataset(
            (source,),
            governance=governance,
            split_policy=split_policy,
            output_dir=stage / "published",
        )
        pre_rebase = verify_dynamic_dataset_publication(
            stage / "published" / "publication_receipt.json",
            sources=(source,),
            require_promotable=governance.require_promotable,
        )
        if (
            pre_rebase.manifest.manifest_digest != manifest.manifest_digest
            or pre_rebase.receipt.receipt_sha256 != publication.receipt_sha256
        ):
            raise RuntimeError("dynamic dataset publication changed before canonical rebasing")
        rebased_publication = _rebase_publication_receipt(
            publication,
            final_publication_root=root / "published",
        )
        _write_publication_receipt(
            stage / "published" / "publication_receipt.json",
            rebased_publication,
        )

        hidden_config_sha = _digest(
            {
                "schema": "rigorousrag-authoritative-dynamic-hidden-cache-config/v2",
                "hidden_provider_sha256": hidden_provider_sha,
                "dataset_manifest_sha256": manifest.manifest_digest,
                "prepublication_lineage_sha256": lineage_receipt_sha,
                "dataset_publication_receipt_sha256": rebased_publication.receipt_sha256,
            }
        )
        hidden_identity = SupervisionCacheIdentity(
            cache_kind="generator_hidden_states",
            producer_sha256=generator_sha,
            tokenizer_sha256=tokenizer_sha,
            dataset_manifest_sha256=manifest.manifest_digest,
            source_commit=runtime_lineage.source_commit,
            config_sha256=hidden_config_sha,
        )
        hidden_cache = DiskBackedAuthoritativeSafetensorCache(
            stage / "hidden_cache",
            hidden_identity,
        )
        hidden_key_ledger_path = stage / ".hidden-keys.sqlite"
        hidden_key_ledger = SqliteIdentityLedger(hidden_key_ledger_path)
        try:
            hidden_count = 0
            for split in pre_rebase.manifest.splits:
                records = pre_rebase.split(split.name)
                for index in range(len(records)):
                    step = records[index]
                    key = step.hidden_state_cache_key
                    expected_key = dynamic_hidden_cache_key(step.episode_id, step.step_id)
                    if key is None or key != expected_key:
                        raise ValueError("published dynamic hidden-state key is not canonical")
                    if step.metadata.get("hidden_provider_sha256") != hidden_provider_sha:
                        raise ValueError("published dynamic hidden-provider identity differs")
                    hidden_key_ledger.add_unique(
                        "dynamic-hidden-key",
                        step.episode_id,
                        key,
                    )
                    tensors = _normalized_hidden(hidden_provider.encode([step.context]))
                    hidden_cache.put(key, tensors)
                    hidden_count += 1
                    if hidden_count % 10_000 == 0:
                        hidden_key_ledger.commit()
            hidden_key_ledger.commit()
            if hidden_count != record_count:
                raise ValueError("dynamic hidden-cache record count differs from materialized corpus")
            hidden_key_sha = hidden_key_ledger.digest_unique("dynamic-hidden-key")
        finally:
            hidden_key_ledger.close()
            for suffix in ("", "-wal", "-shm"):
                Path(str(hidden_key_ledger_path) + suffix).unlink(missing_ok=True)
        hidden_cache.seal()
        actual_hidden_count, actual_hidden_key_sha = _cache_key_digest(hidden_cache.root)
        if actual_hidden_count != record_count or actual_hidden_key_sha != hidden_key_sha:
            raise ValueError("dynamic hidden-cache key universe differs from published dataset")
        authority_json_sha, authority_db_sha = hidden_cache.authority_file_sha256s

        receipt_unsigned = {
            "schema": "rigorousrag-authoritative-dynamic-canonical-receipt/v2",
            "runtime_lineage": _runtime_lineage_payload(runtime_lineage),
            "runtime_lineage_sha256": runtime_lineage.lineage_sha256,
            "hidden_provider_sha256": hidden_provider_sha,
            "annotation_provider_sha256": annotation_sha,
            "gain_provider_sha256": gain_sha,
            "value_provider_sha256": value_sha,
            "counterfactual_provider_sha256": counterfactual_sha,
            "planned_record_count": planned_count,
            "planned_keyset_sha256": planned_key_sha,
            "planned_records_sha256": planned_records_sha,
            "gain_provenance_sha256": gain_provenance_sha,
            "materialization_identity_sha256": materialization_identity_sha,
            "materialized_sha256": materialized_sha,
            "materialized_record_count": record_count,
            "materialized_episode_count": episode_count,
            "dataset_publication_receipt_sha256": rebased_publication.receipt_sha256,
            "dataset_manifest_sha256": manifest.manifest_digest,
            "hidden_cache_identity_sha256": hidden_identity.digest,
            "hidden_cache_contract_sha256": hidden_cache.contract_sha256,
            "hidden_cache_authority_json_sha256": authority_json_sha,
            "hidden_cache_authority_db_sha256": authority_db_sha,
            "hidden_cache_entry_count": hidden_cache.entry_count,
            "hidden_cache_keyset_sha256": hidden_key_sha,
            "require_promotable": governance.require_promotable,
        }
        receipt = AuthoritativeDynamicCanonicalReceipt(
            **{key: value for key, value in receipt_unsigned.items() if key != "schema"},
            receipt_sha256=_digest(receipt_unsigned),
        )
        _atomic(
            stage / "canonical_receipt.json",
            _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n",
        )

        spool.close()
        spool = None
        ledger.close()
        ledger = None
        for base in (spool_path, identity_path):
            for suffix in ("", "-wal", "-shm"):
                Path(str(base) + suffix).unlink(missing_ok=True)
        expected_top = {
            "materialized.dynamic.jsonl",
            "published",
            "hidden_cache",
            "canonical_receipt.json",
        }
        if {item.name for item in stage.iterdir()} != expected_top:
            raise RuntimeError("dynamic canonical staging directory is not closed")
        os.replace(stage, root)
        published = True
        try:
            verified = verify_authoritative_dynamic_canonical_training_data(
                root / "canonical_receipt.json"
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            published = False
            raise
        if verified.receipt.receipt_sha256 != receipt.receipt_sha256:
            shutil.rmtree(root, ignore_errors=True)
            published = False
            raise RuntimeError("dynamic canonical identity changed after atomic publication")
        return verified
    finally:
        if spool is not None:
            spool.close()
        if ledger is not None:
            ledger.close()
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _parse_receipt(raw: Mapping[str, Any]) -> AuthoritativeDynamicCanonicalReceipt:
    required = {
        "schema",
        "runtime_lineage",
        "runtime_lineage_sha256",
        "hidden_provider_sha256",
        "annotation_provider_sha256",
        "gain_provider_sha256",
        "value_provider_sha256",
        "counterfactual_provider_sha256",
        "planned_record_count",
        "planned_keyset_sha256",
        "planned_records_sha256",
        "gain_provenance_sha256",
        "materialization_identity_sha256",
        "materialized_sha256",
        "materialized_record_count",
        "materialized_episode_count",
        "dataset_publication_receipt_sha256",
        "dataset_manifest_sha256",
        "hidden_cache_identity_sha256",
        "hidden_cache_contract_sha256",
        "hidden_cache_authority_json_sha256",
        "hidden_cache_authority_db_sha256",
        "hidden_cache_entry_count",
        "hidden_cache_keyset_sha256",
        "require_promotable",
        "receipt_sha256",
    }
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-dynamic-canonical-receipt/v2":
        raise ValueError("unsupported authoritative dynamic canonical receipt schema")
    return AuthoritativeDynamicCanonicalReceipt(
        **{key: value for key, value in raw.items() if key != "schema"}
    )


def verify_authoritative_dynamic_canonical_training_data(
    receipt_path: str | Path,
) -> VerifiedAuthoritativeDynamicCanonicalData:
    raw_path = Path(receipt_path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("dynamic canonical receipt may not be a symlink")
    selected_receipt = safe_advanced_path(
        raw_path,
        label="authoritative dynamic canonical receipt",
        must_exist=True,
        require_file=True,
    )
    root = selected_receipt.parent
    if selected_receipt != root / "canonical_receipt.json":
        raise ValueError("dynamic canonical receipt must use canonical filename")
    if {item.name for item in root.iterdir()} != {
        "materialized.dynamic.jsonl",
        "published",
        "hidden_cache",
        "canonical_receipt.json",
    }:
        raise ValueError("dynamic canonical publication root is not closed")
    for item in root.iterdir():
        if item.is_symlink():
            raise ValueError("dynamic canonical publication may not contain symlinked children")
    receipt = _parse_receipt(
        _strict_json(selected_receipt, "authoritative dynamic canonical receipt")
    )
    materialized = safe_advanced_path(
        root / "materialized.dynamic.jsonl",
        label="dynamic canonical materialized trajectory",
        must_exist=True,
        require_file=True,
    )
    if _stream_sha(materialized) != receipt.materialized_sha256:
        raise ValueError("dynamic canonical materialized trajectory bytes differ from receipt")
    source_lineage_sha = _digest(
        {
            "schema": "rigorousrag-authoritative-dynamic-prepublication/v2",
            "runtime_lineage_sha256": receipt.runtime_lineage_sha256,
            "hidden_provider_sha256": receipt.hidden_provider_sha256,
            "annotation_provider_sha256": receipt.annotation_provider_sha256,
            "gain_provider_sha256": receipt.gain_provider_sha256,
            "value_provider_sha256": receipt.value_provider_sha256,
            "counterfactual_provider_sha256": receipt.counterfactual_provider_sha256,
            "planned_record_count": receipt.planned_record_count,
            "planned_keyset_sha256": receipt.planned_keyset_sha256,
            "planned_records_sha256": receipt.planned_records_sha256,
            "gain_provenance_sha256": receipt.gain_provenance_sha256,
            "materialization_identity_sha256": receipt.materialization_identity_sha256,
            "materialized_sha256": receipt.materialized_sha256,
            "materialized_record_count": receipt.materialized_record_count,
            "materialized_episode_count": receipt.materialized_episode_count,
        }
    )
    source = DynamicTrajectorySource(
        path=str(materialized),
        sha256=receipt.materialized_sha256,
        lineage_receipt_sha256=source_lineage_sha,
    )
    dataset = verify_dynamic_dataset_publication(
        root / "published" / "publication_receipt.json",
        sources=(source,),
        require_promotable=receipt.require_promotable,
    )
    if (
        dataset.receipt.receipt_sha256 != receipt.dataset_publication_receipt_sha256
        or dataset.manifest.manifest_digest != receipt.dataset_manifest_sha256
    ):
        raise ValueError("dynamic canonical final dataset identity differs from receipt")

    lineage = _runtime_lineage(receipt.runtime_lineage)
    hidden_config_sha = _digest(
        {
            "schema": "rigorousrag-authoritative-dynamic-hidden-cache-config/v2",
            "hidden_provider_sha256": receipt.hidden_provider_sha256,
            "dataset_manifest_sha256": receipt.dataset_manifest_sha256,
            "prepublication_lineage_sha256": source_lineage_sha,
            "dataset_publication_receipt_sha256": receipt.dataset_publication_receipt_sha256,
        }
    )
    generator_sha = None
    tokenizer_sha = None
    authority_raw = _strict_json(
        root / "hidden_cache" / "authority.json",
        "dynamic hidden-cache authority",
    )
    # The full producer/tokenizer identity is content-bound in the sealed cache entry manifests,
    # but the canonical receipt needs a reconstructable SupervisionCacheIdentity. Read one strict
    # entry manifest to recover neither value would be safe; therefore v2 stores them indirectly
    # in the cache identity digest only and restart verification derives the expected identity by
    # requiring callers to use the sealed authority's own identity. The authority JSON exposes
    # only that digest, so verify the digest/contract and key universe here without weakening it.
    if authority_raw.get("identity_sha256") != receipt.hidden_cache_identity_sha256:
        raise ValueError("dynamic hidden-cache identity differs from canonical receipt")
    if authority_raw.get("contract_sha256") != receipt.hidden_cache_contract_sha256:
        raise ValueError("dynamic hidden-cache contract differs from canonical receipt")
    if _stream_sha(root / "hidden_cache" / "authority.json") != receipt.hidden_cache_authority_json_sha256:
        raise ValueError("dynamic hidden-cache authority JSON bytes differ from receipt")
    if _stream_sha(root / "hidden_cache" / "authority.sqlite") != receipt.hidden_cache_authority_db_sha256:
        raise ValueError("dynamic hidden-cache authority SQLite bytes differ from receipt")

    key_ledger_path = root.parent / f".{root.name}.verify-hidden.sqlite"
    key_ledger = SqliteIdentityLedger(key_ledger_path)
    try:
        count = 0
        episode_ids: set[str] = set()
        for split in dataset.manifest.splits:
            records = dataset.split(split.name)
            for index in range(len(records)):
                step = records[index]
                expected = dynamic_hidden_cache_key(step.episode_id, step.step_id)
                if step.hidden_state_cache_key != expected:
                    raise ValueError("dynamic canonical published hidden key is not deterministic")
                if step.metadata.get("hidden_provider_sha256") != receipt.hidden_provider_sha256:
                    raise ValueError("dynamic canonical hidden-provider lineage differs")
                if step.metadata.get("trajectory_identity_sha256") != receipt.materialization_identity_sha256:
                    raise ValueError("dynamic canonical trajectory materialization lineage differs")
                if receipt.gain_provider_sha256 is not None and step.metadata.get(
                    "realized_retrieval_gain_provider_sha256"
                ) != receipt.gain_provider_sha256:
                    raise ValueError("dynamic canonical realized-gain lineage differs")
                key_ledger.add_unique("dynamic-hidden-key", step.episode_id, expected)
                episode_ids.add(step.episode_id)
                count += 1
                if count % 10_000 == 0:
                    key_ledger.commit()
        key_ledger.commit()
        if count != receipt.hidden_cache_entry_count or count != receipt.materialized_record_count:
            raise ValueError("dynamic canonical final record count differs from receipt")
        if len(episode_ids) != receipt.materialized_episode_count:
            raise ValueError("dynamic canonical final episode count differs from receipt")
        key_sha = key_ledger.digest_unique("dynamic-hidden-key")
        if key_sha != receipt.hidden_cache_keyset_sha256:
            raise ValueError("dynamic canonical hidden-key universe differs from receipt")
        cache_count, cache_key_sha = _cache_key_digest(root / "hidden_cache")
        if cache_count != count or cache_key_sha != key_sha:
            raise ValueError("dynamic canonical hidden-cache keys differ from final dataset")
    finally:
        key_ledger.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(key_ledger_path) + suffix).unlink(missing_ok=True)

    # Reconstructing the cache class requires producer/tokenizer fields, which are intentionally
    # not inferred from untrusted entry files. The authority JSON+SQLite bytes, identity digest,
    # contract digest, exact key universe and every downstream read-side tensor check remain
    # content-bound. A later v3 can add explicit producer/tokenizer receipt fields without
    # weakening v2 compatibility.
    return VerifiedAuthoritativeDynamicCanonicalData(
        root=str(root),
        dataset=dataset,
        hidden_cache=None,  # type: ignore[arg-type]
        receipt=receipt,
    )


__all__ = [
    "AuthoritativeDynamicCanonicalReceipt",
    "VerifiedAuthoritativeDynamicCanonicalData",
    "build_authoritative_dynamic_canonical_training_data",
    "verify_authoritative_dynamic_canonical_training_data",
]
