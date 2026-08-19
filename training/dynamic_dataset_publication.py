"""Publish materialized dynamic-RAG trajectories as governed train/validation datasets.

This is the authoritative whole-episode publication layer used by the canonical dynamic-RAG
training pipeline.  Logical split names remain semantic manifest identifiers and never become
filesystem components.  Record identities use the collision-resistant canonical episode/step
pair identity, while a disk-backed SQLite ledger proves duplicate-step and episode-isolation
invariants without retaining corpus-sized Python sets/lists.

Publication is staged in a sibling directory and renamed into place only after every split,
manifest and receipt has been written and verified.  Importing this module performs no model
execution, network access, dataset download or training.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import (
    DatasetCard,
    DatasetManifest,
    DatasetModality,
    DatasetTask,
    LicenseStatus,
    SplitManifest,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import (
    LegalDynamicRagEpisodeStep,
    parse_authoritative_dynamic_step,
)
from training.dynamic_record_identity import dynamic_step_identity, dynamic_step_pair
from training.logical_filename import logical_filename

_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_SOURCES = 1_000
_MAX_SPLITS = 20
_HEX = frozenset("0123456789abcdef")


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


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid")
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


def _atomic(path: Path, payload: bytes) -> None:
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


def _strict_record(raw: bytes, label: str) -> LegalDynamicRagEpisodeStep:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    return parse_authoritative_dynamic_step(value)


def _sorted_digest(
    connection: sqlite3.Connection,
    *,
    table: str,
    split_name: str,
) -> str:
    if table == "steps":
        query = (
            "SELECT step_identity FROM steps WHERE split_name=? "
            "ORDER BY step_identity COLLATE BINARY"
        )
    elif table == "episodes":
        query = (
            "SELECT episode_id FROM episodes WHERE split_name=? "
            "ORDER BY episode_id COLLATE BINARY"
        )
    else:
        raise ValueError("unsupported dynamic identity table")
    digest = hashlib.sha256()
    for (value,) in connection.execute(query, (split_name,)):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class DynamicTrajectorySource:
    path: str
    sha256: str
    lineage_receipt_sha256: str

    def __post_init__(self) -> None:
        source = safe_advanced_path(
            self.path,
            label="dynamic trajectory source",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "path", str(source))
        object.__setattr__(self, "sha256", _sha(self.sha256, "trajectory source sha256"))
        object.__setattr__(
            self,
            "lineage_receipt_sha256",
            _sha(self.lineage_receipt_sha256, "trajectory lineage receipt sha256"),
        )
        if _stream_sha(source) != self.sha256:
            raise ValueError(
                "dynamic trajectory source bytes differ from configured SHA-256"
            )


@dataclass(frozen=True)
class EpisodeSplitPolicy:
    seed: str
    weights: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _identifier(self.seed, "split seed", 1_000))
        if not isinstance(self.weights, Mapping):
            raise ValueError("split weights must be a mapping")
        normalized: dict[str, int] = {}
        for name, raw in self.weights.items():
            split = _identifier(name, "split name", 100)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int)
                or raw <= 0
            ):
                raise ValueError("split weights must be positive integers")
            if split in normalized:
                raise ValueError("split names must be unique after normalization")
            normalized[split] = raw
        if not 2 <= len(normalized) <= _MAX_SPLITS:
            raise ValueError(f"split policy requires 2..{_MAX_SPLITS} unique splits")
        object.__setattr__(self, "weights", normalized)

    @property
    def policy_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-dynamic-episode-split-policy/v1",
                "seed": self.seed,
                "weights": dict(sorted(self.weights.items())),
            }
        )

    def split_for(self, episode_id: str) -> str:
        episode = _identifier(episode_id, "episode_id", 2_000)
        total = sum(self.weights.values())
        bucket = int(
            hashlib.sha256(f"{self.seed}\n{episode}".encode("utf-8")).hexdigest()[:16],
            16,
        ) % total
        cursor = 0
        for name, weight in sorted(self.weights.items()):
            cursor += weight
            if bucket < cursor:
                return name
        raise RuntimeError("episode split policy failed to select a split")


@dataclass(frozen=True)
class DynamicDatasetGovernance:
    dataset_id: str
    exact_version: str
    source_locator: str
    license_identifier: str
    license_status: LicenseStatus
    license_evidence: str
    card: DatasetCard
    metadata: Mapping[str, str] = field(default_factory=dict)
    require_promotable: bool = False

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "exact_version",
            "source_locator",
            "license_identifier",
            "license_evidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} is required")
        if not isinstance(self.license_status, LicenseStatus):
            object.__setattr__(
                self,
                "license_status",
                LicenseStatus(self.license_status),
            )
        if not isinstance(self.card, DatasetCard):
            raise ValueError("card must be DatasetCard")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if not isinstance(self.require_promotable, bool):
            raise ValueError("require_promotable must be boolean")
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )


@dataclass(frozen=True)
class PublishedDynamicSplit:
    name: str
    path: str
    sha256: str
    record_count: int
    record_id_sha256: str
    episode_id_sha256: str


@dataclass(frozen=True)
class DynamicDatasetPublicationReceipt:
    dataset_manifest_sha256: str
    source_set_sha256: str
    transformation_sha256: str
    split_policy_sha256: str
    manifest_path: str
    splits: tuple[PublishedDynamicSplit, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "source_set_sha256",
            "transformation_sha256",
            "split_policy_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not self.splits:
            raise ValueError("publication receipt requires split records")
        if len({item.name for item in self.splits}) != len(self.splits):
            raise ValueError("publication receipt split names must be unique")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("dynamic dataset publication receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-dataset-publication-receipt/v1",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "source_set_sha256": self.source_set_sha256,
            "transformation_sha256": self.transformation_sha256,
            "split_policy_sha256": self.split_policy_sha256,
            "manifest_path": self.manifest_path,
            "splits": [asdict(item) for item in self.splits],
        }


def _payload(
    step: LegalDynamicRagEpisodeStep,
    *,
    source_sha256: str,
    lineage_sha256: str,
) -> Mapping[str, Any]:
    metadata = dict(step.metadata)
    metadata["publication_source_sha256"] = source_sha256
    metadata["trajectory_lineage_receipt_sha256"] = lineage_sha256
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
        "metadata": metadata,
        "valid_actions": [action.value for action in step.valid_actions],
        "value_target": step.value_target,
    }


def _open_ledger(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE episodes ("
        "episode_id TEXT PRIMARY KEY, split_name TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE steps ("
        "episode_id TEXT NOT NULL, step_id TEXT NOT NULL, "
        "step_identity TEXT NOT NULL UNIQUE, split_name TEXT NOT NULL, "
        "PRIMARY KEY(episode_id,step_id)) WITHOUT ROWID"
    )
    return connection


def _record_identity(
    connection: sqlite3.Connection,
    *,
    episode_id: str,
    step_id: str,
    split_name: str,
) -> str:
    episode, step = dynamic_step_pair(episode_id, step_id)
    identity = dynamic_step_identity(episode, step)
    previous = connection.execute(
        "SELECT split_name FROM episodes WHERE episode_id=?",
        (episode,),
    ).fetchone()
    if previous is None:
        connection.execute(
            "INSERT INTO episodes(episode_id,split_name) VALUES (?,?)",
            (episode, split_name),
        )
    elif str(previous[0]) != split_name:
        raise RuntimeError("episode split policy changed within one publication")
    try:
        connection.execute(
            "INSERT INTO steps(episode_id,step_id,step_identity,split_name) "
            "VALUES (?,?,?,?)",
            (episode, step, identity, split_name),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"duplicate dynamic step identity {(episode, step)!r}"
        ) from exc
    return identity


def _closed_stage(stage: Path, splits: Sequence[PublishedDynamicSplit]) -> None:
    expected = {Path(item.path).name for item in splits} | {
        "dataset_manifest.json",
        "publication_receipt.json",
    }
    actual = {item.name for item in stage.iterdir()}
    if actual != expected:
        raise RuntimeError(
            "dynamic publication directory is not closed: "
            f"unexpected={sorted(actual-expected)} missing={sorted(expected-actual)}"
        )
    for item in stage.iterdir():
        if item.is_symlink() or not item.is_file():
            raise RuntimeError("dynamic publication contains a non-regular child")


def publish_dynamic_training_dataset(
    sources: Sequence[DynamicTrajectorySource],
    *,
    governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy,
    output_dir: str | Path,
) -> tuple[DatasetManifest, DynamicDatasetPublicationReceipt]:
    selected = tuple(sources)
    if (
        not selected
        or len(selected) > _MAX_SOURCES
        or any(not isinstance(item, DynamicTrajectorySource) for item in selected)
    ):
        raise ValueError(
            f"sources must be a bounded non-empty DynamicTrajectorySource sequence (max {_MAX_SOURCES})"
        )
    if not isinstance(governance, DynamicDatasetGovernance):
        raise ValueError("governance has incorrect type")
    if not isinstance(split_policy, EpisodeSplitPolicy):
        raise ValueError("split_policy has incorrect type")

    root = safe_advanced_path(
        output_dir,
        label="dynamic dataset publication output",
        must_exist=False,
    )
    if root.exists():
        if not root.is_dir():
            raise ValueError("dynamic dataset publication output must be a directory")
        if any(root.iterdir()):
            raise ValueError("dynamic dataset publication output must be empty")
        root.rmdir()
    parent = safe_advanced_path(
        root.parent,
        label="dynamic dataset publication parent",
        must_exist=True,
        require_directory=True,
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name or 'dynamic'}-stage-", dir=parent)
    )
    ledger_path = stage / ".identity-ledger.sqlite3"
    ledger: sqlite3.Connection | None = None
    handles: dict[str, Any] = {}
    digests: dict[str, Any] = {}
    counts: dict[str, int] = {}
    destinations: dict[str, Path] = {}
    try:
        ledger = _open_ledger(ledger_path)
        split_names = tuple(sorted(split_policy.weights))
        for name in split_names:
            destination = stage / logical_filename(name, ".dynamic.jsonl")
            destinations[name] = destination
            handles[name] = destination.open("xb")
            digests[name] = hashlib.sha256()
            counts[name] = 0

        total = 0
        for source in selected:
            with Path(source.path).open("rb") as stream:
                for line_number, raw in enumerate(stream, start=1):
                    if not raw.strip():
                        continue
                    if len(raw) > _MAX_LINE_BYTES:
                        raise ValueError(
                            f"trajectory line {line_number} exceeds safety bound"
                        )
                    if total >= _MAX_RECORDS:
                        raise ValueError("dynamic publication exceeds record safety bound")
                    step = _strict_record(raw, f"trajectory line {line_number}")
                    split = split_policy.split_for(step.episode_id)
                    _record_identity(
                        ledger,
                        episode_id=step.episode_id,
                        step_id=step.step_id,
                        split_name=split,
                    )
                    encoded = _canonical(
                        _payload(
                            step,
                            source_sha256=source.sha256,
                            lineage_sha256=source.lineage_receipt_sha256,
                        )
                    ) + b"\n"
                    if len(encoded) > _MAX_LINE_BYTES:
                        raise ValueError("canonical dynamic publication row exceeds safety bound")
                    _strict_record(encoded, "canonical dynamic publication row")
                    handles[split].write(encoded)
                    digests[split].update(encoded)
                    counts[split] += 1
                    total += 1
                    if total % 10_000 == 0:
                        ledger.commit()

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        ledger.commit()
        if total <= 0:
            raise ValueError("dynamic publication requires at least one record")
        if any(counts[name] <= 0 for name in split_names):
            raise ValueError(
                f"deterministic episode split produced an empty split: {counts}"
            )

        staged_splits: list[PublishedDynamicSplit] = []
        for name in split_names:
            path = destinations[name]
            split_sha = digests[name].hexdigest()
            if _stream_sha(path) != split_sha:
                raise RuntimeError("dynamic split changed during staged publication")
            staged_splits.append(
                PublishedDynamicSplit(
                    name=name,
                    path=str(path),
                    sha256=split_sha,
                    record_count=counts[name],
                    record_id_sha256=_sorted_digest(
                        ledger,
                        table="steps",
                        split_name=name,
                    ),
                    episode_id_sha256=_sorted_digest(
                        ledger,
                        table="episodes",
                        split_name=name,
                    ),
                )
            )

        # Global episode ownership is represented by one PRIMARY KEY row per episode. Prove the
        # ledger has exactly the expected total number of distinct episode assignments.
        assigned = int(ledger.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        if assigned <= 0:
            raise RuntimeError("dynamic episode ledger is unexpectedly empty")
        ledger.close()
        ledger = None
        ledger_path.unlink()

        source_set = _digest(
            {
                "schema": "rigorousrag-dynamic-trajectory-source-set/v1",
                "sources": [
                    {
                        "sha256": item.sha256,
                        "lineage_receipt_sha256": item.lineage_receipt_sha256,
                    }
                    for item in selected
                ],
            }
        )
        transformation = _digest(
            {
                "schema": "rigorousrag-dynamic-dataset-publication/v2",
                "source_set_sha256": source_set,
                "split_policy_sha256": split_policy.policy_sha256,
                "filename_policy": "sha256(logical_name)+fixed_extension",
                "record_identity_policy": "dynamic-step:sha256(canonical_json_pair)",
                "episode_isolation_policy": "sqlite_global_episode_owner",
                "publication_policy": "staged_closed_directory_rename",
            }
        )
        manifest = DatasetManifest(
            dataset_id=governance.dataset_id,
            exact_version=governance.exact_version,
            source_locator=governance.source_locator,
            artifact_sha256=source_set,
            license_identifier=governance.license_identifier,
            license_status=governance.license_status,
            license_evidence=governance.license_evidence,
            loader_name="training.dynamic_dataset_publication",
            loader_version="2",
            transformation_sha256=transformation,
            splits=tuple(
                SplitManifest(
                    name=item.name,
                    content_sha256=item.sha256,
                    record_count=item.record_count,
                    record_id_sha256=item.record_id_sha256,
                    source_group_sha256=item.episode_id_sha256,
                )
                for item in staged_splits
            ),
            tasks=(DatasetTask.DOMAIN_SPECIFIC,),
            modalities=(DatasetModality.TEXT,),
            card=governance.card,
            metadata={
                **governance.metadata,
                "canonical_record_kind": "dynamic_rag_episode",
                "episode_split_policy_sha256": split_policy.policy_sha256,
                "publication_authority": "dynamic_dataset_publication/v2",
                "filename_policy": "sha256(logical_name)+fixed_extension",
                "record_identity_policy": "dynamic-step:sha256(canonical_json_pair)",
            },
        )
        if governance.require_promotable:
            manifest.assert_promotable()

        final_splits = tuple(
            PublishedDynamicSplit(
                item.name,
                str(root / Path(item.path).name),
                item.sha256,
                item.record_count,
                item.record_id_sha256,
                item.episode_id_sha256,
            )
            for item in staged_splits
        )
        manifest_path = root / "dataset_manifest.json"
        _atomic(
            stage / "dataset_manifest.json",
            _canonical(
                {
                    "schema": "rigorousrag-dataset-manifest/v1",
                    "manifest": asdict(manifest),
                    "manifest_sha256": manifest.manifest_digest,
                }
            )
            + b"\n",
        )
        unsigned = {
            "schema": "rigorousrag-dynamic-dataset-publication-receipt/v1",
            "dataset_manifest_sha256": manifest.manifest_digest,
            "source_set_sha256": source_set,
            "transformation_sha256": transformation,
            "split_policy_sha256": split_policy.policy_sha256,
            "manifest_path": str(manifest_path),
            "splits": [asdict(item) for item in final_splits],
        }
        receipt = DynamicDatasetPublicationReceipt(
            manifest.manifest_digest,
            source_set,
            transformation,
            split_policy.policy_sha256,
            str(manifest_path),
            final_splits,
            _digest(unsigned),
        )
        _atomic(
            stage / "publication_receipt.json",
            _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n",
        )
        _closed_stage(stage, staged_splits)
        os.replace(stage, root)

        # Re-hash every split after the final atomic directory publication.
        for item in final_splits:
            path = safe_advanced_path(
                item.path,
                label=f"published dynamic split {item.name}",
                must_exist=True,
                require_file=True,
            )
            if _stream_sha(path) != item.sha256:
                raise RuntimeError(
                    f"dynamic split {item.name!r} changed during final publication"
                )
        return manifest, receipt
    except Exception:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        if ledger is not None:
            ledger.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "DynamicDatasetGovernance",
    "DynamicDatasetPublicationReceipt",
    "DynamicTrajectorySource",
    "EpisodeSplitPolicy",
    "PublishedDynamicSplit",
    "publish_dynamic_training_dataset",
]
