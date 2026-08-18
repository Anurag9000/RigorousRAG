"""Immutable supervised-data manifests from resolved active-learning adjudication gold.

Gold export is a new derived artifact. It never rewrites the acquisition batch or review
history. Each training row is bound to the exact active-learning case mapping and the
latest immutable adjudication resolution supplied by ExpertAdjudicationStore.export_gold.
Task-specific binary mappings are explicit inputs rather than inferred from label names.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from evaluation.expert_adjudication import GoldLabel
from orchestration.active_learning_adjudication import ActiveLearningMaterializationReceipt
from tools.cross_profile_fusion import ScoreCalibrationExample


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _weight(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("weight must be finite and positive")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError("weight must be finite and positive")
    return selected


@dataclass(frozen=True)
class ActiveLearningGoldExample:
    task_id: str
    item_sha256: str
    case_id: str
    label: str
    round_index: int
    resolution_revision: int
    resolution_digest: str
    candidate_sha256: str
    route_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "item_sha256", _sha(self.item_sha256, "item_sha256"))
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id", 1000))
        object.__setattr__(self, "label", _text(self.label, "label", 1000))
        for name in ("round_index", "resolution_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("resolution_digest", "candidate_sha256", "route_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))

    @property
    def example_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-active-learning-gold-example/v1", **asdict(self)})


@dataclass(frozen=True)
class ActiveLearningGoldManifest:
    owner_id: str
    label_contract_sha256: str
    materialization_receipt_sha256s: tuple[str, ...]
    examples: tuple[ActiveLearningGoldExample, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "label_contract_sha256", _sha(self.label_contract_sha256, "label_contract_sha256"))
        receipts = tuple(sorted(_sha(value, "materialization receipt sha256") for value in self.materialization_receipt_sha256s))
        if not receipts or len(set(receipts)) != len(receipts):
            raise ValueError("materialization receipt identities must be unique and non-empty")
        object.__setattr__(self, "materialization_receipt_sha256s", receipts)
        examples = tuple(sorted(self.examples, key=lambda row: (row.task_id, row.item_sha256, row.case_id)))
        if not examples:
            raise ValueError("gold manifest requires at least one resolved example")
        if len({row.case_id for row in examples}) != len(examples) or len({(row.task_id, row.item_sha256) for row in examples}) != len(examples):
            raise ValueError("gold manifest examples must be unique by case and task/item")
        object.__setattr__(self, "examples", examples)
        expected = _digest(self._payload())
        provided = _sha(self.manifest_sha256, "manifest_sha256")
        if expected != provided:
            raise ValueError("manifest_sha256 does not match active-learning gold manifest")
        object.__setattr__(self, "manifest_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-active-learning-gold-manifest/v1",
            "owner_id": self.owner_id,
            "label_contract_sha256": self.label_contract_sha256,
            "materialization_receipt_sha256s": self.materialization_receipt_sha256s,
            "examples": [asdict(row) for row in self.examples],
        }


def build_active_learning_gold_manifest(
    *,
    owner_id: str,
    gold_labels: Sequence[GoldLabel],
    materializations: Sequence[ActiveLearningMaterializationReceipt],
    label_contract_sha256: str,
    require_all_materialized_resolved: bool = False,
) -> ActiveLearningGoldManifest:
    owner = _text(owner_id, "owner_id")
    labels = tuple(gold_labels)
    receipts = tuple(materializations)
    if not receipts or any(not isinstance(receipt, ActiveLearningMaterializationReceipt) for receipt in receipts):
        raise ValueError("materializations must be a non-empty receipt collection")
    if any(receipt.owner_id != owner for receipt in receipts):
        raise ValueError("materialization owner differs from gold manifest owner")
    case_map = {}
    for receipt in receipts:
        for row in receipt.cases:
            previous = case_map.get(row.case_id)
            identity = (row.task_id, row.item_sha256, row.candidate_sha256, row.route_sha256)
            if previous is not None and previous != identity:
                raise ValueError("case id maps to conflicting active-learning identities")
            case_map[row.case_id] = identity
    if not case_map:
        raise ValueError("materializations contain no expert cases")
    label_map = {}
    for gold in labels:
        if not isinstance(gold, GoldLabel):
            raise ValueError("gold_labels contains an invalid value")
        if gold.case_id in label_map:
            raise ValueError("gold_labels contains duplicate case ids")
        label_map[gold.case_id] = gold
    unknown = set(label_map) - set(case_map)
    if unknown:
        raise ValueError("gold export contains cases outside the active-learning materializations")
    if require_all_materialized_resolved and set(case_map) != set(label_map):
        raise ValueError("not every materialized active-learning case has resolved gold")
    examples = []
    for case_id in sorted(label_map):
        gold = label_map[case_id]
        task_id, item_sha256, candidate_sha256, route_sha256 = case_map[case_id]
        if gold.item_sha256 != item_sha256:
            raise ValueError("gold item identity differs from active-learning case mapping")
        examples.append(
            ActiveLearningGoldExample(
                task_id=task_id,
                item_sha256=item_sha256,
                case_id=case_id,
                label=gold.label,
                round_index=gold.round_index,
                resolution_revision=gold.resolution_revision,
                resolution_digest=gold.resolution_digest,
                candidate_sha256=candidate_sha256,
                route_sha256=route_sha256,
            )
        )
    payload = {
        "schema": "rigorousrag-active-learning-gold-manifest/v1",
        "owner_id": owner,
        "label_contract_sha256": _sha(label_contract_sha256, "label_contract_sha256"),
        "materialization_receipt_sha256s": tuple(sorted(receipt.receipt_sha256 for receipt in receipts)),
        "examples": [asdict(row) for row in sorted(examples, key=lambda row: (row.task_id, row.item_sha256, row.case_id))],
    }
    return ActiveLearningGoldManifest(**payload, manifest_sha256=_digest(payload))


@dataclass(frozen=True)
class BinaryLabelMapping:
    task_id: str
    positive_labels: tuple[str, ...]
    negative_labels: tuple[str, ...]
    mapping_sha256: str

    @classmethod
    def build(cls, *, task_id: str, positive_labels: Iterable[str], negative_labels: Iterable[str]) -> "BinaryLabelMapping":
        task = _text(task_id, "task_id")
        positive = tuple(sorted({_text(value, "positive label", 1000) for value in positive_labels}))
        negative = tuple(sorted({_text(value, "negative label", 1000) for value in negative_labels}))
        if not positive or not negative or set(positive) & set(negative):
            raise ValueError("binary label mapping requires disjoint non-empty positive and negative label sets")
        payload = {"schema": "rigorousrag-active-learning-binary-label-map/v1", "task_id": task, "positive_labels": positive, "negative_labels": negative}
        return cls(task, positive, negative, _digest(payload))

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        positive = tuple(sorted(_text(value, "positive label", 1000) for value in self.positive_labels))
        negative = tuple(sorted(_text(value, "negative label", 1000) for value in self.negative_labels))
        if not positive or not negative or set(positive) & set(negative):
            raise ValueError("binary label mapping is invalid")
        object.__setattr__(self, "positive_labels", positive)
        object.__setattr__(self, "negative_labels", negative)
        expected = _digest({"schema": "rigorousrag-active-learning-binary-label-map/v1", "task_id": self.task_id, "positive_labels": positive, "negative_labels": negative})
        provided = _sha(self.mapping_sha256, "mapping_sha256")
        if expected != provided:
            raise ValueError("mapping_sha256 does not match binary label mapping")
        object.__setattr__(self, "mapping_sha256", provided)

    def target(self, label: str) -> bool | None:
        selected = _text(label, "label", 1000)
        if selected in self.positive_labels:
            return True
        if selected in self.negative_labels:
            return False
        return None


@dataclass(frozen=True)
class ScoredGoldItem:
    item_sha256: str
    raw_score: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_sha256", _sha(self.item_sha256, "item_sha256"))
        if isinstance(self.raw_score, bool):
            raise ValueError("raw_score must be finite")
        score = float(self.raw_score)
        if not math.isfinite(score):
            raise ValueError("raw_score must be finite")
        object.__setattr__(self, "raw_score", score)
        object.__setattr__(self, "weight", _weight(self.weight))


def calibration_examples_from_active_learning_gold(
    manifest: ActiveLearningGoldManifest,
    *,
    mapping: BinaryLabelMapping,
    scored_items: Sequence[ScoredGoldItem],
    require_complete_scores: bool = True,
) -> tuple[ScoreCalibrationExample, ...]:
    if not isinstance(manifest, ActiveLearningGoldManifest):
        raise ValueError("manifest must be ActiveLearningGoldManifest")
    if not isinstance(mapping, BinaryLabelMapping):
        raise ValueError("mapping must be BinaryLabelMapping")
    rows = tuple(scored_items)
    if any(not isinstance(row, ScoredGoldItem) for row in rows):
        raise ValueError("scored_items contains invalid values")
    if len({row.item_sha256 for row in rows}) != len(rows):
        raise ValueError("scored_items contains duplicate item identities")
    score_map = {row.item_sha256: row for row in rows}
    relevant_gold = [row for row in manifest.examples if row.task_id == mapping.task_id]
    if not relevant_gold:
        raise ValueError("gold manifest contains no examples for binary label mapping task")
    output = []
    missing = []
    for gold in relevant_gold:
        target = mapping.target(gold.label)
        if target is None:
            continue
        scored = score_map.get(gold.item_sha256)
        if scored is None:
            missing.append(gold.item_sha256)
            continue
        output.append(ScoreCalibrationExample(scored.raw_score, target, scored.weight))
    if require_complete_scores and missing:
        raise ValueError("one or more mapped gold examples have no bound raw score")
    if not output:
        raise ValueError("no active-learning gold examples could be converted to calibration examples")
    return tuple(output)


__all__ = [
    "ActiveLearningGoldExample",
    "ActiveLearningGoldManifest",
    "BinaryLabelMapping",
    "ScoredGoldItem",
    "build_active_learning_gold_manifest",
    "calibration_examples_from_active_learning_gold",
]
