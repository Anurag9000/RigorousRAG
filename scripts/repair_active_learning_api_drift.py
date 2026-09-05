#!/usr/bin/env python3
"""One-shot, fail-closed repair for active-learning/authority API and artifact drift.

Every textual precondition must match exactly before files are changed. CI runs the
focused suites before the repaired tree may be committed. Delete this script and its
write-capable workflow after the verified repair lands.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_count(path: str, old: str, new: str, *, expected: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise RuntimeError(
            f"{path}: expected exactly {expected} repair anchor(s), found {count}: {old!r}"
        )
    target.write_text(text.replace(old, new, expected), encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    replace_count(path, old, new, expected=1)


def main() -> int:
    # Reconcile the active-learning route with the current adjudication API.
    replace_once(
        "orchestration/active_learning_adjudication.py",
        "from evaluation.expert_adjudication import CasePolicy, ExpertAdjudicationStore, LabelSchema",
        "from evaluation.expert_adjudication import AdjudicationPolicy, ExpertAdjudicationStore, LabelSchema",
    )
    replace_once(
        "orchestration/active_learning_adjudication.py",
        "    policy: CasePolicy = CasePolicy()",
        "    policy: AdjudicationPolicy = AdjudicationPolicy()",
    )
    replace_once(
        "orchestration/active_learning_adjudication.py",
        "        if not isinstance(self.policy, CasePolicy):\n            raise ValueError(\"policy must be CasePolicy\")",
        "        if not isinstance(self.policy, AdjudicationPolicy):\n            raise ValueError(\"policy must be AdjudicationPolicy\")",
    )
    replace_once(
        "orchestration/active_learning_adjudication.py",
        "            schema=route.schema,\n            policy=route.policy,\n            now=timestamp,",
        "            schema=route.schema,\n            now=timestamp,",
    )
    replace_once(
        "tests/unit/test_active_learning.py",
        "from evaluation.expert_adjudication import CasePolicy, ExpertAdjudicationStore, LabelSchema",
        "from evaluation.expert_adjudication import AdjudicationPolicy, ExpertAdjudicationStore, LabelSchema",
    )
    replace_once(
        "tests/unit/test_active_learning.py",
        "        CasePolicy(min_independent_reviews=2, require_adjudicator_on_disagreement=True),",
        "        AdjudicationPolicy(minimum_independent_reviews=2),",
    )

    # Restore the compact immutable resolved-gold lineage value used by the
    # active-learning derived-dataset builder. This is deliberately separate from
    # the store's richer governed GoldLabelRecord.
    replace_once(
        "evaluation/expert_adjudication.py",
        "@dataclass(frozen=True)\nclass GoldLabelRecord:\n",
        '''@dataclass(frozen=True)\nclass GoldLabel:\n    \"\"\"Compact resolved-label lineage used by active-learning derived datasets.\"\"\"\n\n    case_id: str\n    item_sha256: str\n    round_index: int\n    label: str\n    resolution_revision: int\n    resolution_digest: str\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, \"case_id\", _text(self.case_id, \"case_id\", 1000))\n        object.__setattr__(self, \"item_sha256\", _sha(self.item_sha256, \"item_sha256\"))\n        object.__setattr__(self, \"label\", _text(self.label, \"label\", 300))\n        for name in (\"round_index\", \"resolution_revision\"):\n            value = getattr(self, name)\n            if isinstance(value, bool) or not isinstance(value, int) or value < 0:\n                raise ValueError(f\"{name} must be non-negative\")\n        object.__setattr__(\n            self, \"resolution_digest\", _sha(self.resolution_digest, \"resolution_digest\")\n        )\n\n\n@dataclass(frozen=True)\nclass GoldLabelRecord:\n''',
    )
    replace_once(
        "evaluation/expert_adjudication.py",
        '__all__ = ["AdjudicationCase", "AdjudicationPolicy", "CaseRecord", "ExpertAdjudicationStore", "ExpertJudgment", "GoldLabelManifest", "GoldLabelRecord", "LabelSchema", "ResolutionReceipt", "ReviewClaim", "write_gold_manifest"]',
        '__all__ = ["AdjudicationCase", "AdjudicationPolicy", "CaseRecord", "ExpertAdjudicationStore", "ExpertJudgment", "GoldLabel", "GoldLabelManifest", "GoldLabelRecord", "LabelSchema", "ResolutionReceipt", "ReviewClaim", "write_gold_manifest"]',
    )

    # Keep schema discriminators and serialized nested values in content-addressed
    # payloads, but construct dataclasses from their typed runtime values.
    replace_once(
        "evaluation/active_learning.py",
        "    return ActiveLearningBatch(**payload, batch_sha256=_digest(payload))",
        '''    return ActiveLearningBatch(\n        owner_id=payload[\"owner_id\"],\n        policy_sha256=payload[\"policy_sha256\"],\n        candidate_pool_sha256=payload[\"candidate_pool_sha256\"],\n        blocked_items_sha256=payload[\"blocked_items_sha256\"],\n        selected=tuple(selected_rows),\n        total_estimated_cost=total_cost,\n        batch_sha256=_digest(payload),\n    )''',
    )
    replace_once(
        "evaluation/active_learning_gold.py",
        "    return ActiveLearningGoldManifest(**payload, manifest_sha256=_digest(payload))",
        '''    return ActiveLearningGoldManifest(\n        owner_id=payload[\"owner_id\"],\n        label_contract_sha256=payload[\"label_contract_sha256\"],\n        materialization_receipt_sha256s=payload[\"materialization_receipt_sha256s\"],\n        examples=tuple(examples),\n        manifest_sha256=_digest(payload),\n    )''',
    )
    replace_once(
        "orchestration/active_learning_cycle.py",
        "        receipt = ActiveLearningCycleReceipt(**payload, receipt_sha256=_digest(payload))",
        "        constructor = dict(payload)\n        constructor.pop(\"schema\")\n        receipt = ActiveLearningCycleReceipt(**constructor, receipt_sha256=_digest(payload))",
    )
    replace_once(
        "orchestration/active_learning_cycle.py",
        "    receipt = ActiveLearningCycleReceipt(**payload, receipt_sha256=_digest(payload))",
        "    constructor = dict(payload)\n    constructor.pop(\"schema\")\n    receipt = ActiveLearningCycleReceipt(**constructor, receipt_sha256=_digest(payload))",
    )
    replace_once(
        "orchestration/multi_region_authority.py",
        "    return RegionAuthorityDecision(**payload, decision_sha256=_digest(payload))",
        "    constructor = dict(payload)\n    constructor.pop(\"schema\")\n    return RegionAuthorityDecision(**constructor, decision_sha256=_digest(payload))",
    )
    replace_once(
        "orchestration/current_multi_region_authority.py",
        "        return RegionAuthorityDecision(**payload, decision_sha256=_digest(payload))",
        "        constructor = dict(payload)\n        constructor.pop(\"schema\")\n        return RegionAuthorityDecision(**constructor, decision_sha256=_digest(payload))",
    )

    # Preserve both multi-region suites while avoiding pytest's duplicate-basename
    # import collision under its default import mode.
    old_test = ROOT / "tests/unit/test_multi_region_authority.py"
    new_test = ROOT / "tests/unit/test_multi_region_authority_unit.py"
    if not old_test.is_file() or new_test.exists():
        raise RuntimeError("multi-region test rename precondition failed")
    old_test.rename(new_test)

    subprocess.run(
        [
            "python", "-m", "py_compile",
            "evaluation/active_learning.py",
            "evaluation/expert_adjudication.py",
            "evaluation/active_learning_gold.py",
            "orchestration/active_learning_adjudication.py",
            "orchestration/active_learning_cycle.py",
            "orchestration/multi_region_authority.py",
            "orchestration/current_multi_region_authority.py",
            "tests/unit/test_active_learning.py",
            "tests/unit/test_active_learning_gold.py",
            "tests/unit/test_multi_region_authority_unit.py",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python", "-m", "pytest", "-q", "-o", "addopts=",
            "tests/unit/test_expert_adjudication.py",
            "tests/unit/test_active_learning.py",
            "tests/unit/test_active_learning_cycle.py",
            "tests/unit/test_active_learning_gold.py",
            "tests/test_multi_region_authority.py",
            "tests/unit/test_multi_region_authority_unit.py",
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
