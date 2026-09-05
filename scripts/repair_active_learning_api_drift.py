#!/usr/bin/env python3
"""One-shot, fail-closed repair for active-learning/adjudication API drift.

This script is intentionally transactional: every textual precondition must match
exactly once before files are changed. CI runs the affected tests before committing
the resulting tree. Delete this script and its workflow after the repair lands.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one repair anchor, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    # The current adjudication authority names the policy AdjudicationPolicy and
    # applies it during reconciliation. Active-learning materialization only creates
    # immutable cases, so it must not pass the removed policy argument to create_case.
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

    # Keep the active-learning test on the current public policy contract.
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

    # Re-introduce the compact immutable resolved-gold lineage value object consumed
    # by evaluation.active_learning_gold. It is distinct from GoldLabelRecord, which
    # is the store's richer governed-manifest record.
    anchor = "@dataclass(frozen=True)\nclass GoldLabelRecord:\n"
    insertion = '''@dataclass(frozen=True)\nclass GoldLabel:\n    \"\"\"Compact resolved-label lineage used by active-learning derived datasets.\"\"\"\n\n    case_id: str\n    item_sha256: str\n    round_index: int\n    label: str\n    resolution_revision: int\n    resolution_digest: str\n\n    def __post_init__(self) -> None:\n        object.__setattr__(self, \"case_id\", _text(self.case_id, \"case_id\", 1000))\n        object.__setattr__(self, \"item_sha256\", _sha(self.item_sha256, \"item_sha256\"))\n        object.__setattr__(self, \"label\", _text(self.label, \"label\", 300))\n        for name in (\"round_index\", \"resolution_revision\"):\n            value = getattr(self, name)\n            if isinstance(value, bool) or not isinstance(value, int) or value < 0:\n                raise ValueError(f\"{name} must be non-negative\")\n        object.__setattr__(\n            self, \"resolution_digest\", _sha(self.resolution_digest, \"resolution_digest\")\n        )\n\n\n@dataclass(frozen=True)\nclass GoldLabelRecord:\n'''
    replace_once("evaluation/expert_adjudication.py", anchor, insertion)
    replace_once(
        "evaluation/expert_adjudication.py",
        '__all__ = ["AdjudicationCase", "AdjudicationPolicy", "CaseRecord", "ExpertAdjudicationStore", "ExpertJudgment", "GoldLabelManifest", "GoldLabelRecord", "LabelSchema", "ResolutionReceipt", "ReviewClaim", "write_gold_manifest"]',
        '__all__ = ["AdjudicationCase", "AdjudicationPolicy", "CaseRecord", "ExpertAdjudicationStore", "ExpertJudgment", "GoldLabel", "GoldLabelManifest", "GoldLabelRecord", "LabelSchema", "ResolutionReceipt", "ReviewClaim", "write_gold_manifest"]',
    )

    # Pytest's default import mode cannot collect two same-basename test modules in
    # tests/ and tests/unit/. Preserve both suites but give the unit module a unique name.
    old_test = ROOT / "tests/unit/test_multi_region_authority.py"
    new_test = ROOT / "tests/unit/test_multi_region_authority_unit.py"
    if not old_test.is_file() or new_test.exists():
        raise RuntimeError("multi-region test rename precondition failed")
    old_test.rename(new_test)

    subprocess.run(
        [
            "python",
            "-m",
            "py_compile",
            "evaluation/expert_adjudication.py",
            "evaluation/active_learning_gold.py",
            "orchestration/active_learning_adjudication.py",
            "tests/unit/test_active_learning.py",
            "tests/unit/test_active_learning_gold.py",
            "tests/unit/test_multi_region_authority_unit.py",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
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
