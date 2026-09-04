from __future__ import annotations

from pathlib import Path

import pytest

from training.authoritative_retrieval_training_cli import _stage_specs, _step, _tree_sha256
from training.distilled_steps import (
    DistilledColBERTContrastiveStep,
    DistilledDenseContrastiveStep,
    DistilledSparseContrastiveStep,
)
from training.torch_engine import ListwiseCrossEncoderStep


def test_local_artifact_tree_digest_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    (root / "nested").mkdir(parents=True)
    (root / "config.json").write_text('{"hidden":4}\n', encoding="utf-8")
    (root / "nested" / "weights.bin").write_bytes(b"abc")

    first = _tree_sha256(root)
    second = _tree_sha256(root)
    assert first == second
    assert len(first) == 64

    (root / "nested" / "weights.bin").write_bytes(b"abcd")
    assert _tree_sha256(root) != first


def test_retrieval_stages_must_evaluate_for_early_stopping() -> None:
    valid = _stage_specs(
        {
            "stages": [
                {
                    "name": "contrastive",
                    "max_optimizer_steps": 10,
                    "learning_rate": 1e-5,
                    "checkpoint_every_steps": 2,
                    "evaluate_every_steps": 2,
                }
            ]
        }
    )
    assert valid[0].evaluate_every_steps == 2

    with pytest.raises(ValueError, match="must evaluate"):
        _stage_specs(
            {
                "stages": [
                    {
                        "name": "invalid",
                        "max_optimizer_steps": 10,
                        "learning_rate": 1e-5,
                        "checkpoint_every_steps": 2,
                    }
                ]
            }
        )


def test_architecture_step_mapping_covers_all_trainable_retrieval_families() -> None:
    dense = _step("dense", {"loss": {"distillation_weight": 0.5}})
    splade = _step("splade", {"loss": {"distillation_weight": 0.5}})
    unicoil = _step("unicoil", {"loss": {"distillation_weight": 0.5}})
    colbert = _step("colbert", {"loss": {"distillation_weight": 0.5}})
    cross = _step("cross_encoder", {"loss": {"listwise_temperature": 0.7}})

    assert isinstance(dense, DistilledDenseContrastiveStep)
    assert dense.config.distillation_weight == 0.5
    assert isinstance(splade, DistilledSparseContrastiveStep)
    assert isinstance(unicoil, DistilledSparseContrastiveStep)
    assert isinstance(colbert, DistilledColBERTContrastiveStep)
    assert isinstance(cross, ListwiseCrossEncoderStep)
    assert cross.temperature == 0.7
