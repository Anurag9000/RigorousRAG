from __future__ import annotations

from pathlib import Path

import pytest

from training.authoritative_retrieval_training_cli import _stage_specs, _step, _tree_sha256
from training.distilled_steps import (
    DistilledColBERTContrastiveStep,
    DistilledDenseContrastiveStep,
    DistilledSparseContrastiveStep,
)
from training.torch_engine import (
    ColBERTContrastiveStep,
    DenseContrastiveStep,
    ListwiseCrossEncoderStep,
    SparseContrastiveStep,
)


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


def test_architecture_step_mapping_covers_base_and_distilled_paths() -> None:
    dense_base = _step("dense", {"step_variant": "base", "loss": {"retrieval_temperature": 0.2}})
    dense_distilled = _step("dense", {"step_variant": "distilled", "loss": {"distillation_weight": 0.5}})
    splade_base = _step("splade", {"step_variant": "base", "loss": {}})
    splade_distilled = _step("splade", {"step_variant": "distilled", "loss": {"distillation_weight": 0.5}})
    unicoil_base = _step("unicoil", {"step_variant": "base", "loss": {}})
    colbert_base = _step("colbert", {"step_variant": "base", "loss": {}})
    colbert_distilled = _step("colbert", {"step_variant": "distilled", "loss": {"distillation_weight": 0.5}})
    cross = _step("cross_encoder", {"step_variant": "listwise", "loss": {"listwise_temperature": 0.7}})

    assert isinstance(dense_base, DenseContrastiveStep)
    assert dense_base.temperature == 0.2
    assert isinstance(dense_distilled, DistilledDenseContrastiveStep)
    assert dense_distilled.config.distillation_weight == 0.5
    assert isinstance(splade_base, SparseContrastiveStep)
    assert isinstance(splade_distilled, DistilledSparseContrastiveStep)
    assert isinstance(unicoil_base, SparseContrastiveStep)
    assert isinstance(colbert_base, ColBERTContrastiveStep)
    assert isinstance(colbert_distilled, DistilledColBERTContrastiveStep)
    assert isinstance(cross, ListwiseCrossEncoderStep)
    assert cross.temperature == 0.7


def test_invalid_step_variants_fail_closed() -> None:
    with pytest.raises(ValueError, match="base or distilled"):
        _step("dense", {"step_variant": "mystery", "loss": {}})
    with pytest.raises(ValueError, match="base or listwise"):
        _step("cross_encoder", {"step_variant": "distilled", "loss": {}})
