"""Authoritative construction path for supply-chain-admitted local learned adapters.

The existing local HF providers remain reusable low-level primitives. Production serving
code that requires supply-chain admission should construct them through these factories,
which re-verify the admitted model/tokenizer trees immediately before provider creation.
No model is loaded by importing this module; loading remains lazy inside the providers.
"""

from __future__ import annotations

from typing import Sequence

from models.admitted_local_artifacts import AdmittedLocalArtifactBinding, require_admitted_local_binding
from models.governed_embedding_profiles import EmbeddingModelProfile
from models.local_hf_adapters import (
    HFDenseEmbeddingProvider,
    LocalHFColBERTProvider,
    LocalHFCrossEncoderProvider,
    LocalHFSpladeProvider,
)


def build_admitted_dense_provider(
    profile: EmbeddingModelProfile,
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    device: str = "auto",
    batch_size: int = 32,
) -> HFDenseEmbeddingProvider:
    binding = require_admitted_local_binding(admitted_binding)
    # Provider construction re-runs governed profile/tree validation in addition to the
    # supply-chain admission checks above.
    return HFDenseEmbeddingProvider(profile, binding, device=device, batch_size=batch_size)


def build_admitted_splade_provider(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    artifact_digest: str,
    device: str = "auto",
    batch_size: int = 16,
    max_length: int = 512,
    activation_threshold: float = 0.0,
    special_token_ids: Sequence[int] = (),
) -> LocalHFSpladeProvider:
    binding = require_admitted_local_binding(admitted_binding)
    if artifact_digest != binding.model_tree_sha256:
        raise ValueError("SPLADE artifact_digest must equal the admitted model tree digest")
    return LocalHFSpladeProvider(
        binding,
        artifact_digest=artifact_digest,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        activation_threshold=activation_threshold,
        special_token_ids=special_token_ids,
    )


def build_admitted_colbert_provider(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    artifact_digest: str,
    projection_dim: int = 128,
    device: str = "auto",
    max_length: int = 512,
    special_token_ids: Sequence[int] = (),
) -> LocalHFColBERTProvider:
    binding = require_admitted_local_binding(admitted_binding)
    if artifact_digest != binding.model_tree_sha256:
        raise ValueError("ColBERT artifact_digest must equal the admitted model tree digest")
    return LocalHFColBERTProvider(
        binding,
        artifact_digest=artifact_digest,
        projection_dim=projection_dim,
        device=device,
        max_length=max_length,
        special_token_ids=special_token_ids,
    )


def build_admitted_cross_encoder_provider(
    admitted_binding: AdmittedLocalArtifactBinding,
    *,
    artifact_digest: str,
    device: str = "auto",
    max_length: int = 512,
    score_index: int = 0,
) -> LocalHFCrossEncoderProvider:
    binding = require_admitted_local_binding(admitted_binding)
    if artifact_digest != binding.model_tree_sha256:
        raise ValueError("cross-encoder artifact_digest must equal the admitted model tree digest")
    return LocalHFCrossEncoderProvider(
        binding,
        artifact_digest=artifact_digest,
        device=device,
        max_length=max_length,
        score_index=score_index,
    )


__all__ = [
    "build_admitted_colbert_provider",
    "build_admitted_cross_encoder_provider",
    "build_admitted_dense_provider",
    "build_admitted_splade_provider",
]
