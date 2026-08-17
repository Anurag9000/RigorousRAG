"""Differentiable PyTorch losses for learned retrieval, reranking and distillation.

Unlike the repository's framework-neutral reference mathematics, these functions preserve
autograd graphs and are intended to be called directly by the executable training engine.
No model or dataset is loaded by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional training dependency.
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("training losses require the optional PyTorch dependency")


def _positive(value: float, label: str) -> float:
    selected = float(value)
    if not selected > 0.0:
        raise ValueError(f"{label} must be positive")
    return selected


@dataclass(frozen=True)
class SparsePenaltyWeights:
    query_l1: float = 0.0
    document_l1: float = 0.0
    query_flops: float = 0.0
    document_flops: float = 0.0

    def __post_init__(self) -> None:
        for name in ("query_l1", "document_l1", "query_flops", "document_flops"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass
class TensorLossBreakdown:
    total: Any
    retrieval: Any
    distillation: Any | None = None
    query_l1: Any | None = None
    document_l1: Any | None = None
    query_flops: Any | None = None
    document_flops: Any | None = None
    auxiliary: dict[str, Any] | None = None


def distributed_gather_with_local_grad(tensor: Any) -> tuple[Any, int]:
    """Gather equal-shaped tensors while retaining autograd for the local rank.

    Remote tensors are treated as constants.  This is the common in-batch-negative
    approximation when a full gradient-aware distributed gather primitive is unavailable.
    The returned offset is the local row offset in the concatenated tensor.
    """

    _require_torch()
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return tensor, 0
    world = dist.get_world_size()
    rank = dist.get_rank()
    shapes: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(shapes, tuple(tensor.shape))
    if any(tuple(shape) != tuple(tensor.shape) for shape in shapes):
        raise ValueError("distributed in-batch gathering requires equal tensor shapes on every rank")
    detached = tensor.detach()
    gathered = [torch.empty_like(detached) for _ in range(world)]
    dist.all_gather(gathered, detached)
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0), rank * tensor.size(0)


def in_batch_info_nce(
    scores: Any,
    *,
    positive_indices: Any | None = None,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
) -> Any:
    """Cross entropy over a query-by-candidate score matrix."""

    _require_torch()
    if scores.ndim != 2 or scores.size(0) < 1 or scores.size(1) < 1:
        raise ValueError("scores must have shape [queries, candidates]")
    selected_temperature = _positive(temperature, "temperature")
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError("label_smoothing must be in [0,1)")
    if positive_indices is None:
        if scores.size(0) > scores.size(1):
            raise ValueError("identity positives require at least as many candidates as queries")
        positive_indices = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    if positive_indices.ndim != 1 or positive_indices.numel() != scores.size(0):
        raise ValueError("positive_indices must contain one candidate index per query")
    if torch.any(positive_indices < 0) or torch.any(positive_indices >= scores.size(1)):
        raise ValueError("positive index lies outside candidate dimension")
    return F.cross_entropy(
        scores / selected_temperature,
        positive_indices.to(dtype=torch.long),
        label_smoothing=float(label_smoothing),
    )


def distributed_biencoder_info_nce(
    query_embeddings: Any,
    document_embeddings: Any,
    *,
    temperature: float = 0.05,
    label_smoothing: float = 0.0,
) -> Any:
    """In-batch dense contrastive loss with optional cross-rank negatives."""

    _require_torch()
    if query_embeddings.ndim != 2 or document_embeddings.ndim != 2:
        raise ValueError("bi-encoder embeddings must be rank-2 tensors")
    if query_embeddings.shape != document_embeddings.shape:
        raise ValueError("aligned in-batch positives require equal query/document shapes")
    gathered_documents, offset = distributed_gather_with_local_grad(document_embeddings)
    scores = query_embeddings @ gathered_documents.transpose(0, 1)
    positives = torch.arange(query_embeddings.size(0), device=scores.device, dtype=torch.long) + offset
    return in_batch_info_nce(
        scores,
        positive_indices=positives,
        temperature=temperature,
        label_smoothing=label_smoothing,
    )


def sparse_l1(weights: Any) -> Any:
    _require_torch()
    if weights.ndim != 2:
        raise ValueError("sparse weights must have shape [batch, vocab]")
    return weights.abs().sum(dim=-1).mean()


def sparse_flops(weights: Any) -> Any:
    """SPLADE FLOPS proxy: sum over squared mean absolute vocabulary activation."""

    _require_torch()
    if weights.ndim != 2:
        raise ValueError("sparse weights must have shape [batch, vocab]")
    return weights.abs().mean(dim=0).pow(2).sum()


def sparse_retrieval_objective(
    scores: Any,
    positive_indices: Any,
    query_weights: Any,
    document_weights: Any,
    *,
    temperature: float = 1.0,
    penalties: SparsePenaltyWeights = SparsePenaltyWeights(),
    student_logits: Any | None = None,
    teacher_logits: Any | None = None,
    distillation_weight: float = 0.0,
    teacher_temperature: float = 1.0,
) -> TensorLossBreakdown:
    _require_torch()
    retrieval = in_batch_info_nce(scores, positive_indices=positive_indices, temperature=temperature)
    q_l1 = sparse_l1(query_weights)
    d_l1 = sparse_l1(document_weights)
    q_flops = sparse_flops(query_weights)
    d_flops = sparse_flops(document_weights)
    total = (
        retrieval
        + penalties.query_l1 * q_l1
        + penalties.document_l1 * d_l1
        + penalties.query_flops * q_flops
        + penalties.document_flops * d_flops
    )
    distillation = None
    if student_logits is not None or teacher_logits is not None:
        if student_logits is None or teacher_logits is None:
            raise ValueError("student_logits and teacher_logits must be supplied together")
        distillation = distillation_kl(student_logits, teacher_logits, temperature=teacher_temperature)
        if float(distillation_weight) < 0.0:
            raise ValueError("distillation_weight must be non-negative")
        total = total + float(distillation_weight) * distillation
    return TensorLossBreakdown(
        total=total,
        retrieval=retrieval,
        distillation=distillation,
        query_l1=q_l1,
        document_l1=d_l1,
        query_flops=q_flops,
        document_flops=d_flops,
    )


def pairwise_softplus(positive_scores: Any, negative_scores: Any, *, margin: float = 0.0) -> Any:
    _require_torch()
    if positive_scores.shape != negative_scores.shape or positive_scores.numel() == 0:
        raise ValueError("positive and negative score tensors must be non-empty and aligned")
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")
    return F.softplus(float(margin) - positive_scores + negative_scores).mean()


def pairwise_margin_ranking(positive_scores: Any, negative_scores: Any, *, margin: float = 1.0) -> Any:
    _require_torch()
    if positive_scores.shape != negative_scores.shape or positive_scores.numel() == 0:
        raise ValueError("positive and negative score tensors must be non-empty and aligned")
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")
    target = torch.ones_like(positive_scores)
    return F.margin_ranking_loss(positive_scores, negative_scores, target, margin=float(margin))


def listnet_loss(scores: Any, relevance: Any, *, temperature: float = 1.0) -> Any:
    """ListNet cross entropy for batched equal-length candidate lists."""

    _require_torch()
    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have aligned [batch, candidates] shapes")
    selected_temperature = _positive(temperature, "temperature")
    target = F.softmax(relevance.to(dtype=scores.dtype) / selected_temperature, dim=-1)
    log_probability = F.log_softmax(scores / selected_temperature, dim=-1)
    return -(target * log_probability).sum(dim=-1).mean()


def listmle_loss(scores: Any, relevance: Any) -> Any:
    """ListMLE negative log-likelihood using descending relevance permutations."""

    _require_torch()
    if scores.ndim != 2 or relevance.shape != scores.shape:
        raise ValueError("scores and relevance must have aligned [batch, candidates] shapes")
    order = torch.argsort(relevance, dim=-1, descending=True, stable=True)
    ordered_scores = torch.gather(scores, 1, order)
    # log(sum(exp(scores_j..end))) computed stably by reverse logcumsumexp.
    denominators = torch.logcumsumexp(torch.flip(ordered_scores, dims=[1]), dim=1)
    denominators = torch.flip(denominators, dims=[1])
    return (denominators - ordered_scores).sum(dim=-1).mean()


def distillation_kl(student_logits: Any, teacher_logits: Any, *, temperature: float = 1.0) -> Any:
    _require_torch()
    if student_logits.shape != teacher_logits.shape or student_logits.numel() == 0:
        raise ValueError("student and teacher logits must be non-empty and aligned")
    selected_temperature = _positive(temperature, "temperature")
    student_log = F.log_softmax(student_logits / selected_temperature, dim=-1)
    teacher_probability = F.softmax(teacher_logits.detach() / selected_temperature, dim=-1)
    return (
        F.kl_div(student_log, teacher_probability, reduction="batchmean")
        * selected_temperature
        * selected_temperature
    )


def margin_mse_distillation(student_scores: Any, teacher_scores: Any) -> Any:
    """MSE over relative score margins, useful for pair/list distillation."""

    _require_torch()
    if student_scores.shape != teacher_scores.shape or student_scores.ndim < 2:
        raise ValueError("student/teacher scores must be aligned candidate lists")
    student_margin = student_scores - student_scores[..., :1]
    teacher_margin = teacher_scores.detach() - teacher_scores.detach()[..., :1]
    return F.mse_loss(student_margin, teacher_margin)


def matryoshka_contrastive_loss(
    query_embeddings: Any,
    document_embeddings: Any,
    dimensions: Sequence[int],
    *,
    temperature: float = 0.05,
    weights: Sequence[float] | None = None,
) -> Any:
    """Nested-dimension dense loss for one model serving multiple embedding widths."""

    _require_torch()
    if query_embeddings.shape != document_embeddings.shape or query_embeddings.ndim != 2:
        raise ValueError("query/document embeddings must be aligned rank-2 tensors")
    if not dimensions:
        raise ValueError("dimensions must be non-empty")
    selected_dimensions = tuple(int(value) for value in dimensions)
    if any(value <= 0 or value > query_embeddings.size(-1) for value in selected_dimensions):
        raise ValueError("matryoshka dimension is outside embedding width")
    if len(set(selected_dimensions)) != len(selected_dimensions):
        raise ValueError("matryoshka dimensions must be unique")
    if weights is None:
        selected_weights = [1.0 / len(selected_dimensions)] * len(selected_dimensions)
    else:
        if len(weights) != len(selected_dimensions):
            raise ValueError("weights must align with dimensions")
        selected_weights = [float(value) for value in weights]
        if any(value < 0.0 for value in selected_weights) or sum(selected_weights) <= 0.0:
            raise ValueError("matryoshka weights must be non-negative with positive total")
        total = sum(selected_weights)
        selected_weights = [value / total for value in selected_weights]
    loss = query_embeddings.new_zeros(())
    for dimension, weight in zip(selected_dimensions, selected_weights):
        queries = F.normalize(query_embeddings[:, :dimension], p=2, dim=-1)
        documents = F.normalize(document_embeddings[:, :dimension], p=2, dim=-1)
        loss = loss + weight * distributed_biencoder_info_nce(
            queries,
            documents,
            temperature=temperature,
        )
    return loss


def symmetric_contrastive_loss(
    query_embeddings: Any,
    document_embeddings: Any,
    *,
    temperature: float = 0.05,
) -> Any:
    """Average query->document and document->query contrastive objectives."""

    _require_torch()
    if query_embeddings.shape != document_embeddings.shape or query_embeddings.ndim != 2:
        raise ValueError("query/document embeddings must be aligned rank-2 tensors")
    scores = query_embeddings @ document_embeddings.transpose(0, 1)
    labels = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    forward = in_batch_info_nce(scores, positive_indices=labels, temperature=temperature)
    backward = in_batch_info_nce(scores.transpose(0, 1), positive_indices=labels, temperature=temperature)
    return 0.5 * (forward + backward)


__all__ = [
    "SparsePenaltyWeights",
    "TensorLossBreakdown",
    "distributed_biencoder_info_nce",
    "distributed_gather_with_local_grad",
    "distillation_kl",
    "in_batch_info_nce",
    "listmle_loss",
    "listnet_loss",
    "margin_mse_distillation",
    "matryoshka_contrastive_loss",
    "pairwise_margin_ranking",
    "pairwise_softplus",
    "sparse_flops",
    "sparse_l1",
    "sparse_retrieval_objective",
    "symmetric_contrastive_loss",
]
