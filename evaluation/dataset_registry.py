"""Static governed catalog for retrieval, domain and multi-hop evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationDatasetSpec:
    name: str
    task: str
    domain: str
    format: str
    graded_relevance: bool
    multihop: bool
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in ("name", "task", "domain", "format"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 200
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise ValueError(f"{field_name} is invalid.")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.graded_relevance, bool) or not isinstance(self.multihop, bool):
            raise ValueError("dataset flags must be booleans.")
        if not isinstance(self.notes, str) or len(self.notes) > 2_000 or "\x00" in self.notes:
            raise ValueError("notes are invalid.")


DATASET_REGISTRY: dict[str, EvaluationDatasetSpec] = {
    "scifact": EvaluationDatasetSpec(
        "scifact",
        "claim-to-scientific-evidence retrieval",
        "scientific",
        "beir",
        False,
        False,
        "Use for scientific evidence and citation-grounding retrieval.",
    ),
    "nfcorpus": EvaluationDatasetSpec(
        "nfcorpus",
        "biomedical information retrieval",
        "biomedical",
        "beir",
        True,
        False,
    ),
    "fiqa": EvaluationDatasetSpec(
        "fiqa",
        "financial question retrieval",
        "financial",
        "beir",
        True,
        False,
    ),
    "trec-covid": EvaluationDatasetSpec(
        "trec-covid",
        "pandemic scientific literature retrieval",
        "biomedical",
        "beir",
        True,
        False,
    ),
    "arguana": EvaluationDatasetSpec(
        "arguana",
        "counter-argument retrieval",
        "argumentation",
        "beir",
        False,
        False,
    ),
    "cqadupstack": EvaluationDatasetSpec(
        "cqadupstack",
        "duplicate community-question retrieval",
        "general",
        "beir",
        False,
        False,
    ),
    "hotpotqa": EvaluationDatasetSpec(
        "hotpotqa",
        "multi-hop question answering evidence retrieval",
        "general",
        "multihop-json",
        False,
        True,
    ),
    "musique": EvaluationDatasetSpec(
        "musique",
        "compositional multi-hop evidence retrieval",
        "general",
        "multihop-json",
        False,
        True,
    ),
    "qasper": EvaluationDatasetSpec(
        "qasper",
        "scientific paper question answering and evidence grounding",
        "scientific",
        "multihop-json",
        False,
        True,
        "Governed adapter supports paper context and question-level answer extraction.",
    ),
    "miracl": EvaluationDatasetSpec(
        "miracl",
        "multilingual passage retrieval",
        "multilingual",
        "multilingual-json",
        False,
        False,
        "Use language metadata for per-language retrieval and ranking breakdowns.",
    ),
    "cuad": EvaluationDatasetSpec(
        "cuad",
        "contract clause retrieval and question answering",
        "legal",
        "domain-adapter",
        False,
        False,
        "Requires an explicit local adapter before evaluation.",
    ),
    "finqa": EvaluationDatasetSpec(
        "finqa",
        "financial report evidence retrieval and reasoning",
        "financial",
        "domain-adapter",
        False,
        True,
        "Requires an explicit local adapter before evaluation.",
    ),
    "pubmedqa": EvaluationDatasetSpec(
        "pubmedqa",
        "biomedical literature question answering",
        "biomedical",
        "domain-adapter",
        False,
        False,
        "Requires an explicit local adapter before evaluation.",
    ),
}


def get_dataset_spec(name: str) -> EvaluationDatasetSpec:
    if not isinstance(name, str):
        raise ValueError("dataset name must be a string.")
    selected = name.strip().lower()
    if selected not in DATASET_REGISTRY:
        raise KeyError(selected)
    return DATASET_REGISTRY[selected]


def list_dataset_specs(
    *,
    domain: str | None = None,
    multihop: bool | None = None,
    format: str | None = None,
) -> tuple[EvaluationDatasetSpec, ...]:
    if domain is not None and not isinstance(domain, str):
        raise ValueError("domain must be a string or null.")
    if multihop is not None and not isinstance(multihop, bool):
        raise ValueError("multihop must be boolean or null.")
    if format is not None and not isinstance(format, str):
        raise ValueError("format must be a string or null.")
    selected_domain = None if domain is None else domain.strip().lower()
    selected_format = None if format is None else format.strip().lower()
    return tuple(
        spec
        for _, spec in sorted(DATASET_REGISTRY.items())
        if (selected_domain is None or spec.domain.lower() == selected_domain)
        and (multihop is None or spec.multihop == multihop)
        and (selected_format is None or spec.format.lower() == selected_format)
    )


__all__ = [
    "DATASET_REGISTRY",
    "EvaluationDatasetSpec",
    "get_dataset_spec",
    "list_dataset_specs",
]
