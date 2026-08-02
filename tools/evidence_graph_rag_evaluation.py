"""Strict evaluation metrics for bounded evidence-graph selections."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from tools.evidence_graph_rag import GraphEvidenceSelection

_MAX_CASES = 1_000_000
_MAX_REFERENCES = 100_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bounded_tuple(values: Iterable[Any], label: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    result = tuple(values)
    if len(result) > _MAX_REFERENCES:
        raise ValueError(f"{label} exceeds the item limit.")
    return result


def _prf(predicted: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    if not predicted:
        return 1.0, 0.0, 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


@dataclass(frozen=True, order=True)
class GraphNodeLocator:
    doc_id: str
    generation: int
    node_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", 1, 2**63 - 1),
        )
        object.__setattr__(self, "node_id", _digest(self.node_id, "node_id"))

    @property
    def locator_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphRAGGoldCase:
    query_id: str
    graph_set_id: str
    graph_set_digest: str
    query_digest: str
    relevant_nodes: tuple[GraphNodeLocator, ...]
    required_edge_ids: tuple[str, ...]
    should_abstain: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id", 500))
        for name in ("graph_set_id", "graph_set_digest", "query_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.relevant_nodes, tuple):
            object.__setattr__(self, "relevant_nodes", tuple(self.relevant_nodes))
        if len(self.relevant_nodes) > _MAX_REFERENCES or any(
            not isinstance(value, GraphNodeLocator) for value in self.relevant_nodes
        ):
            raise ValueError("relevant_nodes must be bounded GraphNodeLocator values.")
        values = tuple(sorted(set(self.relevant_nodes)))
        object.__setattr__(self, "relevant_nodes", values)
        edges = tuple(sorted(set(_digest(value, "required_edge_id") for value in self.required_edge_ids)))
        if len(edges) > _MAX_REFERENCES:
            raise ValueError("required_edge_ids exceed the limit.")
        object.__setattr__(self, "required_edge_ids", edges)
        if not isinstance(self.should_abstain, bool):
            raise ValueError("should_abstain must be boolean.")
        if self.should_abstain and (values or edges):
            raise ValueError("abstention cases may not contain required nodes or edges.")

    @property
    def case_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphRAGEvaluation:
    query_id: str
    case_digest: str
    selection_digest: str
    node_precision: float
    node_recall: float
    node_f1: float
    document_precision: float
    document_recall: float
    document_f1: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    complete_required_path: bool
    lineage_completeness: float
    abstention_correct: bool
    evidence_count: int
    traversal_count: int
    estimated_work_units: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id", 500))
        object.__setattr__(self, "case_digest", _digest(self.case_digest, "case_digest"))
        object.__setattr__(
            self, "selection_digest", _digest(self.selection_digest, "selection_digest")
        )
        for name in (
            "node_precision",
            "node_recall",
            "node_f1",
            "document_precision",
            "document_recall",
            "document_f1",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "lineage_completeness",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not isinstance(self.complete_required_path, bool):
            raise ValueError("complete_required_path must be boolean.")
        if not isinstance(self.abstention_correct, bool):
            raise ValueError("abstention_correct must be boolean.")
        for name in ("evidence_count", "traversal_count", "estimated_work_units"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 0, 10_000_000),
            )

    @property
    def evaluation_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphRAGAggregate:
    case_count: int
    macro_node_precision: float
    macro_node_recall: float
    macro_node_f1: float
    macro_document_precision: float
    macro_document_recall: float
    macro_document_f1: float
    macro_edge_precision: float
    macro_edge_recall: float
    macro_edge_f1: float
    complete_required_path_rate: float
    mean_lineage_completeness: float
    abstention_accuracy: float
    mean_evidence_count: float
    mean_traversal_count: float
    mean_estimated_work_units: float
    evaluation_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_count", _integer(self.case_count, "case_count", 1, _MAX_CASES))
        for name in (
            "macro_node_precision",
            "macro_node_recall",
            "macro_node_f1",
            "macro_document_precision",
            "macro_document_recall",
            "macro_document_f1",
            "macro_edge_precision",
            "macro_edge_recall",
            "macro_edge_f1",
            "complete_required_path_rate",
            "mean_lineage_completeness",
            "abstention_accuracy",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in (
            "mean_evidence_count",
            "mean_traversal_count",
            "mean_estimated_work_units",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        values = tuple(_digest(value, "evaluation_digest") for value in self.evaluation_digests)
        if len(values) != self.case_count:
            raise ValueError("evaluation_digests must match case_count.")
        object.__setattr__(self, "evaluation_digests", values)

    @property
    def aggregate_digest(self) -> str:
        return _sha256(asdict(self))


def evaluate_graph_selection(
    selection: GraphEvidenceSelection,
    gold: GraphRAGGoldCase,
) -> GraphRAGEvaluation:
    if not isinstance(selection, GraphEvidenceSelection):
        raise ValueError("selection must be GraphEvidenceSelection.")
    if not isinstance(gold, GraphRAGGoldCase):
        raise ValueError("gold must be GraphRAGGoldCase.")
    if (
        selection.graph_set_id != gold.graph_set_id
        or selection.graph_set_digest != gold.graph_set_digest
        or selection.query_digest != gold.query_digest
    ):
        raise ValueError("selection and gold identities differ.")
    predicted_nodes = {
        GraphNodeLocator(item.doc_id, item.generation, item.node_id)
        for item in selection.items
    }
    gold_nodes = set(gold.relevant_nodes)
    node_p, node_r, node_f1 = _prf(predicted_nodes, gold_nodes)
    predicted_docs = {value.doc_id for value in predicted_nodes}
    gold_docs = {value.doc_id for value in gold_nodes}
    doc_p, doc_r, doc_f1 = _prf(predicted_docs, gold_docs)
    predicted_edges = {step.edge_id for step in selection.traversals}
    gold_edges = set(gold.required_edge_ids)
    edge_p, edge_r, edge_f1 = _prf(predicted_edges, gold_edges)
    step_digests = {step.step_digest for step in selection.traversals}
    expanded = [item for item in selection.items if item.origin != "lexical"]
    lineage_complete = (
        1.0
        if not expanded
        else sum(
            bool(item.lineage_step_digests)
            and all(value in step_digests for value in item.lineage_step_digests)
            for item in expanded
        )
        / len(expanded)
    )
    return GraphRAGEvaluation(
        query_id=gold.query_id,
        case_digest=gold.case_digest,
        selection_digest=selection.selection_digest,
        node_precision=node_p,
        node_recall=node_r,
        node_f1=node_f1,
        document_precision=doc_p,
        document_recall=doc_r,
        document_f1=doc_f1,
        edge_precision=edge_p,
        edge_recall=edge_r,
        edge_f1=edge_f1,
        complete_required_path=gold_edges <= predicted_edges,
        lineage_completeness=lineage_complete,
        abstention_correct=selection.abstained == gold.should_abstain,
        evidence_count=len(selection.items),
        traversal_count=len(selection.traversals),
        estimated_work_units=selection.estimated_work_units,
    )


def aggregate_graph_evaluations(
    evaluations: Iterable[GraphRAGEvaluation],
) -> GraphRAGAggregate:
    values = _bounded_tuple(evaluations, "evaluations")
    if not values or any(not isinstance(value, GraphRAGEvaluation) for value in values):
        raise ValueError("evaluations must contain GraphRAGEvaluation values.")
    count = len(values)

    def mean(name: str) -> float:
        return sum(float(getattr(value, name)) for value in values) / count

    return GraphRAGAggregate(
        case_count=count,
        macro_node_precision=mean("node_precision"),
        macro_node_recall=mean("node_recall"),
        macro_node_f1=mean("node_f1"),
        macro_document_precision=mean("document_precision"),
        macro_document_recall=mean("document_recall"),
        macro_document_f1=mean("document_f1"),
        macro_edge_precision=mean("edge_precision"),
        macro_edge_recall=mean("edge_recall"),
        macro_edge_f1=mean("edge_f1"),
        complete_required_path_rate=mean("complete_required_path"),
        mean_lineage_completeness=mean("lineage_completeness"),
        abstention_accuracy=mean("abstention_correct"),
        mean_evidence_count=mean("evidence_count"),
        mean_traversal_count=mean("traversal_count"),
        mean_estimated_work_units=mean("estimated_work_units"),
        evaluation_digests=tuple(value.evaluation_digest for value in values),
    )


def gold_case_from_mapping(value: Mapping[str, Any]) -> GraphRAGGoldCase:
    if not isinstance(value, Mapping) or set(value) != {
        "query_id",
        "graph_set_id",
        "graph_set_digest",
        "query_digest",
        "relevant_nodes",
        "required_edge_ids",
        "should_abstain",
    }:
        raise ValueError("graph RAG gold case schema is invalid.")
    nodes = value["relevant_nodes"]
    if not isinstance(nodes, list):
        raise ValueError("relevant_nodes must be a JSON array.")
    return GraphRAGGoldCase(
        query_id=value["query_id"],
        graph_set_id=value["graph_set_id"],
        graph_set_digest=value["graph_set_digest"],
        query_digest=value["query_digest"],
        relevant_nodes=tuple(GraphNodeLocator(**item) for item in nodes),
        required_edge_ids=tuple(value["required_edge_ids"]),
        should_abstain=value["should_abstain"],
    )


__all__ = [
    "GraphNodeLocator",
    "GraphRAGAggregate",
    "GraphRAGEvaluation",
    "GraphRAGGoldCase",
    "aggregate_graph_evaluations",
    "evaluate_graph_selection",
    "gold_case_from_mapping",
]
