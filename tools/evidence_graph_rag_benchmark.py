"""Strict query-digest-only benchmark fixtures for evidence-graph selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from tools.evidence_graph_rag_evaluation import (
    GraphNodeLocator,
    GraphRAGAggregate,
    GraphRAGEvaluation,
    GraphRAGGoldCase,
    aggregate_graph_evaluations,
    gold_case_from_mapping,
)

_MAX_RUNS = 10_000
_MAX_CASES_PER_RUN = 1_000_000
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


def _tuple(values: Iterable[Any], label: str, maximum: int) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    result = tuple(values)
    if len(result) > maximum:
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


@dataclass(frozen=True)
class GraphRAGSelectionObservation:
    graph_set_id: str
    graph_set_digest: str
    query_digest: str
    selection_digest: str
    selected_nodes: tuple[GraphNodeLocator, ...]
    traversal_edge_ids: tuple[str, ...]
    expanded_lineage_valid: tuple[bool, ...]
    abstained: bool
    evidence_count: int
    traversal_count: int
    estimated_work_units: int

    def __post_init__(self) -> None:
        for name in (
            "graph_set_id",
            "graph_set_digest",
            "query_digest",
            "selection_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.selected_nodes, tuple):
            object.__setattr__(self, "selected_nodes", tuple(self.selected_nodes))
        nodes = tuple(sorted(set(self.selected_nodes)))
        if len(nodes) > _MAX_REFERENCES or any(
            not isinstance(value, GraphNodeLocator) for value in nodes
        ):
            raise ValueError("selected_nodes must be bounded GraphNodeLocator values.")
        object.__setattr__(self, "selected_nodes", nodes)
        edges = tuple(
            sorted(set(_digest(value, "traversal_edge_id") for value in self.traversal_edge_ids))
        )
        if len(edges) > _MAX_REFERENCES:
            raise ValueError("traversal_edge_ids exceed the item limit.")
        object.__setattr__(self, "traversal_edge_ids", edges)
        lineage = tuple(self.expanded_lineage_valid)
        if len(lineage) > _MAX_REFERENCES or any(not isinstance(value, bool) for value in lineage):
            raise ValueError("expanded_lineage_valid must be bounded booleans.")
        object.__setattr__(self, "expanded_lineage_valid", lineage)
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean.")
        for name in ("evidence_count", "traversal_count", "estimated_work_units"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 0, 10_000_000),
            )
        if self.evidence_count != len(nodes):
            raise ValueError("evidence_count must equal selected node count.")
        if self.traversal_count < len(edges):
            raise ValueError("traversal_count may not be below distinct edge count.")
        if len(lineage) > self.evidence_count:
            raise ValueError("expanded lineage observations exceed evidence count.")
        if self.abstained != (self.evidence_count == 0):
            raise ValueError("abstained must exactly reflect empty evidence.")

    @property
    def observation_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphRAGBenchmarkCase:
    gold: GraphRAGGoldCase
    observation: GraphRAGSelectionObservation

    def __post_init__(self) -> None:
        if not isinstance(self.gold, GraphRAGGoldCase):
            raise ValueError("gold must be GraphRAGGoldCase.")
        if not isinstance(self.observation, GraphRAGSelectionObservation):
            raise ValueError("observation must be GraphRAGSelectionObservation.")
        if (
            self.gold.graph_set_id != self.observation.graph_set_id
            or self.gold.graph_set_digest != self.observation.graph_set_digest
            or self.gold.query_digest != self.observation.query_digest
        ):
            raise ValueError("gold and selection observation identities differ.")

    @property
    def case_digest(self) -> str:
        return _sha256(
            {
                "gold": self.gold.case_digest,
                "observation": self.observation.observation_digest,
            }
        )


@dataclass(frozen=True)
class GraphRAGBenchmarkRun:
    run_id: str
    seed: int
    cases: tuple[GraphRAGBenchmarkCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id", 500))
        object.__setattr__(self, "seed", _integer(self.seed, "seed", 0, 2**63 - 1))
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases or len(self.cases) > _MAX_CASES_PER_RUN or any(
            not isinstance(value, GraphRAGBenchmarkCase) for value in self.cases
        ):
            raise ValueError("cases must contain bounded GraphRAGBenchmarkCase values.")
        query_ids = [value.gold.query_id for value in self.cases]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query IDs must be unique inside each run.")

    @property
    def run_contract_digest(self) -> str:
        return _sha256(
            {
                "run_id": self.run_id,
                "seed": self.seed,
                "gold_case_digests": [value.gold.case_digest for value in self.cases],
            }
        )

    @property
    def run_result_digest(self) -> str:
        return _sha256(
            {
                "run_contract_digest": self.run_contract_digest,
                "case_digests": [value.case_digest for value in self.cases],
            }
        )


@dataclass(frozen=True)
class GraphRAGBenchmarkFixture:
    benchmark_id: str
    runs: tuple[GraphRAGBenchmarkRun, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id", 500))
        if not isinstance(self.runs, tuple):
            object.__setattr__(self, "runs", tuple(self.runs))
        if not self.runs or len(self.runs) > _MAX_RUNS or any(
            not isinstance(value, GraphRAGBenchmarkRun) for value in self.runs
        ):
            raise ValueError("runs must contain bounded GraphRAGBenchmarkRun values.")
        if len({value.run_id for value in self.runs}) != len(self.runs):
            raise ValueError("run IDs must be unique.")
        contracts = [
            tuple(case.gold.case_digest for case in value.cases) for value in self.runs
        ]
        if any(value != contracts[0] for value in contracts[1:]):
            raise ValueError("every run must use the same ordered gold-case contract.")
        if self.schema_version != 1:
            raise ValueError("benchmark fixture schema is unsupported.")

    @property
    def benchmark_fingerprint(self) -> str:
        return _sha256(
            {
                "scope": "rigorousrag-evidence-graph-rag-benchmark-v1",
                "benchmark_id": self.benchmark_id,
                "ordered_gold_case_digests": [
                    value.gold.case_digest for value in self.runs[0].cases
                ],
                "run_seeds": [value.seed for value in self.runs],
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True)
class GraphRAGBenchmarkRunReport:
    run_id: str
    seed: int
    aggregate: GraphRAGAggregate
    run_contract_digest: str
    run_result_digest: str

    @property
    def report_digest(self) -> str:
        return _sha256(
            {
                "run_id": self.run_id,
                "seed": self.seed,
                "aggregate_digest": self.aggregate.aggregate_digest,
                "run_contract_digest": self.run_contract_digest,
                "run_result_digest": self.run_result_digest,
            }
        )


@dataclass(frozen=True)
class GraphRAGBenchmarkReport:
    benchmark_id: str
    benchmark_fingerprint: str
    run_count: int
    seed_count: int
    case_count_per_run: int
    run_reports: tuple[GraphRAGBenchmarkRunReport, ...]
    aggregate: GraphRAGAggregate
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id", 500))
        object.__setattr__(
            self,
            "benchmark_fingerprint",
            _digest(self.benchmark_fingerprint, "benchmark_fingerprint"),
        )
        object.__setattr__(self, "run_count", _integer(self.run_count, "run_count", 1, _MAX_RUNS))
        object.__setattr__(self, "seed_count", _integer(self.seed_count, "seed_count", 1, _MAX_RUNS))
        object.__setattr__(
            self,
            "case_count_per_run",
            _integer(self.case_count_per_run, "case_count_per_run", 1, _MAX_CASES_PER_RUN),
        )
        if not isinstance(self.run_reports, tuple) or len(self.run_reports) != self.run_count:
            raise ValueError("run_reports must match run_count.")
        if any(not isinstance(value, GraphRAGBenchmarkRunReport) for value in self.run_reports):
            raise ValueError("run_reports contain unsupported values.")
        if not isinstance(self.aggregate, GraphRAGAggregate):
            raise ValueError("aggregate must be GraphRAGAggregate.")
        if self.aggregate.case_count != self.run_count * self.case_count_per_run:
            raise ValueError("aggregate case count differs from report dimensions.")
        if self.schema_version != 1:
            raise ValueError("benchmark report schema is unsupported.")

    @property
    def report_digest(self) -> str:
        return _sha256(
            {
                "benchmark_id": self.benchmark_id,
                "benchmark_fingerprint": self.benchmark_fingerprint,
                "run_count": self.run_count,
                "seed_count": self.seed_count,
                "case_count_per_run": self.case_count_per_run,
                "run_report_digests": [value.report_digest for value in self.run_reports],
                "aggregate_digest": self.aggregate.aggregate_digest,
                "schema_version": self.schema_version,
            }
        )


def evaluate_graph_observation(
    observation: GraphRAGSelectionObservation,
    gold: GraphRAGGoldCase,
) -> GraphRAGEvaluation:
    if not isinstance(observation, GraphRAGSelectionObservation):
        raise ValueError("observation must be GraphRAGSelectionObservation.")
    if not isinstance(gold, GraphRAGGoldCase):
        raise ValueError("gold must be GraphRAGGoldCase.")
    if (
        observation.graph_set_id != gold.graph_set_id
        or observation.graph_set_digest != gold.graph_set_digest
        or observation.query_digest != gold.query_digest
    ):
        raise ValueError("observation and gold identities differ.")
    predicted_nodes = set(observation.selected_nodes)
    gold_nodes = set(gold.relevant_nodes)
    node_p, node_r, node_f1 = _prf(predicted_nodes, gold_nodes)
    predicted_docs = {value.doc_id for value in predicted_nodes}
    gold_docs = {value.doc_id for value in gold_nodes}
    doc_p, doc_r, doc_f1 = _prf(predicted_docs, gold_docs)
    predicted_edges = set(observation.traversal_edge_ids)
    gold_edges = set(gold.required_edge_ids)
    edge_p, edge_r, edge_f1 = _prf(predicted_edges, gold_edges)
    lineage = observation.expanded_lineage_valid
    lineage_completeness = 1.0 if not lineage else sum(lineage) / len(lineage)
    return GraphRAGEvaluation(
        query_id=gold.query_id,
        case_digest=gold.case_digest,
        selection_digest=observation.selection_digest,
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
        lineage_completeness=lineage_completeness,
        abstention_correct=observation.abstained == gold.should_abstain,
        evidence_count=observation.evidence_count,
        traversal_count=observation.traversal_count,
        estimated_work_units=observation.estimated_work_units,
    )


def run_graph_rag_benchmark(fixture: GraphRAGBenchmarkFixture) -> GraphRAGBenchmarkReport:
    if not isinstance(fixture, GraphRAGBenchmarkFixture):
        raise ValueError("fixture must be GraphRAGBenchmarkFixture.")
    reports: list[GraphRAGBenchmarkRunReport] = []
    all_evaluations: list[GraphRAGEvaluation] = []
    for run in fixture.runs:
        evaluations = tuple(
            evaluate_graph_observation(value.observation, value.gold)
            for value in run.cases
        )
        aggregate = aggregate_graph_evaluations(evaluations)
        reports.append(
            GraphRAGBenchmarkRunReport(
                run_id=run.run_id,
                seed=run.seed,
                aggregate=aggregate,
                run_contract_digest=run.run_contract_digest,
                run_result_digest=run.run_result_digest,
            )
        )
        all_evaluations.extend(evaluations)
    return GraphRAGBenchmarkReport(
        benchmark_id=fixture.benchmark_id,
        benchmark_fingerprint=fixture.benchmark_fingerprint,
        run_count=len(fixture.runs),
        seed_count=len({value.seed for value in fixture.runs}),
        case_count_per_run=len(fixture.runs[0].cases),
        run_reports=tuple(reports),
        aggregate=aggregate_graph_evaluations(tuple(all_evaluations)),
    )


def _observation_from_mapping(value: Mapping[str, Any]) -> GraphRAGSelectionObservation:
    if not isinstance(value, Mapping) or set(value) != {
        "graph_set_id",
        "graph_set_digest",
        "query_digest",
        "selection_digest",
        "selected_nodes",
        "traversal_edge_ids",
        "expanded_lineage_valid",
        "abstained",
        "evidence_count",
        "traversal_count",
        "estimated_work_units",
    }:
        raise ValueError("graph RAG selection observation schema is invalid.")
    if not isinstance(value["selected_nodes"], list):
        raise ValueError("selected_nodes must be a JSON array.")
    return GraphRAGSelectionObservation(
        graph_set_id=value["graph_set_id"],
        graph_set_digest=value["graph_set_digest"],
        query_digest=value["query_digest"],
        selection_digest=value["selection_digest"],
        selected_nodes=tuple(GraphNodeLocator(**item) for item in value["selected_nodes"]),
        traversal_edge_ids=tuple(value["traversal_edge_ids"]),
        expanded_lineage_valid=tuple(value["expanded_lineage_valid"]),
        abstained=value["abstained"],
        evidence_count=value["evidence_count"],
        traversal_count=value["traversal_count"],
        estimated_work_units=value["estimated_work_units"],
    )


def fixture_from_mapping(value: Mapping[str, Any]) -> GraphRAGBenchmarkFixture:
    if not isinstance(value, Mapping) or set(value) != {
        "benchmark_id",
        "runs",
        "schema_version",
    }:
        raise ValueError("graph RAG benchmark fixture schema is invalid.")
    if not isinstance(value["runs"], list):
        raise ValueError("runs must be a JSON array.")
    runs: list[GraphRAGBenchmarkRun] = []
    for raw_run in value["runs"]:
        if not isinstance(raw_run, Mapping) or set(raw_run) != {"run_id", "seed", "cases"}:
            raise ValueError("graph RAG benchmark run schema is invalid.")
        if not isinstance(raw_run["cases"], list):
            raise ValueError("cases must be a JSON array.")
        cases: list[GraphRAGBenchmarkCase] = []
        for raw_case in raw_run["cases"]:
            if not isinstance(raw_case, Mapping) or set(raw_case) != {"gold", "observation"}:
                raise ValueError("graph RAG benchmark case schema is invalid.")
            cases.append(
                GraphRAGBenchmarkCase(
                    gold=gold_case_from_mapping(raw_case["gold"]),
                    observation=_observation_from_mapping(raw_case["observation"]),
                )
            )
        runs.append(
            GraphRAGBenchmarkRun(
                run_id=raw_run["run_id"],
                seed=raw_run["seed"],
                cases=tuple(cases),
            )
        )
    return GraphRAGBenchmarkFixture(
        benchmark_id=value["benchmark_id"],
        runs=tuple(runs),
        schema_version=value["schema_version"],
    )


__all__ = [
    "GraphRAGBenchmarkCase",
    "GraphRAGBenchmarkFixture",
    "GraphRAGBenchmarkReport",
    "GraphRAGBenchmarkRun",
    "GraphRAGBenchmarkRunReport",
    "GraphRAGSelectionObservation",
    "evaluate_graph_observation",
    "fixture_from_mapping",
    "run_graph_rag_benchmark",
]
