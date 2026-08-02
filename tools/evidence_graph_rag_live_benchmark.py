"""Live execution bridge from authoritative GraphRAG selection to text-free reports."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from tools.evidence_graph_rag import (
    GraphEvidenceSelection,
    select_current_graph_set_evidence,
)
from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkCase,
    GraphRAGBenchmarkFixture,
    GraphRAGBenchmarkReport,
    GraphRAGBenchmarkRun,
    GraphRAGSelectionObservation,
    run_graph_rag_benchmark,
)
from tools.evidence_graph_rag_evaluation import (
    GraphNodeLocator,
    GraphRAGGoldCase,
    gold_case_from_mapping,
)

_MAX_RUNS = 10_000
_MAX_CASES = 1_000_000
_ALLOWED_SELECTOR_KEYS = frozenset(
    {
        "node_types",
        "within_edge_types",
        "cross_edge_types",
        "per_document_hits",
        "max_lexical_seeds",
        "max_within_per_seed",
        "max_cross_depth",
        "max_cross_per_seed",
        "max_total_items",
    }
)


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _query(value: Any, expected_digest: str) -> str:
    if not isinstance(value, str):
        raise ValueError("query resolver must return text.")
    cleaned = value.strip()
    if not cleaned or "\x00" in cleaned or len(cleaned) > 20_000:
        raise ValueError("resolved query is empty, invalid or too long.")
    if hashlib.sha256(cleaned.encode("utf-8")).hexdigest() != expected_digest:
        raise ValueError("resolved query digest differs from the governed gold case.")
    return cleaned


def _selector_config(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not set(value) <= _ALLOWED_SELECTOR_KEYS:
        raise ValueError("selector_config contains unsupported fields.")
    result = dict(value)
    for name in ("node_types", "within_edge_types", "cross_edge_types"):
        if name in result:
            raw = result[name]
            if not isinstance(raw, (list, tuple)) or not raw or any(
                not isinstance(item, str) for item in raw
            ):
                raise ValueError(f"selector_config.{name} must be non-empty text values.")
            result[name] = tuple(raw)
    for name in set(result) - {"node_types", "within_edge_types", "cross_edge_types"}:
        result[name] = _integer(result[name], f"selector_config.{name}", 0, 10_000_000)
    return result


@dataclass(frozen=True)
class GraphRAGLiveBenchmarkPlan:
    benchmark_id: str
    run_seeds: tuple[int, ...]
    gold_cases: tuple[GraphRAGGoldCase, ...]
    selector_config: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id", 500))
        seeds = tuple(
            _integer(value, "run_seed", 0, 2**63 - 1) for value in self.run_seeds
        )
        if not seeds or len(seeds) > _MAX_RUNS or len(set(seeds)) != len(seeds):
            raise ValueError("run_seeds must be bounded and unique.")
        object.__setattr__(self, "run_seeds", seeds)
        if not isinstance(self.gold_cases, tuple):
            object.__setattr__(self, "gold_cases", tuple(self.gold_cases))
        cases = tuple(self.gold_cases)
        if not cases or len(cases) > _MAX_CASES or any(
            not isinstance(value, GraphRAGGoldCase) for value in cases
        ):
            raise ValueError("gold_cases must contain bounded GraphRAGGoldCase values.")
        if len({value.query_id for value in cases}) != len(cases):
            raise ValueError("gold case query IDs must be unique.")
        object.__setattr__(self, "gold_cases", cases)
        object.__setattr__(self, "selector_config", _selector_config(self.selector_config))
        if self.schema_version != 1:
            raise ValueError("live benchmark plan schema is unsupported.")

    @property
    def plan_fingerprint(self) -> str:
        import json

        return hashlib.sha256(
            json.dumps(
                {
                    "scope": "rigorousrag-live-evidence-graph-benchmark-v1",
                    "benchmark_id": self.benchmark_id,
                    "run_seeds": self.run_seeds,
                    "gold_case_digests": [value.case_digest for value in self.gold_cases],
                    "selector_config": dict(self.selector_config),
                    "schema_version": self.schema_version,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class GraphRAGLiveBenchmarkResult:
    plan_fingerprint: str
    report: GraphRAGBenchmarkReport
    query_text_persisted: bool = False
    evidence_text_persisted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.plan_fingerprint, str) or len(self.plan_fingerprint) != 64:
            raise ValueError("plan_fingerprint must be a SHA-256 digest.")
        if not isinstance(self.report, GraphRAGBenchmarkReport):
            raise ValueError("report must be GraphRAGBenchmarkReport.")
        if self.query_text_persisted or self.evidence_text_persisted:
            raise ValueError("live benchmark results may not persist query/evidence text.")


def observation_from_selection(selection: GraphEvidenceSelection) -> GraphRAGSelectionObservation:
    if not isinstance(selection, GraphEvidenceSelection):
        raise ValueError("selection must be GraphEvidenceSelection.")
    step_digests = {value.step_digest for value in selection.traversals}
    expanded = [value for value in selection.items if value.origin != "lexical"]
    return GraphRAGSelectionObservation(
        graph_set_id=selection.graph_set_id,
        graph_set_digest=selection.graph_set_digest,
        query_digest=selection.query_digest,
        selection_digest=selection.selection_digest,
        selected_nodes=tuple(
            GraphNodeLocator(value.doc_id, value.generation, value.node_id)
            for value in selection.items
        ),
        traversal_edge_ids=tuple(value.edge_id for value in selection.traversals),
        expanded_lineage_valid=tuple(
            bool(value.lineage_step_digests)
            and all(item in step_digests for item in value.lineage_step_digests)
            for value in expanded
        ),
        abstained=selection.abstained,
        evidence_count=len(selection.items),
        traversal_count=len(selection.traversals),
        estimated_work_units=selection.estimated_work_units,
    )


def execute_live_graph_rag_benchmark(
    plan: GraphRAGLiveBenchmarkPlan,
    *,
    query_resolver: Callable[[str], str],
    selection_runner: Callable[..., GraphEvidenceSelection],
) -> GraphRAGLiveBenchmarkResult:
    """Execute governed cases while reducing query/evidence text immediately."""

    if not isinstance(plan, GraphRAGLiveBenchmarkPlan):
        raise ValueError("plan must be GraphRAGLiveBenchmarkPlan.")
    if not callable(query_resolver) or not callable(selection_runner):
        raise ValueError("query_resolver and selection_runner must be callable.")
    runs: list[GraphRAGBenchmarkRun] = []
    for run_index, seed in enumerate(plan.run_seeds):
        cases: list[GraphRAGBenchmarkCase] = []
        for gold in plan.gold_cases:
            query = _query(query_resolver(gold.query_id), gold.query_digest)
            selection = selection_runner(
                query=query,
                query_id=gold.query_id,
                seed=seed,
                selector_config=dict(plan.selector_config),
            )
            observation = observation_from_selection(selection)
            cases.append(GraphRAGBenchmarkCase(gold=gold, observation=observation))
            del query
            del selection
        runs.append(
            GraphRAGBenchmarkRun(
                run_id=f"run-{run_index:04d}-seed-{seed}",
                seed=seed,
                cases=tuple(cases),
            )
        )
    fixture = GraphRAGBenchmarkFixture(
        benchmark_id=plan.benchmark_id,
        runs=tuple(runs),
    )
    return GraphRAGLiveBenchmarkResult(
        plan_fingerprint=plan.plan_fingerprint,
        report=run_graph_rag_benchmark(fixture),
    )


def execute_authoritative_graph_rag_benchmark(
    plan: GraphRAGLiveBenchmarkPlan,
    *,
    owner_id: str,
    graph_set_key: str,
    query_resolver: Callable[[str], str],
    set_store: Any,
    generations: Any,
    graphs: Any,
) -> GraphRAGLiveBenchmarkResult:
    """Execute the governed plan through the actual current-set selector."""

    owner = _identifier(owner_id, "owner_id", 200)
    key = _identifier(graph_set_key, "graph_set_key", 500)

    def runner(
        *,
        query: str,
        query_id: str,
        seed: int,
        selector_config: Mapping[str, Any],
    ) -> GraphEvidenceSelection:
        # The deterministic baseline currently does not consume a seed. The seed is
        # retained in the benchmark contract for future governed stochastic adapters.
        del query_id, seed
        return select_current_graph_set_evidence(
            owner_id=owner,
            graph_set_key=key,
            query=query,
            set_store=set_store,
            generations=generations,
            graphs=graphs,
            **dict(selector_config),
        )

    return execute_live_graph_rag_benchmark(
        plan,
        query_resolver=query_resolver,
        selection_runner=runner,
    )


def plan_from_mapping(value: Mapping[str, Any]) -> GraphRAGLiveBenchmarkPlan:
    if not isinstance(value, Mapping) or set(value) != {
        "benchmark_id",
        "run_seeds",
        "gold_cases",
        "selector_config",
        "schema_version",
    }:
        raise ValueError("live graph RAG benchmark plan schema is invalid.")
    if not isinstance(value["gold_cases"], list) or not isinstance(value["run_seeds"], list):
        raise ValueError("gold_cases and run_seeds must be JSON arrays.")
    return GraphRAGLiveBenchmarkPlan(
        benchmark_id=value["benchmark_id"],
        run_seeds=tuple(value["run_seeds"]),
        gold_cases=tuple(gold_case_from_mapping(item) for item in value["gold_cases"]),
        selector_config=value["selector_config"],
        schema_version=value["schema_version"],
    )


__all__ = [
    "GraphRAGLiveBenchmarkPlan",
    "GraphRAGLiveBenchmarkResult",
    "execute_authoritative_graph_rag_benchmark",
    "execute_live_graph_rag_benchmark",
    "observation_from_selection",
    "plan_from_mapping",
]
