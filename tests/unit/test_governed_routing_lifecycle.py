from __future__ import annotations

from evaluation.resource_measurement import ProviderUsage, measure_call
from tools.adapter_registry import AdapterRegistry, AdapterVersion
from tools.adaptive_retrieval import RetrievalAttempt, analyze_query
from tools.adaptive_retrieval_runner import run_adaptive_retrieval
from tools.adaptive_trace_governance import (
    TraceRetentionPolicy,
    apply_trace_retention,
    export_privacy_safe_traces,
)
from tools.adaptive_trace_store import AdaptiveTraceStore
from tools.domain_routing import route_query_by_domain
from tools.embedding_registry import resolve_embedding_profile
from tools.governed_adapter_factories import (
    make_governed_adapter_factory,
    request_from_active_adapter,
)
from tools.plan_ranking import PlanTrainingExample, fit_pairwise_plan_ranker, heuristic_rank_attempts
from tools.query_normalization import normalize_query_context


def _trace_result():
    def search(_query: str, **_kwargs):
        return (
            {"doc_id": "doc-a", "source_id": "a", "score": 0.95, "source_kind": "local"},
            {"doc_id": "doc-b", "source_id": "b", "score": 0.90, "source_kind": "local"},
        )

    return run_adaptive_retrieval("trace query", search=search, owner_id="alice", max_attempts=2)


def test_trace_retention_and_privacy_safe_export(tmp_path) -> None:
    store = AdaptiveTraceStore(tmp_path / "adaptive-traces.db")
    result = _trace_result()
    store.record_result(query="secret query one", owner_id="alice", result=result, run_id="run-1", started_at=100.0, completed_at=101.0)
    store.record_result(query="secret query two", owner_id="alice", result=result, run_id="run-2", started_at=200.0, completed_at=202.0)
    policy = TraceRetentionPolicy(retain_latest=2, export_limit=2, include_attempts=True)
    exported = export_privacy_safe_traces(store, owner_id="alice", export_secret="0123456789abcdef-secret", policy=policy)
    assert len(exported) == 2
    assert exported[0].owner_pseudonym != "alice"
    assert exported[0].run_pseudonym not in {"run-1", "run-2"}
    assert len(exported[0].query_sha256) == 64
    assert exported[0].attempts
    rendered = repr(exported)
    assert "secret query one" not in rendered
    assert "secret query two" not in rendered
    report = apply_trace_retention(store, owner_id="alice", policy=TraceRetentionPolicy(retain_latest=1, export_limit=1))
    assert report.deleted_runs == 1
    assert report.retention_cap == 1
    assert report.retained_sample_size == 1
    assert len(store.list_runs(owner_id="alice", limit=10)) == 1


def test_domain_routing_uses_validated_classifier_and_fallback() -> None:
    classified = route_query_by_domain(
        "Compare two retrieval methods",
        classifier=lambda _query: {"scientific": 0.95, "general": 0.05},
        classifier_version="domain-model-1",
    )
    assert classified.used_classifier is True
    assert classified.domain == "scientific"
    assert classified.route == "scholarly"
    fallback = route_query_by_domain(
        "What changed in the latest release?",
        classifier=lambda _query: {"scientific": 0.4},
        classifier_version="domain-model-1",
        confidence_threshold=0.8,
    )
    assert fallback.used_classifier is False
    assert fallback.route == "web"


def test_entity_and_temporal_normalization_is_explicit_and_bounded() -> None:
    context = normalize_query_context(
        "Compare RAG-X on 2024-05-01 with its 2023 result.",
        entity_aliases={"RAG-X": ("system:rag-x", "RAG X")},
    )
    assert len(context.entities) == 1
    assert context.entities[0].canonical_id == "system:rag-x"
    assert context.entity_source == "fallback"
    assert [item.precision for item in context.temporal_ranges] == ["year", "day"] or [item.precision for item in context.temporal_ranges] == ["day", "year"]
    assert {item.surface for item in context.temporal_ranges} == {"2024-05-01", "2023"}
    assert all(item.end_utc >= item.start_utc for item in context.temporal_ranges)


def test_pairwise_plan_ranker_learns_preferred_attempt() -> None:
    analysis = analyze_query("Find evidence for DOI 10.1234/example")
    sparse = RetrievalAttempt("corpus-sparse", 5, 10)
    dense = RetrievalAttempt("dense", 5, 10)
    example = PlanTrainingExample(analysis, sparse, dense)
    ranker = fit_pairwise_plan_ranker((example,), version="ranker-1", epochs=80, learning_rate=0.1)
    ranked = ranker.rank(analysis, (dense, sparse))
    assert ranked[0].attempt == sparse
    assert ranker.score(analysis, sparse) > ranker.score(analysis, dense)
    assert heuristic_rank_attempts(analysis, (dense, sparse))[0].attempt == sparse


def test_resource_measurement_uses_real_counters_and_provider_usage() -> None:
    measured = measure_call(
        lambda: sum(range(5_000)),
        provider_usage=ProviderUsage(prompt_tokens=10, completion_tokens=4, cost_units=0.25),
    )
    assert measured.value == sum(range(5_000))
    assert measured.usage.wall_ms >= 0.0
    assert measured.usage.cpu_ms >= 0.0
    assert measured.usage.python_peak_allocated_bytes >= 0
    assert measured.usage.process_peak_rss_bytes is None or measured.usage.process_peak_rss_bytes >= 0
    assert measured.usage.provider.prompt_tokens == 10
    assert measured.usage.provider.completion_tokens == 4
    assert measured.usage.provider.cost_units == 0.25


def test_governed_adapter_factories_use_active_registry_and_validate_dimensions() -> None:
    registry = AdapterRegistry()
    registry.register(
        AdapterVersion(
            name="embedding-adapter",
            version="1.2.3",
            kind="embedding",
            artifact_uri="memory://embedding-adapter",
            checksum_sha256="a" * 64,
        )
    )
    registry.promote("embedding-adapter", "1.2.3")

    class Backend:
        def __init__(self, dimensions: int) -> None:
            self.dimensions = dimensions

        def encode_passages(self, passages):
            return tuple(tuple(0.01 for _ in range(self.dimensions)) for _ in passages)

    for alias in ("instructor-base", "specter2", "bge-m3"):
        request = request_from_active_adapter(
            registry,
            profile_alias=alias,
            adapter_name="embedding-adapter",
            pinned_revision="revision-deadbeef",
        )
        assert request.allow_download is False
        assert request.trust_remote_code is False
        profile = resolve_embedding_profile(alias, allow_compatibility=False)
        factory = make_governed_adapter_factory(request, lambda selected, _request: Backend(selected.dimensions or 1))
        encoder = factory(profile)
        vectors = encoder.encode_passages(("bounded passage",))
        assert len(vectors) == 1
        assert len(vectors[0]) == profile.dimensions
        assert len(encoder.model_instance_id) == 64
