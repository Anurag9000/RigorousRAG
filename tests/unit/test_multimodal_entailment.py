import hashlib

import pytest

from tools.model_artifacts import ModelArtifactSpec
from tools.multimodal_entailment import (
    GovernedImageTextEntailmentAdapter,
    GovernedTableChartEntailmentAdapter,
    RegionPayload,
)
from tools.multimodal_evidence import NormalizedBBox, build_evidence_region, content_digest


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec(kind: str) -> ModelArtifactSpec:
    return ModelArtifactSpec(
        kind=kind,
        model_id="org/model",
        revision="v1",
        config_sha256=digest("config"),
        weights_sha256=digest("weights"),
        tokenizer_sha256=digest("tokenizer"),
        languages=("en",),
    )


def payload(kind: str, value: bytes = b"region bytes") -> RegionPayload:
    region = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=digest("source"),
        page_number=3,
        kind=kind,
        bbox=NormalizedBBox(0.1, 0.2, 0.8, 0.9),
        content_sha256=content_digest(value),
        extractor_id="layout-v1",
    )
    return RegionPayload(region=region, payload=value)


def test_image_text_decision_is_bound_to_server_coordinate_citation():
    adapter = GovernedImageTextEntailmentAdapter(
        spec("image_text"),
        lambda claim, data: {
            "label": "entailed",
            "score": 0.92,
            "rationale_code": "direct_match",
            "private_rationale": "must not escape",
        },
    )
    evidence = payload("figure")
    result = adapter.evaluate("the figure shows an increase", evidence)
    assert result.region_id == evidence.region.region_id
    assert result.citation.region_id == evidence.region.region_id
    assert result.citation.page_number == 3
    assert result.label == "entailed"
    assert result.rationale_code == "direct_match"
    assert not hasattr(result, "private_rationale")


def test_region_payload_must_match_authoritative_content_digest():
    good = payload("chart")
    with pytest.raises(ValueError, match="content digest"):
        RegionPayload(region=good.region, payload=b"substituted")


def test_table_chart_adapter_preserves_region_kind_and_numeric_conflict():
    seen = []

    def infer(claim, data, kind):
        seen.append((claim, data, kind))
        return {
            "label": "contradicted",
            "score": 0.88,
            "rationale_code": "numeric_conflict",
        }

    adapter = GovernedTableChartEntailmentAdapter(spec("table_chart"), infer)
    evidence = payload("table")
    result = adapter.evaluate("revenue was 12", evidence)
    assert result.label == "contradicted"
    assert result.rationale_code == "numeric_conflict"
    assert seen[0][2] == "table"


def test_adapter_kind_boundaries_fail_closed():
    image = GovernedImageTextEntailmentAdapter(
        spec("image_text"),
        lambda *_args: {"label": "entailed", "score": 1.0, "rationale_code": "direct_match"},
    )
    with pytest.raises(ValueError, match="figure or chart"):
        image.evaluate("claim", payload("table"))

    table = GovernedTableChartEntailmentAdapter(
        spec("table_chart"),
        lambda *_args: {"label": "entailed", "score": 1.0, "rationale_code": "direct_match"},
    )
    with pytest.raises(ValueError, match="table or chart"):
        table.evaluate("claim", payload("figure"))


def test_private_model_failure_is_replaced_by_bounded_error():
    adapter = GovernedImageTextEntailmentAdapter(
        spec("image_text"),
        lambda *_args: (_ for _ in ()).throw(RuntimeError("secret provider details")),
    )
    with pytest.raises(RuntimeError, match="image-text entailment execution failed") as captured:
        adapter.evaluate("claim", payload("figure"))
    assert "secret provider details" not in str(captured.value)
