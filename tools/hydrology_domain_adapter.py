"""Hydrology domain adapter for the generic scientific-domain registry."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

from tools.document_ir import ScientificDocumentIR
from tools.domain_adapter import DomainDescriptor, DomainQueryFeatures, ScientificDomainAdapter
from tools.graph_reasoning import GraphEdge, GraphNode

_KEYWORDS = {
    "rainfall": ("rainfall", "precipitation", "chirps", "hyetograph", "idf"),
    "hydrology": ("hydrograph", "runoff", "catchment", "watershed", "basin", "hec-hms"),
    "hydraulics": ("hec-ras", "water surface", "stage", "discharge", "velocity", "inundation", "floodplain"),
    "reservoir": ("reservoir", "dam", "spillway", "gate", "storage", "frl", "mwl", "breach"),
    "geospatial": ("raster", "geotiff", "crs", "epsg", "dem", "reach", "river"),
}
_VALUE_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:m3/s|cumecs?|mm/h|mm|mcm|m\b|km2|km²|ft\b)", re.I)


class HydrologyDomainAdapter(ScientificDomainAdapter):
    @property
    def descriptor(self) -> DomainDescriptor:
        return DomainDescriptor(
            domain_id="hydrology",
            version="1.0.0",
            label="Hydrology, hydraulic modelling and flood evidence",
            supported_mime_types=("application/pdf", "text/csv", "application/json", "image/tiff"),
            supported_languages=("en", "multilingual"),
            capabilities=("timeseries", "raster", "geospatial", "hec-hms", "hec-ras", "chirps", "unit-reasoning"),
            metadata_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_type": {"type": "string"},
                    "scenario_id": {"type": "string"},
                    "crs": {"type": "string"},
                    "variable": {"type": "string"},
                    "unit": {"type": "string"},
                    "location_id": {"type": "string"},
                },
            },
        )

    def normalize_metadata(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(metadata, Mapping) or len(metadata) > 128:
            raise ValueError("hydrology metadata must be a bounded mapping")
        allowed = {"model_type", "scenario_id", "crs", "variable", "unit", "location_id"}
        output: dict[str, str] = {}
        for key, value in metadata.items():
            name = str(key).strip().lower()
            if name not in allowed or value is None:
                continue
            text = " ".join(str(value).replace("\x00", " ").split())
            if text and len(text) <= 1000:
                output[name] = text
        if "model_type" in output:
            aliases = {"hms": "hec-hms", "hechms": "hec-hms", "ras": "hec-ras", "hecras": "hec-ras"}
            output["model_type"] = aliases.get(output["model_type"].casefold(), output["model_type"].casefold())
        return output

    def query_features(self, query: str) -> DomainQueryFeatures:
        if not isinstance(query, str) or not query.strip() or len(query) > 20_000:
            raise ValueError("query is invalid")
        normalized = " ".join(query.casefold().split())
        scores: dict[str, float] = {}
        terms: list[str] = []
        for group, keywords in _KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword in normalized]
            if hits:
                scores[group] = min(1.0, 0.35 + 0.2 * len(hits))
                terms.extend(hits)
            else:
                scores[group] = 0.0
        if _VALUE_UNIT_RE.search(query):
            scores["quantitative"] = 0.85
        if any(token in normalized for token in ("compare", "versus", " vs ", "difference", "peak")):
            scores["scenario_comparison"] = 0.75
        filters: dict[str, Any] = {}
        if "hec-ras" in normalized:
            filters["model_type"] = "hec-ras"
        elif "hec-hms" in normalized:
            filters["model_type"] = "hec-hms"
        elif "chirps" in normalized:
            filters["source_family"] = "chirps"
        return DomainQueryFeatures("hydrology", scores, filters, tuple(dict.fromkeys(terms)))

    def unit_aliases(self) -> Mapping[str, str]:
        return {
            "cumec": "m3/s",
            "cumecs": "m3/s",
            "cms": "m3/s",
            "m³/s": "m3/s",
            "m^3/s": "m3/s",
            "mcm": "MCM",
            "million cubic metres": "MCM",
            "km²": "km2",
            "km^2": "km2",
        }

    def enrich_document_graph(self, document: ScientificDocumentIR) -> tuple[Sequence[GraphNode], Sequence[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for block in document.blocks:
            text = block.text.casefold()
            matched_groups = [group for group, keywords in _KEYWORDS.items() if any(keyword in text for keyword in keywords)]
            for group in matched_groups:
                content_sha = hashlib.sha256(f"{document.source_sha256}:{block.block_id}:{group}".encode("utf-8")).hexdigest()
                node_id = hashlib.sha256(f"hydrology:{document.doc_id}:{block.block_id}:{group}".encode("utf-8")).hexdigest()
                nodes.append(
                    GraphNode(
                        node_id=node_id,
                        kind="entity",
                        source_id=document.doc_id,
                        content_sha256=content_sha,
                        label=group,
                        attributes={"domain": "hydrology", "block_id": block.block_id, "page_number": str(block.page_number)},
                    )
                )
        return tuple(nodes), tuple(edges)

    def report_fields(self) -> Sequence[str]:
        return (
            "model_type",
            "scenario_id",
            "location_id",
            "variable",
            "unit",
            "peak_value",
            "peak_time",
            "integrated_volume",
            "crs",
            "source_artifact",
        )


__all__ = ["HydrologyDomainAdapter"]
