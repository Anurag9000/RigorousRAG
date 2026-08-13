"""Validated domain-aware route selection with deterministic fallback.

Injected classifiers may improve routing, but unsupported labels, malformed scores,
or low confidence never bypass the existing deterministic route policy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tools.adaptive_route_experiments import ROUTES, RouteExperimentCase, select_route

_MAX_DOMAINS = 128
_MAX_QUERY = 20_000
_ROUTE_SET = frozenset(ROUTES)
_SCOPES = frozenset({"uploaded", "public", "mixed"})

DomainClassifier = Callable[[str], Mapping[str, float]]


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _routes(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("available_routes must be an iterable of route names.")
    try:
        rows = tuple(dict.fromkeys(values))
    except Exception as exc:
        raise ValueError("available_routes is not safely iterable.") from exc
    if not rows or len(rows) > len(ROUTES) or any(route not in _ROUTE_SET for route in rows):
        raise ValueError("available_routes contains an unsupported route.")
    return rows


def _domain_routes(values: Mapping[str, str] | None) -> dict[str, str]:
    defaults = {
        "general": "corpus-hybrid",
        "scholarly": "scholarly",
        "scientific": "scholarly",
        "research": "scholarly",
        "current": "web",
        "news": "web",
        "exact": "corpus-sparse",
        "structured": "corpus-hybrid",
    }
    if values is None:
        return defaults
    if not isinstance(values, Mapping) or not 1 <= len(values) <= _MAX_DOMAINS:
        raise ValueError("domain_routes must be a bounded non-empty mapping.")
    result: dict[str, str] = {}
    for raw_domain, raw_route in values.items():
        domain = _identifier(raw_domain, "domain", 100).lower()
        route = _identifier(raw_route, "route", 100)
        if route not in _ROUTE_SET:
            raise ValueError(f"unsupported route for domain {domain!r}.")
        result[domain] = route
    return result


@dataclass(frozen=True)
class DomainRoutingDecision:
    domain: str
    confidence: float
    route: str
    source: str
    classifier_version: str | None
    fallback_route: str

    @property
    def used_classifier(self) -> bool:
        return self.source == "classifier"


def route_query_by_domain(
    query: str,
    *,
    scope: str = "mixed",
    classifier: DomainClassifier | None = None,
    classifier_version: str | None = None,
    confidence_threshold: float = 0.70,
    domain_routes: Mapping[str, str] | None = None,
    available_routes: Iterable[str] = ROUTES,
) -> DomainRoutingDecision:
    """Route a query using a validated classifier or the existing heuristic fallback."""

    if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY:
        raise ValueError("query must be a bounded non-empty string.")
    if scope not in _SCOPES:
        raise ValueError("scope must be uploaded, public, or mixed.")
    threshold = _unit(confidence_threshold, "confidence_threshold")
    available = _routes(available_routes)
    mapping = _domain_routes(domain_routes)
    fallback_case = RouteExperimentCase("domain-routing-fallback", query, scope=scope, domain="general")
    fallback = select_route(fallback_case, available_routes=available)
    if classifier is None:
        return DomainRoutingDecision("general", 0.0, fallback, "fallback", None, fallback)
    if not callable(classifier):
        raise ValueError("classifier must be callable.")
    version = _identifier(classifier_version, "classifier_version", 200) if classifier_version is not None else None
    if version is None:
        raise ValueError("classifier_version is required when classifier is supplied.")
    try:
        raw = classifier(query)
    except Exception:
        return DomainRoutingDecision("general", 0.0, fallback, "fallback", version, fallback)
    if not isinstance(raw, Mapping) or not raw or len(raw) > _MAX_DOMAINS:
        return DomainRoutingDecision("general", 0.0, fallback, "fallback", version, fallback)
    scores: list[tuple[str, float]] = []
    for raw_domain, raw_score in raw.items():
        try:
            domain = _identifier(raw_domain, "domain", 100).lower()
            score = _unit(raw_score, "domain score")
        except ValueError:
            continue
        if domain in mapping:
            scores.append((domain, score))
    if not scores:
        return DomainRoutingDecision("general", 0.0, fallback, "fallback", version, fallback)
    domain, confidence = max(scores, key=lambda item: (item[1], item[0]))
    route = mapping[domain]
    if confidence < threshold or route not in available:
        return DomainRoutingDecision(domain, confidence, fallback, "fallback", version, fallback)
    return DomainRoutingDecision(domain, confidence, route, "classifier", version, fallback)


__all__ = ["DomainClassifier", "DomainRoutingDecision", "route_query_by_domain"]
