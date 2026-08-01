"""Bounded deterministic query decomposition and dependency planning."""

from __future__ import annotations

import hashlib
import itertools
import json
import operator
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_MAX_QUERY_CHARS = 20_000
_MAX_SUBQUESTIONS = 12
_MAX_TEXT_CHARS = 4_000
_MAX_ENTITIES = 20
_MAX_CONSTRAINTS = 20
_NODE_KEYS = {
    "question_id", "text", "depends_on", "entities",
    "temporal_constraints", "relation",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,120})[\"']")
_CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9-]{1,40})(?:\s+[A-Z][A-Za-z0-9-]{1,40}){0,4}\b"
)
_SPLIT_RE = re.compile(
    r"\s*(?:;|\band then\b|\bthen\b|\balso\b|\bwhile\b)\s*",
    flags=re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:compare|contrast)\s+(.{1,300}?)\s+"
    r"(?:and|with|versus|vs\.?|against)\s+(.{1,300}?)(?:[?.!]|$)",
    flags=re.IGNORECASE,
)
_VERSUS_RE = re.compile(
    r"\b(.{1,200}?)\s+(?:versus|vs\.?|against)\s+"
    r"(.{1,200}?)(?:[?.!]|$)",
    flags=re.IGNORECASE,
)
_TEMPORAL_WORDS = {
    "before", "after", "during", "between", "since", "until",
    "latest", "recent", "historical", "timeline",
}


def _contains_control(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) == 127
        for character in value
    )


def _bounded_text(value: Any, label: str, maximum: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    if not value or len(value) > maximum or _contains_control(value):
        raise ValueError(f"{label} must contain 1-{maximum} valid characters.")
    rendered = " ".join(value.split())
    if not rendered:
        raise ValueError(f"{label} must contain 1-{maximum} valid characters.")
    return rendered


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _bounded_items(values: Any, label: str, maximum: int) -> list[Any]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence.")
    try:
        rows = list(itertools.islice(iter(values), maximum + 1))
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    if len(rows) > maximum:
        raise ValueError(f"{label} may contain at most {maximum} values.")
    return rows


def _bounded_strings(
    values: Any,
    *,
    label: str,
    maximum_items: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    if values is None:
        return ()
    rows = _bounded_items(values, label, maximum_items)
    result: list[str] = []
    seen: set[str] = set()
    for item in rows:
        text = _bounded_text(item, label, maximum_chars)
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class Subquestion:
    question_id: str
    text: str
    depends_on: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    temporal_constraints: tuple[str, ...] = ()
    relation: str = "lookup"

    def __post_init__(self) -> None:
        identifier = _bounded_text(self.question_id, "question_id", 64)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", identifier):
            raise ValueError("question_id contains invalid characters.")
        object.__setattr__(self, "question_id", identifier)
        object.__setattr__(self, "text", _bounded_text(self.text, "subquestion"))
        dependencies = _bounded_strings(
            self.depends_on,
            label="depends_on",
            maximum_items=_MAX_SUBQUESTIONS,
            maximum_chars=64,
        )
        if identifier in dependencies:
            raise ValueError("a subquestion cannot depend on itself.")
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(
            self,
            "entities",
            _bounded_strings(
                self.entities,
                label="entities",
                maximum_items=_MAX_ENTITIES,
                maximum_chars=200,
            ),
        )
        object.__setattr__(
            self,
            "temporal_constraints",
            _bounded_strings(
                self.temporal_constraints,
                label="temporal_constraints",
                maximum_items=_MAX_CONSTRAINTS,
                maximum_chars=200,
            ),
        )
        relation = _bounded_text(self.relation, "relation", 64).lower()
        if relation not in {"lookup", "compare", "explain", "synthesize", "temporal"}:
            raise ValueError("relation is invalid.")
        object.__setattr__(self, "relation", relation)


@dataclass(frozen=True)
class DecompositionPlan:
    query: str
    subquestions: tuple[Subquestion, ...]
    batches: tuple[tuple[str, ...], ...]
    terminal_questions: tuple[str, ...]
    fingerprint: str

    def by_id(self) -> dict[str, Subquestion]:
        return {item.question_id: item for item in self.subquestions}


def _extract_entities(text: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for match in (*_QUOTED_RE.findall(text), *_CAPITALIZED_RE.findall(text)):
        candidate = " ".join(match.split())
        key = candidate.casefold()
        if key in {"what", "which", "when", "where", "why", "how", "compare"}:
            continue
        if candidate and key not in seen:
            seen.add(key)
            values.append(candidate)
        if len(values) >= _MAX_ENTITIES:
            break
    return tuple(values)


def _extract_temporal_constraints(text: str) -> tuple[str, ...]:
    result: list[str] = list(dict.fromkeys(_YEAR_RE.findall(text)))
    tokens = {token.lower() for token in _TOKEN_RE.findall(text)}
    result.extend(sorted(tokens & _TEMPORAL_WORDS))
    return tuple(result[:_MAX_CONSTRAINTS])


def _relation(text: str) -> str:
    lowered = text.lower()
    if any(value in lowered for value in ("compare", "contrast", " versus ", " vs ")):
        return "compare"
    if any(value in lowered for value in ("before", "after", "timeline", "trend", "latest")):
        return "temporal"
    if lowered.startswith(("why", "how")) or "explain" in lowered:
        return "explain"
    if any(value in lowered for value in ("combine", "synthesize", "overall", "based on")):
        return "synthesize"
    return "lookup"


def _heuristic_nodes(query: str, maximum: int) -> tuple[Subquestion, ...]:
    comparison = _COMPARISON_RE.search(query) or _VERSUS_RE.search(query)
    if comparison and maximum >= 3:
        left = _bounded_text(comparison.group(1), "comparison entity", 300).strip(" ,")
        right = _bounded_text(comparison.group(2), "comparison entity", 300).strip(" ,")
        first = Subquestion(
            "q1", f"Find evidence about {left} relevant to: {query}",
            entities=_extract_entities(left),
            temporal_constraints=_extract_temporal_constraints(query),
        )
        second = Subquestion(
            "q2", f"Find evidence about {right} relevant to: {query}",
            entities=_extract_entities(right),
            temporal_constraints=_extract_temporal_constraints(query),
        )
        final = Subquestion(
            "q3", query, depends_on=("q1", "q2"),
            entities=tuple(dict.fromkeys((*first.entities, *second.entities))),
            temporal_constraints=_extract_temporal_constraints(query),
            relation="compare",
        )
        return first, second, final

    clauses = [value for value in _SPLIT_RE.split(query) if value]
    if len(clauses) <= 1:
        return (
            Subquestion(
                "q1", query, entities=_extract_entities(query),
                temporal_constraints=_extract_temporal_constraints(query),
                relation=_relation(query),
            ),
        )
    clauses = clauses[: max(1, maximum - 1)]
    nodes = [
        Subquestion(
            f"q{index}", clause, entities=_extract_entities(clause),
            temporal_constraints=_extract_temporal_constraints(clause),
            relation=_relation(clause),
        )
        for index, clause in enumerate(clauses, start=1)
    ]
    if len(nodes) < maximum:
        nodes.append(
            Subquestion(
                f"q{len(nodes) + 1}", query,
                depends_on=tuple(item.question_id for item in nodes),
                entities=_extract_entities(query),
                temporal_constraints=_extract_temporal_constraints(query),
                relation="synthesize",
            )
        )
    return tuple(nodes)


def _node_from_mapping(value: Mapping[str, Any], index: int) -> Subquestion:
    try:
        unknown = set(value) - _NODE_KEYS
        if unknown:
            raise ValueError(f"subquestion contains unknown fields: {sorted(unknown)!r}.")
        return Subquestion(
            value.get("question_id", f"q{index}"),
            value.get("text"),
            depends_on=value.get("depends_on", ()),
            entities=value.get("entities", ()),
            temporal_constraints=value.get("temporal_constraints", ()),
            relation=value.get("relation", "lookup"),
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("subquestion mappings must be safely readable.") from exc


def _validate_and_batch(
    nodes: Sequence[Subquestion],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    by_id: dict[str, Subquestion] = {}
    for node in nodes:
        if node.question_id in by_id:
            raise ValueError("question_id values must be unique.")
        by_id[node.question_id] = node
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {identifier: 0 for identifier in by_id}
    for node in nodes:
        for dependency in node.depends_on:
            if dependency not in by_id:
                raise ValueError("every dependency must reference a declared subquestion.")
            indegree[node.question_id] += 1
            children[dependency].append(node.question_id)
    ready = deque(sorted(identifier for identifier, degree in indegree.items() if degree == 0))
    batches: list[tuple[str, ...]] = []
    visited = 0
    while ready:
        batch = tuple(ready)
        ready.clear()
        batches.append(batch)
        next_ready: list[str] = []
        for identifier in batch:
            visited += 1
            for child in sorted(children.get(identifier, ())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        ready.extend(sorted(next_ready))
    if visited != len(nodes):
        raise ValueError("subquestion dependencies must form an acyclic graph.")
    terminals = tuple(sorted(identifier for identifier in by_id if not children.get(identifier)))
    return tuple(batches), terminals


def build_decomposition_plan(
    query: str,
    *,
    proposed_subquestions: Iterable[Subquestion | Mapping[str, Any]] | None = None,
    max_subquestions: int = 8,
) -> DecompositionPlan:
    cleaned = _bounded_text(query, "query", _MAX_QUERY_CHARS)
    maximum = _integer(max_subquestions, "max_subquestions", 1, _MAX_SUBQUESTIONS)
    if proposed_subquestions is None:
        nodes = _heuristic_nodes(cleaned, maximum)
    else:
        raw = _bounded_items(
            proposed_subquestions, "proposed_subquestions", maximum
        )
        if not raw:
            raise ValueError(
                f"proposed_subquestions must contain 1-{maximum} values."
            )
        converted: list[Subquestion] = []
        for index, value in enumerate(raw, start=1):
            if isinstance(value, Subquestion):
                converted.append(value)
            elif isinstance(value, Mapping):
                converted.append(_node_from_mapping(value, index))
            else:
                raise ValueError(
                    "subquestions must be Subquestion values or mappings."
                )
        nodes = tuple(converted)
    batches, terminals = _validate_and_batch(nodes)
    payload = {
        "query": cleaned,
        "subquestions": [
            {
                "question_id": node.question_id,
                "text": node.text,
                "depends_on": node.depends_on,
                "entities": node.entities,
                "temporal_constraints": node.temporal_constraints,
                "relation": node.relation,
            }
            for node in nodes
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DecompositionPlan(cleaned, nodes, batches, terminals, fingerprint)


__all__ = ["DecompositionPlan", "Subquestion", "build_decomposition_plan"]
