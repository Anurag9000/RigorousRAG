"""Closed-schema authority for model outputs before publication or downstream use.

Model-generated JSON is untrusted data. Parsing rejects duplicate keys, non-standard JSON
numbers, excessive depth/size and fields outside a repository-owned schema. Grounded answer
validation additionally restricts citations to server-owned evidence identities. Standard
model-authored role/tool/function-call fields are reserved and cannot be declared as output
schema fields; tool execution uses the separate trusted planner authorization boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_FIELD_TYPES = frozenset({"string", "integer", "number", "boolean", "string_list"})
_RESERVED_AUTHORITY_FIELDS = frozenset(
    {
        "role",
        "system",
        "developer",
        "tool_call",
        "tool_calls",
        "function_call",
        "function_calls",
    }
)
_HEX = frozenset("0123456789abcdef")
_MAX_OUTPUT_BYTES = 5_000_000
_MAX_DEPTH = 32


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r} is forbidden")


def _depth(value: Any, level: int = 0) -> int:
    if level > _MAX_DEPTH:
        raise ValueError("model output exceeds maximum JSON depth")
    if isinstance(value, Mapping):
        for child in value.values():
            _depth(child, level + 1)
    elif isinstance(value, list):
        for child in value:
            _depth(child, level + 1)
    return level


def parse_strict_model_json(raw: str | bytes) -> Mapping[str, Any]:
    if isinstance(raw, str):
        encoded = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        raise ValueError("model output must be str or bytes")
    if not encoded or len(encoded) > _MAX_OUTPUT_BYTES:
        raise ValueError("model output must be non-empty and bounded")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("model output must be UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("model output is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("model output root must be a JSON object")
    _depth(value)
    return value


@dataclass(frozen=True)
class OutputFieldSpec:
    name: str
    field_type: str
    required: bool = True
    enum_values: tuple[str, ...] = ()
    maximum_length: int | None = None
    maximum_items: int | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, "field name", 200)
        if name in _RESERVED_AUTHORITY_FIELDS:
            raise ValueError(f"field name {name!r} is reserved for server authority")
        object.__setattr__(self, "name", name)
        kind = _text(self.field_type, "field_type", 50).lower()
        if kind not in _FIELD_TYPES:
            raise ValueError(f"field_type must be one of {sorted(_FIELD_TYPES)}")
        object.__setattr__(self, "field_type", kind)
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean")
        enum_values = tuple(sorted({_text(value, "enum value", 1000) for value in self.enum_values}))
        if enum_values and kind not in {"string", "string_list"}:
            raise ValueError("enum_values are supported only for string/string_list fields")
        object.__setattr__(self, "enum_values", enum_values)
        for name_attr in ("maximum_length", "maximum_items"):
            value = getattr(self, name_attr)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name_attr} must be positive when set")


@dataclass(frozen=True)
class ClosedOutputSchema:
    schema_id: str
    schema_version: str
    fields: tuple[OutputFieldSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", _text(self.schema_id, "schema_id", 300))
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version", 100))
        fields = tuple(self.fields)
        if not fields or len(fields) > 1000 or any(not isinstance(value, OutputFieldSpec) for value in fields):
            raise ValueError("fields must be a non-empty bounded OutputFieldSpec sequence")
        if len({value.name for value in fields}) != len(fields):
            raise ValueError("schema field names must be unique")
        object.__setattr__(self, "fields", tuple(sorted(fields, key=lambda value: value.name)))

    @property
    def schema_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-closed-output-schema/v1", **asdict(self)})


def _validate_field(value: Any, spec: OutputFieldSpec) -> Any:
    if spec.field_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"field {spec.name!r} must be string")
        if spec.maximum_length is not None and len(value) > spec.maximum_length:
            raise ValueError(f"field {spec.name!r} exceeds maximum_length")
        if spec.enum_values and value not in spec.enum_values:
            raise ValueError(f"field {spec.name!r} is outside its enum")
        return value
    if spec.field_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"field {spec.name!r} must be integer")
        return value
    if spec.field_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"field {spec.name!r} must be finite number")
        return value
    if spec.field_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"field {spec.name!r} must be boolean")
        return value
    if spec.field_type == "string_list":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"field {spec.name!r} must be a list of strings")
        if spec.maximum_items is not None and len(value) > spec.maximum_items:
            raise ValueError(f"field {spec.name!r} exceeds maximum_items")
        if spec.maximum_length is not None and any(len(item) > spec.maximum_length for item in value):
            raise ValueError(f"field {spec.name!r} contains an oversized item")
        if spec.enum_values and any(item not in spec.enum_values for item in value):
            raise ValueError(f"field {spec.name!r} contains a value outside its enum")
        return tuple(value)
    raise AssertionError("validated field type was lost")


@dataclass(frozen=True)
class ValidatedModelOutput:
    schema_sha256: str
    model_artifact_sha256: str
    context_sha256: str
    raw_output_sha256: str
    canonical_output_json: str
    output_sha256: str

    def __post_init__(self) -> None:
        for name in ("schema_sha256", "model_artifact_sha256", "context_sha256", "raw_output_sha256", "output_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.canonical_output_json, str) or not self.canonical_output_json:
            raise ValueError("canonical_output_json must be non-empty")
        if hashlib.sha256(self.canonical_output_json.encode("utf-8")).hexdigest() != self.output_sha256:
            raise ValueError("canonical output digest mismatch")

    @property
    def value(self) -> Mapping[str, Any]:
        parsed = json.loads(self.canonical_output_json)
        if not isinstance(parsed, Mapping):
            raise RuntimeError("validated output no longer decodes to an object")
        return parsed


def validate_model_output(
    raw_output: str | bytes,
    *,
    schema: ClosedOutputSchema,
    model_artifact_sha256: str,
    context_sha256: str,
) -> ValidatedModelOutput:
    if not isinstance(schema, ClosedOutputSchema):
        raise ValueError("schema must be ClosedOutputSchema")
    parsed = parse_strict_model_json(raw_output)
    allowed = {field.name: field for field in schema.fields}
    extras = set(parsed) - set(allowed)
    if extras:
        raise ValueError(f"model output contains fields outside the closed schema: {sorted(extras)}")
    missing = {field.name for field in schema.fields if field.required and field.name not in parsed}
    if missing:
        raise ValueError(f"model output is missing required fields: {sorted(missing)}")
    normalized: dict[str, Any] = {}
    for name, value in parsed.items():
        normalized[name] = _validate_field(value, allowed[name])
        if isinstance(normalized[name], tuple):
            normalized[name] = list(normalized[name])
    encoded = _canonical(normalized)
    raw_bytes = raw_output.encode("utf-8") if isinstance(raw_output, str) else raw_output
    return ValidatedModelOutput(
        schema_sha256=schema.schema_sha256,
        model_artifact_sha256=_sha(model_artifact_sha256, "model_artifact_sha256"),
        context_sha256=_sha(context_sha256, "context_sha256"),
        raw_output_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        canonical_output_json=encoded.decode("utf-8"),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True)
class GroundedAnswerPolicy:
    answer_field: str = "answer"
    citation_field: str = "citation_ids"
    abstain_field: str = "abstain"
    abstention_reason_field: str = "abstention_reason"
    require_citations_when_answering: bool = True
    maximum_citations: int = 100

    def __post_init__(self) -> None:
        names = (
            _text(self.answer_field, "answer_field", 200),
            _text(self.citation_field, "citation_field", 200),
            _text(self.abstain_field, "abstain_field", 200),
            _text(self.abstention_reason_field, "abstention_reason_field", 200),
        )
        if len(set(names)) != len(names):
            raise ValueError("grounded answer field names must be distinct")
        for field_name in names:
            if field_name in _RESERVED_AUTHORITY_FIELDS:
                raise ValueError("grounded answer field collides with reserved authority field")
        object.__setattr__(self, "answer_field", names[0])
        object.__setattr__(self, "citation_field", names[1])
        object.__setattr__(self, "abstain_field", names[2])
        object.__setattr__(self, "abstention_reason_field", names[3])
        if not isinstance(self.require_citations_when_answering, bool):
            raise ValueError("require_citations_when_answering must be boolean")
        if isinstance(self.maximum_citations, bool) or not isinstance(self.maximum_citations, int) or not 1 <= self.maximum_citations <= 10_000:
            raise ValueError("maximum_citations is invalid")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-grounded-answer-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class GroundedModelOutput:
    validated_output: ValidatedModelOutput
    policy_sha256: str
    answer: str
    citation_ids: tuple[str, ...]
    abstain: bool
    abstention_reason: str | None
    grounded_output_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.validated_output, ValidatedModelOutput):
            raise ValueError("validated_output must be ValidatedModelOutput")
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        if not isinstance(self.answer, str):
            raise ValueError("answer must be string")
        citations = tuple(_text(value, "citation id", 1000) for value in self.citation_ids)
        if len(citations) != len(set(citations)):
            raise ValueError("citation ids must be unique")
        object.__setattr__(self, "citation_ids", citations)
        if not isinstance(self.abstain, bool):
            raise ValueError("abstain must be boolean")
        if self.abstention_reason is not None and not isinstance(self.abstention_reason, str):
            raise ValueError("abstention_reason must be string when present")
        expected = _digest(self._payload())
        provided = _sha(self.grounded_output_sha256, "grounded_output_sha256")
        if expected != provided:
            raise ValueError("grounded output digest mismatch")
        object.__setattr__(self, "grounded_output_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-grounded-model-output/v1",
            "validated_output_sha256": self.validated_output.output_sha256,
            "policy_sha256": self.policy_sha256,
            "answer_sha256": hashlib.sha256(self.answer.encode("utf-8")).hexdigest(),
            "citation_ids": self.citation_ids,
            "abstain": self.abstain,
            "abstention_reason_sha256": None if self.abstention_reason is None else hashlib.sha256(self.abstention_reason.encode("utf-8")).hexdigest(),
        }


def validate_grounded_model_output(
    validated: ValidatedModelOutput,
    *,
    policy: GroundedAnswerPolicy,
    allowed_citation_ids: Sequence[str],
) -> GroundedModelOutput:
    if not isinstance(validated, ValidatedModelOutput):
        raise ValueError("validated must be ValidatedModelOutput")
    if not isinstance(policy, GroundedAnswerPolicy):
        raise ValueError("policy must be GroundedAnswerPolicy")
    value = validated.value
    required = {policy.answer_field, policy.citation_field, policy.abstain_field}
    if not required.issubset(value):
        raise ValueError("validated output lacks grounded answer fields")
    answer = value[policy.answer_field]
    citations = value[policy.citation_field]
    abstain = value[policy.abstain_field]
    reason = value.get(policy.abstention_reason_field)
    if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(item, str) for item in citations) or not isinstance(abstain, bool):
        raise ValueError("grounded answer fields have invalid types")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("abstention reason has invalid type")
    if len(citations) > policy.maximum_citations or len(citations) != len(set(citations)):
        raise ValueError("citation list is duplicated or exceeds maximum_citations")
    allowed = {_text(value, "allowed citation id", 1000) for value in allowed_citation_ids}
    if any(citation not in allowed for citation in citations):
        raise ValueError("model output cites evidence outside the server-owned allowed set")
    if abstain:
        if citations:
            raise ValueError("abstaining output may not publish citations")
        if reason is None or not reason.strip():
            raise ValueError("abstaining output requires an abstention reason")
    elif policy.require_citations_when_answering and answer.strip() and not citations:
        raise ValueError("non-abstaining answer requires at least one server-owned citation")
    payload = {
        "schema": "rigorousrag-grounded-model-output/v1",
        "validated_output_sha256": validated.output_sha256,
        "policy_sha256": policy.policy_sha256,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "citation_ids": tuple(citations),
        "abstain": abstain,
        "abstention_reason_sha256": None if reason is None else hashlib.sha256(reason.encode("utf-8")).hexdigest(),
    }
    return GroundedModelOutput(validated, policy.policy_sha256, answer, tuple(citations), abstain, reason, _digest(payload))


__all__ = [
    "ClosedOutputSchema",
    "GroundedAnswerPolicy",
    "GroundedModelOutput",
    "OutputFieldSpec",
    "ValidatedModelOutput",
    "parse_strict_model_json",
    "validate_grounded_model_output",
    "validate_model_output",
]
