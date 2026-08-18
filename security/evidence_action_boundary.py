"""Fail-closed authority boundary between retrieved evidence and tool execution.

Retrieved evidence can mention or suggest actions, but those suggestions are never
executable authority. A tool request becomes authorized only when an independent trusted
planner selects the same tool and arguments under a governed allow-list/schema contract.
This module deliberately does not execute tools.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_HEX = frozenset("0123456789abcdef")
_MAX_ARGUMENT_BYTES = 1_000_000


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("value must be canonical JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
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


def arguments_sha256(arguments: Mapping[str, Any]) -> str:
    if not isinstance(arguments, Mapping):
        raise ValueError("arguments must be a mapping")
    encoded = _canonical(dict(arguments))
    if len(encoded) > _MAX_ARGUMENT_BYTES:
        raise ValueError("arguments exceed the bounded JSON size")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidenceActionSuggestion:
    evidence_identity_sha256: str
    tool_id: str
    arguments_sha256: str
    suggestion_type: str = "mentioned_action"

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_identity_sha256", _sha(self.evidence_identity_sha256, "evidence_identity_sha256"))
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id", 300))
        object.__setattr__(self, "arguments_sha256", _sha(self.arguments_sha256, "arguments_sha256"))
        object.__setattr__(self, "suggestion_type", _text(self.suggestion_type, "suggestion_type", 100))

    @property
    def suggestion_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-evidence-action-suggestion/v1", **asdict(self)})


@dataclass(frozen=True)
class ToolContract:
    tool_id: str
    schema_sha256: str
    requires_user_confirmation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id", 300))
        object.__setattr__(self, "schema_sha256", _sha(self.schema_sha256, "schema_sha256"))
        if not isinstance(self.requires_user_confirmation, bool):
            raise ValueError("requires_user_confirmation must be boolean")


@dataclass(frozen=True)
class ToolExecutionPolicy:
    policy_id: str
    contracts: tuple[ToolContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        contracts = tuple(self.contracts)
        if not contracts or any(not isinstance(value, ToolContract) for value in contracts):
            raise ValueError("contracts must be a non-empty ToolContract sequence")
        by_id = {value.tool_id: value for value in contracts}
        if len(by_id) != len(contracts):
            raise ValueError("tool contracts must have unique tool ids")
        object.__setattr__(self, "contracts", tuple(sorted(contracts, key=lambda value: value.tool_id)))

    @property
    def policy_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-tool-execution-policy/v1",
                "policy_id": self.policy_id,
                "contracts": [asdict(value) for value in self.contracts],
            }
        )

    def contract_for(self, tool_id: str) -> ToolContract | None:
        selected = _text(tool_id, "tool_id", 300)
        return next((value for value in self.contracts if value.tool_id == selected), None)


@dataclass(frozen=True)
class TrustedPlannerToolDecision:
    planner_id: str
    planner_revision_sha256: str
    user_intent_sha256: str
    tool_id: str
    tool_schema_sha256: str
    arguments_sha256: str
    approved: bool
    reason_code: str
    user_confirmation_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "planner_id", _text(self.planner_id, "planner_id", 300))
        for name in ("planner_revision_sha256", "user_intent_sha256", "tool_schema_sha256", "arguments_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id", 300))
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be boolean")
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code", 200))
        if self.user_confirmation_sha256 is not None:
            object.__setattr__(self, "user_confirmation_sha256", _sha(self.user_confirmation_sha256, "user_confirmation_sha256"))

    @property
    def decision_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-trusted-planner-tool-decision/v1", **asdict(self)})


@dataclass(frozen=True)
class AuthorizedToolRequest:
    tool_id: str
    tool_schema_sha256: str
    arguments_sha256: str
    policy_sha256: str
    planner_decision_sha256: str
    evidence_suggestion_sha256s: tuple[str, ...]
    authorization_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_id", _text(self.tool_id, "tool_id", 300))
        for name in ("tool_schema_sha256", "arguments_sha256", "policy_sha256", "planner_decision_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        suggestions = tuple(sorted({_sha(value, "evidence suggestion sha256") for value in self.evidence_suggestion_sha256s}))
        object.__setattr__(self, "evidence_suggestion_sha256s", suggestions)
        expected = _digest(self._payload())
        provided = _sha(self.authorization_sha256, "authorization_sha256")
        if expected != provided:
            raise ValueError("authorization_sha256 does not match authorized request")
        object.__setattr__(self, "authorization_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-authorized-tool-request/v1",
            "tool_id": self.tool_id,
            "tool_schema_sha256": self.tool_schema_sha256,
            "arguments_sha256": self.arguments_sha256,
            "policy_sha256": self.policy_sha256,
            "planner_decision_sha256": self.planner_decision_sha256,
            "evidence_suggestion_sha256s": self.evidence_suggestion_sha256s,
        }


def authorize_tool_request(
    *,
    policy: ToolExecutionPolicy,
    planner_decision: TrustedPlannerToolDecision,
    arguments: Mapping[str, Any],
    evidence_suggestions: Sequence[EvidenceActionSuggestion] = (),
) -> AuthorizedToolRequest:
    """Authorize only the trusted planner's exact tool/schema/argument decision.

    Evidence suggestions are recorded for provenance but are never sufficient for
    authorization and are not required to agree with the planner. This prevents a caller
    from accidentally treating a retrieved action suggestion as an approval source.
    """

    if not isinstance(policy, ToolExecutionPolicy):
        raise ValueError("policy must be ToolExecutionPolicy")
    if not isinstance(planner_decision, TrustedPlannerToolDecision):
        raise ValueError("planner_decision must be TrustedPlannerToolDecision")
    if not planner_decision.approved:
        raise ValueError("trusted planner did not approve tool execution")
    contract = policy.contract_for(planner_decision.tool_id)
    if contract is None:
        raise ValueError("trusted planner selected a tool outside the allow-list")
    if contract.schema_sha256 != planner_decision.tool_schema_sha256:
        raise ValueError("trusted planner tool schema differs from the governed contract")
    actual_arguments_sha = arguments_sha256(arguments)
    if actual_arguments_sha != planner_decision.arguments_sha256:
        raise ValueError("runtime arguments differ from the trusted planner decision")
    if contract.requires_user_confirmation and planner_decision.user_confirmation_sha256 is None:
        raise ValueError("tool contract requires explicit user confirmation evidence")

    suggestions = tuple(evidence_suggestions)
    if any(not isinstance(value, EvidenceActionSuggestion) for value in suggestions):
        raise ValueError("evidence_suggestions contains invalid values")
    if len({value.suggestion_sha256 for value in suggestions}) != len(suggestions):
        raise ValueError("evidence_suggestions contains duplicate identities")
    payload = {
        "schema": "rigorousrag-authorized-tool-request/v1",
        "tool_id": planner_decision.tool_id,
        "tool_schema_sha256": contract.schema_sha256,
        "arguments_sha256": actual_arguments_sha,
        "policy_sha256": policy.policy_sha256,
        "planner_decision_sha256": planner_decision.decision_sha256,
        "evidence_suggestion_sha256s": tuple(sorted(value.suggestion_sha256 for value in suggestions)),
    }
    return AuthorizedToolRequest(**payload, authorization_sha256=_digest(payload))


def assert_authorized_tool_request(
    request: AuthorizedToolRequest,
    *,
    policy: ToolExecutionPolicy,
    arguments: Mapping[str, Any],
) -> AuthorizedToolRequest:
    """Serving-side guard before an external executor consumes a request."""

    if not isinstance(request, AuthorizedToolRequest):
        raise ValueError("request must be AuthorizedToolRequest")
    if not isinstance(policy, ToolExecutionPolicy):
        raise ValueError("policy must be ToolExecutionPolicy")
    if request.policy_sha256 != policy.policy_sha256:
        raise RuntimeError("tool request was authorized under a stale policy")
    contract = policy.contract_for(request.tool_id)
    if contract is None or contract.schema_sha256 != request.tool_schema_sha256:
        raise RuntimeError("tool request contract is no longer authoritative")
    if arguments_sha256(arguments) != request.arguments_sha256:
        raise RuntimeError("tool request arguments changed after authorization")
    return request


__all__ = [
    "AuthorizedToolRequest",
    "EvidenceActionSuggestion",
    "ToolContract",
    "ToolExecutionPolicy",
    "TrustedPlannerToolDecision",
    "arguments_sha256",
    "assert_authorized_tool_request",
    "authorize_tool_request",
]
