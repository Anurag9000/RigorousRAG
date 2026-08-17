"""Default-deny field-level disclosure policies for shared and federated research.

Policies operate on explicit dotted field paths. They never mutate source objects and they
return both the filtered projection and a deterministic decision trace so downstream
artifacts can bind exactly which disclosure policy produced them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

_MAX_RULES = 2000
_MAX_DEPTH = 32
_MAX_ITEMS = 100_000


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _path(value: str) -> str:
    selected = _text(value, "field_path", 1000)
    if selected.startswith(".") or selected.endswith(".") or ".." in selected:
        raise ValueError("field_path is invalid")
    for part in selected.split("."):
        if part != "*" and (not part or any(ch in part for ch in "[]{}")):
            raise ValueError("field_path contains an invalid segment")
    return selected


def _segments(value: str) -> tuple[str, ...]:
    return tuple(value.split(".")) if value else ()


def _matches(pattern: str, path: str) -> bool:
    expected = _segments(pattern)
    actual = _segments(path)
    if len(expected) != len(actual):
        return False
    return all(left == "*" or left == right for left, right in zip(expected, actual))


def _prefix_matches(pattern: str, path: str) -> bool:
    """Return whether ``path`` can be a parent of some value matched by ``pattern``."""
    expected = _segments(pattern)
    actual = _segments(path)
    if len(expected) <= len(actual):
        return False
    return all(left == "*" or left == right for left, right in zip(expected[: len(actual)], actual))


@dataclass(frozen=True)
class DisclosureRule:
    field_path: str
    effect: str
    roles: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_path", _path(self.field_path))
        effect = _text(self.effect, "effect", 16).lower()
        if effect not in {"allow", "deny"}:
            raise ValueError("effect must be allow or deny")
        object.__setattr__(self, "effect", effect)
        roles = tuple(sorted(set(_text(item, "role", 128).lower() for item in self.roles)))
        purposes = tuple(sorted(set(_text(item, "purpose", 128).lower() for item in self.purposes)))
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "purposes", purposes)
        object.__setattr__(self, "reason", _text(self.reason, "reason", 2000, allow_empty=True))

    def applies(self, *, path: str, role: str, purpose: str) -> bool:
        if not _matches(self.field_path, path):
            return False
        if self.roles and role not in self.roles:
            return False
        if self.purposes and purpose not in self.purposes:
            return False
        return True

    def may_match_descendant(self, *, path: str, role: str, purpose: str) -> bool:
        if not _prefix_matches(self.field_path, path):
            return False
        if self.roles and role not in self.roles:
            return False
        if self.purposes and purpose not in self.purposes:
            return False
        return True


@dataclass(frozen=True)
class DisclosurePolicy:
    policy_id: str
    version: str
    rules: tuple[DisclosureRule, ...]
    default_effect: str = "deny"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 256))
        object.__setattr__(self, "version", _text(self.version, "version", 64))
        if len(self.rules) > _MAX_RULES or any(not isinstance(item, DisclosureRule) for item in self.rules):
            raise ValueError("rules are invalid")
        default = _text(self.default_effect, "default_effect", 16).lower()
        if default not in {"allow", "deny"}:
            raise ValueError("default_effect must be allow or deny")
        object.__setattr__(self, "default_effect", default)

    @property
    def fingerprint(self) -> str:
        payload = {
            "policy_id": self.policy_id,
            "version": self.version,
            "rules": [asdict(item) for item in self.rules],
            "default_effect": self.default_effect,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def decide(self, *, path: str, role: str, purpose: str) -> tuple[str, str]:
        selected_path = _path(path)
        selected_role = _text(role, "role", 128).lower()
        selected_purpose = _text(purpose, "purpose", 128).lower()
        matched = [item for item in self.rules if item.applies(path=selected_path, role=selected_role, purpose=selected_purpose)]
        if not matched:
            return self.default_effect, "default"
        # Deny wins at equal path specificity. More-specific paths win over wildcard-heavy paths.
        matched.sort(
            key=lambda item: (
                sum(1 for segment in _segments(item.field_path) if segment != "*"),
                1 if item.effect == "deny" else 0,
            ),
            reverse=True,
        )
        winner = matched[0]
        return winner.effect, winner.reason or f"rule:{winner.field_path}"

    def has_allowed_descendant(self, *, path: str, role: str, purpose: str) -> bool:
        selected_path = _path(path)
        selected_role = _text(role, "role", 128).lower()
        selected_purpose = _text(purpose, "purpose", 128).lower()
        return any(
            item.effect == "allow"
            and item.may_match_descendant(path=selected_path, role=selected_role, purpose=selected_purpose)
            for item in self.rules
        )


@dataclass(frozen=True)
class DisclosureTraceEntry:
    field_path: str
    effect: str
    reason: str


@dataclass(frozen=True)
class DisclosureProjection:
    value: Any
    policy_fingerprint: str
    role: str
    purpose: str
    trace: tuple[DisclosureTraceEntry, ...]
    allowed_fields: int
    denied_fields: int

    @property
    def fingerprint(self) -> str:
        payload = {
            "value": self.value,
            "policy_fingerprint": self.policy_fingerprint,
            "role": self.role,
            "purpose": self.purpose,
            "trace": [asdict(item) for item in self.trace],
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


def project_disclosure(
    value: Mapping[str, Any],
    policy: DisclosurePolicy,
    *,
    role: str,
    purpose: str,
) -> DisclosureProjection:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    if not isinstance(policy, DisclosurePolicy):
        raise TypeError("policy must be DisclosurePolicy")
    selected_role = _text(role, "role", 128).lower()
    selected_purpose = _text(purpose, "purpose", 128).lower()
    trace: list[DisclosureTraceEntry] = []
    counter = 0

    def walk(current: Any, prefix: str, depth: int) -> tuple[Any, bool]:
        nonlocal counter
        if depth > _MAX_DEPTH:
            raise ValueError("disclosure payload exceeds maximum depth")
        if isinstance(current, Mapping):
            output: dict[str, Any] = {}
            for raw_key, child in current.items():
                counter += 1
                if counter > _MAX_ITEMS:
                    raise ValueError("disclosure payload exceeds item limit")
                key = _text(str(raw_key), "mapping key", 256)
                path = f"{prefix}.{key}" if prefix else key
                effect, reason = policy.decide(path=path, role=selected_role, purpose=selected_purpose)
                trace.append(DisclosureTraceEntry(path, effect, reason))
                if effect == "allow":
                    projected, _ = walk(child, path, depth + 1)
                    output[key] = projected
                    continue
                # A default-denied container may still contain a specifically allowed child.
                # An explicit deny rule is a hard subtree stop and is never bypassed.
                if reason == "default" and isinstance(child, (Mapping, list, tuple)) and policy.has_allowed_descendant(
                    path=path,
                    role=selected_role,
                    purpose=selected_purpose,
                ):
                    projected, has_value = walk(child, path, depth + 1)
                    if has_value:
                        output[key] = projected
            return output, bool(output)
        if isinstance(current, (list, tuple)):
            output_items: list[Any] = []
            for index, child in enumerate(current):
                counter += 1
                if counter > _MAX_ITEMS:
                    raise ValueError("disclosure payload exceeds item limit")
                path = f"{prefix}.{index}" if prefix else str(index)
                effect, reason = policy.decide(path=path, role=selected_role, purpose=selected_purpose)
                trace.append(DisclosureTraceEntry(path, effect, reason))
                if effect == "allow":
                    projected, _ = walk(child, path, depth + 1)
                    output_items.append(projected)
                    continue
                if reason == "default" and isinstance(child, (Mapping, list, tuple)) and policy.has_allowed_descendant(
                    path=path,
                    role=selected_role,
                    purpose=selected_purpose,
                ):
                    projected, has_value = walk(child, path, depth + 1)
                    if has_value:
                        output_items.append(projected)
            return output_items, bool(output_items)
        return current, True

    projected, _ = walk(dict(value), "", 0)
    allowed = sum(1 for item in trace if item.effect == "allow")
    denied = len(trace) - allowed
    return DisclosureProjection(
        value=projected,
        policy_fingerprint=policy.fingerprint,
        role=selected_role,
        purpose=selected_purpose,
        trace=tuple(trace),
        allowed_fields=allowed,
        denied_fields=denied,
    )


__all__ = [
    "DisclosurePolicy",
    "DisclosureProjection",
    "DisclosureRule",
    "DisclosureTraceEntry",
    "project_disclosure",
]
