"""PII redaction, ACL enforcement, revocation propagation, and audit events."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple


_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"
)


def _luhn_valid(raw: str) -> bool:
    digits = [int(ch) for ch in raw if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class Redaction:
    kind: str
    original_hash: str
    replacement: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redactions: Tuple[Redaction, ...]


def redact_pii(text: str) -> RedactionResult:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    redactions: List[Redaction] = []

    def replace(kind: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        replacement = f"[REDACTED_{kind}]"
        redactions.append(Redaction(kind, digest, replacement))
        return replacement

    output = _EMAIL.sub(lambda match: replace("EMAIL", match.group(0)), text)
    output = _PHONE.sub(lambda match: replace("PHONE", match.group(0)), output)

    def replace_ipv4(match: re.Match[str]) -> str:
        value = match.group(0)
        parts = value.split(".")
        if all(0 <= int(part) <= 255 for part in parts):
            return replace("IP", value)
        return value

    output = _IPV4.sub(replace_ipv4, output)

    def replace_card(match: re.Match[str]) -> str:
        value = match.group(0)
        return replace("CARD", value) if _luhn_valid(value) else value

    output = _CARD.sub(replace_card, output)

    def replace_secret(match: re.Match[str]) -> str:
        whole = match.group(0)
        secret = match.group(1)
        redacted = replace("SECRET", secret)
        return whole.replace(secret, redacted)

    output = _SECRET.sub(replace_secret, output)
    return RedactionResult(output, tuple(redactions))


@dataclass(frozen=True)
class AccessContext:
    principal: str
    groups: frozenset[str] = frozenset()
    labels: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ACL:
    owners: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()
    required_labels: frozenset[str] = frozenset()
    public: bool = False

    def allows(self, context: AccessContext) -> bool:
        if self.public:
            return self.required_labels.issubset(context.labels)
        principal_allowed = context.principal in self.owners
        group_allowed = bool(self.groups & context.groups)
        return (principal_allowed or group_allowed) and self.required_labels.issubset(
            context.labels
        )


def filter_authorized(
    items: Iterable[object],
    context: AccessContext,
    *,
    acl_getter=lambda item: getattr(item, "acl", ACL(public=True)),
) -> List[object]:
    output = []
    for item in items:
        acl = acl_getter(item)
        if not isinstance(acl, ACL):
            raise TypeError("acl_getter must return ACL instances.")
        if acl.allows(context):
            output.append(item)
    return output


@dataclass(frozen=True)
class Revocation:
    artifact_id: str
    reason: str
    revoked_at: float


class LineageRegistry:
    """Tracks source->derived artifacts and propagates revocation transitively."""

    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = {}
        self._parents: Dict[str, Set[str]] = {}
        self._revoked: Dict[str, Revocation] = {}

    def link(self, parent_id: str, child_id: str) -> None:
        parent_id = str(parent_id).strip()
        child_id = str(child_id).strip()
        if not parent_id or not child_id:
            raise ValueError("artifact identifiers must be non-empty.")
        if parent_id == child_id:
            raise ValueError("an artifact cannot derive from itself.")
        self._children.setdefault(parent_id, set()).add(child_id)
        self._parents.setdefault(child_id, set()).add(parent_id)

    def descendants(self, artifact_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self._children.get(artifact_id, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._children.get(current, ()))
        return seen

    def revoke(
        self,
        artifact_id: str,
        *,
        reason: str,
        revoked_at: Optional[float] = None,
    ) -> Set[str]:
        when = float(time.time() if revoked_at is None else revoked_at)
        affected = {artifact_id, *self.descendants(artifact_id)}
        for current in affected:
            self._revoked[current] = Revocation(current, str(reason), when)
        return affected

    def is_revoked(self, artifact_id: str) -> bool:
        return artifact_id in self._revoked

    def revocation(self, artifact_id: str) -> Optional[Revocation]:
        return self._revoked.get(artifact_id)


@dataclass(frozen=True)
class SecurityAuditEvent:
    event_type: str
    principal: str
    resource: str
    allowed: bool
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None
    details: Mapping[str, str] = field(default_factory=dict)


class SecurityAuditLog:
    def __init__(self) -> None:
        self._events: List[SecurityAuditEvent] = []

    def append(self, event: SecurityAuditEvent) -> None:
        self._events.append(event)

    def events(self) -> Tuple[SecurityAuditEvent, ...]:
        return tuple(self._events)

    def write_jsonl(self, path: str | Path) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return len(self._events)
