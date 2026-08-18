"""Destination-aware sensitive-data scanning and governed text release.

Native deterministic detectors are a minimum safety net, not a complete PII oracle.
Deployments may add validated scans from external DLP/NER providers and require those
provider attestations in policy. Native findings always participate and cannot be
suppressed by caller-provided scans. Release decisions bind digests and span metadata;
raw sensitive substrings are not stored in receipts.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_CATEGORIES = frozenset(
    {
        "email",
        "phone",
        "address",
        "person_name",
        "government_id",
        "payment_card",
        "financial",
        "health",
        "secret",
        "ip_address",
        "custom_sensitive",
    }
)
_DESTINATIONS = frozenset({"model_input", "external_provider", "audit_export", "benchmark_export", "observability"})
_ACTIONS = frozenset({"allow", "redact", "block"})
_HEX = frozenset("0123456789abcdef")
_MAX_TEXT_BYTES = 20_000_000
_MAX_FINDINGS = 100_000

_EMAIL = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63})(?![\w.-])", re.I)
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer)\b\s*[:=]\s*['\"]?([A-Za-z0-9_./+\-=]{8,})"
)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def _category(value: Any) -> str:
    selected = _text(value, "category", 100).lower()
    if selected not in _CATEGORIES:
        raise ValueError(f"unsupported sensitive-data category {selected!r}")
    return selected


def _destination(value: Any) -> str:
    selected = _text(value, "destination", 100).lower()
    if selected not in _DESTINATIONS:
        raise ValueError(f"unsupported release destination {selected!r}")
    return selected


def _content_sha(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > _MAX_TEXT_BYTES:
        raise ValueError("text must be non-empty and bounded")
    return hashlib.sha256(encoded).hexdigest()


def _luhn(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0


@dataclass(frozen=True)
class SensitiveSpan:
    start: int
    end: int
    category: str
    confidence: float
    detector_id: str
    input_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool) or not isinstance(self.start, int) or not isinstance(self.end, int) or self.start < 0 or self.end <= self.start:
            raise ValueError("sensitive span offsets are invalid")
        object.__setattr__(self, "category", _category(self.category))
        if isinstance(self.confidence, bool):
            raise ValueError("confidence must be in [0,1]")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "detector_id", _text(self.detector_id, "detector_id", 300))
        object.__setattr__(self, "input_sha256", _sha(self.input_sha256, "input_sha256"))

    @property
    def finding_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-sensitive-span/v1", **asdict(self)})


@dataclass(frozen=True)
class SensitiveDataScan:
    detector_id: str
    detector_version_sha256: str
    input_sha256: str
    findings: tuple[SensitiveSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "detector_id", _text(self.detector_id, "detector_id", 300))
        object.__setattr__(self, "detector_version_sha256", _sha(self.detector_version_sha256, "detector_version_sha256"))
        object.__setattr__(self, "input_sha256", _sha(self.input_sha256, "input_sha256"))
        findings = tuple(self.findings)
        if len(findings) > _MAX_FINDINGS or any(not isinstance(value, SensitiveSpan) for value in findings):
            raise ValueError("findings must be a bounded SensitiveSpan sequence")
        if any(value.detector_id != self.detector_id or value.input_sha256 != self.input_sha256 for value in findings):
            raise ValueError("finding detector/input identity differs from scan")
        if len({value.finding_sha256 for value in findings}) != len(findings):
            raise ValueError("scan contains duplicate finding identities")
        object.__setattr__(self, "findings", tuple(sorted(findings, key=lambda value: (value.start, value.end, value.category, value.finding_sha256))))

    @property
    def scan_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-sensitive-data-scan/v1",
                "detector_id": self.detector_id,
                "detector_version_sha256": self.detector_version_sha256,
                "input_sha256": self.input_sha256,
                "finding_sha256s": [value.finding_sha256 for value in self.findings],
            }
        )


def run_native_sensitive_scan(text: str) -> SensitiveDataScan:
    input_sha = _content_sha(text)
    findings: list[SensitiveSpan] = []

    def add(start: int, end: int, category: str, confidence: float) -> None:
        findings.append(SensitiveSpan(start, end, category, confidence, "rigorousrag-native-dlp", input_sha))

    for match in _EMAIL.finditer(text):
        add(match.start(1), match.end(1), "email", 0.99)
    for match in _IPV4.finditer(text):
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        add(match.start(), match.end(), "ip_address", 0.95)
    for match in _SECRET.finditer(text):
        # Redact only the value, not the key name, preserving enough context for debugging.
        add(match.start(2), match.end(2), "secret", 0.95)
    for match in _CARD.finditer(text):
        if _luhn(match.group(0)):
            add(match.start(), match.end(), "payment_card", 0.99)
    return SensitiveDataScan(
        detector_id="rigorousrag-native-dlp",
        detector_version_sha256=_digest({"schema": "rigorousrag-native-dlp/v1", "patterns": ("email", "ipv4", "secret", "luhn_card")}),
        input_sha256=input_sha,
        findings=tuple(findings),
    )


@dataclass(frozen=True)
class SensitiveCategoryRule:
    category: str
    action: str
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _category(self.category))
        action = _text(self.action, "action", 30).lower()
        if action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        object.__setattr__(self, "action", action)
        if isinstance(self.minimum_confidence, bool):
            raise ValueError("minimum_confidence must be in [0,1]")
        confidence = float(self.minimum_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0,1]")
        object.__setattr__(self, "minimum_confidence", confidence)


@dataclass(frozen=True)
class DataReleasePolicy:
    policy_id: str
    destination: str
    rules: tuple[SensitiveCategoryRule, ...]
    required_detector_ids: tuple[str, ...] = ()
    default_action: str = "redact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        object.__setattr__(self, "destination", _destination(self.destination))
        rules = tuple(self.rules)
        if any(not isinstance(value, SensitiveCategoryRule) for value in rules):
            raise ValueError("rules contains invalid values")
        if len({value.category for value in rules}) != len(rules):
            raise ValueError("policy may contain at most one rule per category")
        object.__setattr__(self, "rules", tuple(sorted(rules, key=lambda value: value.category)))
        required = tuple(sorted({_text(value, "required detector id", 300) for value in self.required_detector_ids}))
        object.__setattr__(self, "required_detector_ids", required)
        default = _text(self.default_action, "default_action", 30).lower()
        if default not in _ACTIONS:
            raise ValueError(f"default_action must be one of {sorted(_ACTIONS)}")
        object.__setattr__(self, "default_action", default)

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-data-release-policy/v1", **asdict(self)})

    def rule_for(self, category: str) -> SensitiveCategoryRule | None:
        selected = _category(category)
        return next((value for value in self.rules if value.category == selected), None)


@dataclass(frozen=True)
class DataReleaseDecision:
    input_sha256: str
    destination: str
    policy_sha256: str
    scan_sha256s: tuple[str, ...]
    finding_sha256s: tuple[str, ...]
    action: str
    output_sha256: str | None
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_sha256", _sha(self.input_sha256, "input_sha256"))
        object.__setattr__(self, "destination", _destination(self.destination))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        object.__setattr__(self, "scan_sha256s", tuple(sorted({_sha(value, "scan sha256") for value in self.scan_sha256s})))
        object.__setattr__(self, "finding_sha256s", tuple(sorted({_sha(value, "finding sha256") for value in self.finding_sha256s})))
        action = _text(self.action, "action", 30).lower()
        if action not in _ACTIONS:
            raise ValueError("invalid release action")
        object.__setattr__(self, "action", action)
        if self.output_sha256 is not None:
            object.__setattr__(self, "output_sha256", _sha(self.output_sha256, "output_sha256"))
        if action == "block" and self.output_sha256 is not None:
            raise ValueError("blocked release may not have an output digest")
        if action != "block" and self.output_sha256 is None:
            raise ValueError("non-blocked release requires an output digest")
        reasons = tuple(sorted({_text(value, "reason code", 200) for value in self.reason_codes}))
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match data release decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-data-release-decision/v1",
            "input_sha256": self.input_sha256,
            "destination": self.destination,
            "policy_sha256": self.policy_sha256,
            "scan_sha256s": self.scan_sha256s,
            "finding_sha256s": self.finding_sha256s,
            "action": self.action,
            "output_sha256": self.output_sha256,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class ReleasedText:
    text: str
    decision: DataReleaseDecision

    def __post_init__(self) -> None:
        if not isinstance(self.decision, DataReleaseDecision) or self.decision.action == "block":
            raise ValueError("released text requires a non-blocking release decision")
        if _content_sha(self.text) != self.decision.output_sha256:
            raise ValueError("released text does not match release decision output digest")


def _merged_redaction_segments(findings: Sequence[SensitiveSpan]) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    if not findings:
        return ()
    events = sorted((value.start, value.end, value.category) for value in findings)
    merged: list[tuple[int, int, set[str]]] = []
    for start, end, category in events:
        if not merged or start >= merged[-1][1]:
            merged.append((start, end, {category}))
            continue
        prior_start, prior_end, categories = merged[-1]
        categories.add(category)
        merged[-1] = (prior_start, max(prior_end, end), categories)
    return tuple((start, end, tuple(sorted(categories))) for start, end, categories in merged)


def release_text(
    text: str,
    *,
    policy: DataReleasePolicy,
    external_scans: Sequence[SensitiveDataScan] = (),
) -> tuple[DataReleaseDecision, ReleasedText | None]:
    if not isinstance(policy, DataReleasePolicy):
        raise ValueError("policy must be DataReleasePolicy")
    input_sha = _content_sha(text)
    native = run_native_sensitive_scan(text)
    extras = tuple(external_scans)
    if any(not isinstance(value, SensitiveDataScan) for value in extras):
        raise ValueError("external_scans contains invalid values")
    scans_by_id: dict[str, SensitiveDataScan] = {native.detector_id: native}
    for scan in extras:
        if scan.input_sha256 != input_sha:
            raise ValueError("external scan is bound to different input content")
        existing = scans_by_id.get(scan.detector_id)
        if existing is not None and existing.scan_sha256 != scan.scan_sha256:
            raise ValueError("conflicting scan identity for detector")
        scans_by_id[scan.detector_id] = scan
    scans = tuple(sorted(scans_by_id.values(), key=lambda value: value.detector_id))
    missing_detectors = sorted(set(policy.required_detector_ids) - set(scans_by_id))
    all_findings = tuple(value for scan in scans for value in scan.findings)
    if len(all_findings) > _MAX_FINDINGS:
        raise ValueError("combined sensitive-data findings exceed limit")
    if any(value.end > len(text) for value in all_findings):
        raise ValueError("sensitive-data finding exceeds input text bounds")

    actionable: list[tuple[SensitiveSpan, str]] = []
    for finding in all_findings:
        rule = policy.rule_for(finding.category)
        action = policy.default_action if rule is None else rule.action
        minimum = 0.0 if rule is None else rule.minimum_confidence
        if finding.confidence >= minimum:
            actionable.append((finding, action))
    reasons: list[str] = []
    if missing_detectors:
        reasons.append("required_detector_attestation_missing")
    if any(action == "block" for _, action in actionable):
        reasons.append("blocking_sensitive_category_detected")
    blocked = bool(missing_detectors) or any(action == "block" for _, action in actionable)
    if blocked:
        action = "block"
        output = None
        output_sha = None
    else:
        redact_findings = [finding for finding, selected_action in actionable if selected_action == "redact"]
        if redact_findings:
            action = "redact"
            reasons.append("sensitive_spans_redacted")
            output = text
            for start, end, categories in reversed(_merged_redaction_segments(redact_findings)):
                output = output[:start] + f"[REDACTED:{'+'.join(categories)}]" + output[end:]
        else:
            action = "allow"
            output = text
        output_sha = _content_sha(output)
    payload = {
        "schema": "rigorousrag-data-release-decision/v1",
        "input_sha256": input_sha,
        "destination": policy.destination,
        "policy_sha256": policy.policy_sha256,
        "scan_sha256s": tuple(sorted(value.scan_sha256 for value in scans)),
        "finding_sha256s": tuple(sorted(value.finding_sha256 for value in all_findings)),
        "action": action,
        "output_sha256": output_sha,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    decision = DataReleaseDecision(**payload, decision_sha256=_digest(payload))
    return decision, None if output is None else ReleasedText(output, decision)


__all__ = [
    "DataReleaseDecision",
    "DataReleasePolicy",
    "ReleasedText",
    "SensitiveCategoryRule",
    "SensitiveDataScan",
    "SensitiveSpan",
    "release_text",
    "run_native_sensitive_scan",
]
