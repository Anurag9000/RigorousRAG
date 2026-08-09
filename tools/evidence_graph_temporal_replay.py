"""Deterministic replay of explicit temporal evidence states across as-of points."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from tools.evidence_graph_temporal import temporal_evidence_status
from tools.evidence_graph_types import EvidenceNode

_MAX_NODES = 100_000
_MAX_POINTS = 10_000


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TemporalReplayFrame:
    as_of: float
    active_node_ids: tuple[str, ...]
    not_yet_valid_node_ids: tuple[str, ...]
    expired_node_ids: tuple[str, ...]
    retracted_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _timestamp(self.as_of, "as_of"))
        groups = (
            self.active_node_ids,
            self.not_yet_valid_node_ids,
            self.expired_node_ids,
            self.retracted_node_ids,
        )
        seen: set[str] = set()
        for group in groups:
            if not isinstance(group, tuple):
                raise ValueError("replay node groups must be tuples.")
            if tuple(sorted(group)) != group or len(set(group)) != len(group):
                raise ValueError("replay node groups must be sorted and unique.")
            if seen.intersection(group):
                raise ValueError("replay node groups must be disjoint.")
            seen.update(group)

    @property
    def frame_digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class TemporalReplayTransition:
    node_id: str
    from_status: str
    to_status: str
    at: float

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or len(self.node_id) != 64:
            raise ValueError("node_id must be a deterministic digest.")
        allowed = {"active", "not_yet_valid", "expired", "retracted"}
        if self.from_status not in allowed or self.to_status not in allowed or self.from_status == self.to_status:
            raise ValueError("temporal replay transition is invalid.")
        object.__setattr__(self, "at", _timestamp(self.at, "at"))


@dataclass(frozen=True)
class TemporalReplayReport:
    frames: tuple[TemporalReplayFrame, ...]
    transitions: tuple[TemporalReplayTransition, ...]

    @property
    def report_digest(self) -> str:
        return _digest(
            {
                "frames": [frame.frame_digest for frame in self.frames],
                "transitions": [asdict(transition) for transition in self.transitions],
            }
        )


def replay_temporal_evidence(
    nodes: Sequence[EvidenceNode],
    *,
    as_of_points: Sequence[float],
) -> TemporalReplayReport:
    """Replay explicit valid/retracted metadata over strictly increasing timestamps."""

    if isinstance(nodes, (str, bytes, bytearray)) or len(nodes) > _MAX_NODES:
        raise ValueError("nodes must be a bounded sequence.")
    values = tuple(nodes)
    if any(not isinstance(node, EvidenceNode) for node in values):
        raise ValueError("every node must be EvidenceNode.")
    if len({node.node_id for node in values}) != len(values):
        raise ValueError("node IDs must be unique.")
    if isinstance(as_of_points, (str, bytes, bytearray)) or not as_of_points or len(as_of_points) > _MAX_POINTS:
        raise ValueError("as_of_points must be a bounded non-empty sequence.")
    points = tuple(_timestamp(value, "as_of") for value in as_of_points)
    if tuple(sorted(set(points))) != points:
        raise ValueError("as_of_points must be strictly increasing and unique.")

    frames: list[TemporalReplayFrame] = []
    transitions: list[TemporalReplayTransition] = []
    previous: dict[str, str] = {}
    for point in points:
        groups: dict[str, list[str]] = {
            "active": [],
            "not_yet_valid": [],
            "expired": [],
            "retracted": [],
        }
        current: dict[str, str] = {}
        for node in values:
            status = temporal_evidence_status(node, as_of=point).status
            groups[status].append(node.node_id)
            current[node.node_id] = status
            old = previous.get(node.node_id)
            if old is not None and old != status:
                transitions.append(
                    TemporalReplayTransition(
                        node_id=node.node_id,
                        from_status=old,
                        to_status=status,
                        at=point,
                    )
                )
        frames.append(
            TemporalReplayFrame(
                as_of=point,
                active_node_ids=tuple(sorted(groups["active"])),
                not_yet_valid_node_ids=tuple(sorted(groups["not_yet_valid"])),
                expired_node_ids=tuple(sorted(groups["expired"])),
                retracted_node_ids=tuple(sorted(groups["retracted"])),
            )
        )
        previous = current
    return TemporalReplayReport(frames=tuple(frames), transitions=tuple(transitions))


__all__ = [
    "TemporalReplayFrame",
    "TemporalReplayReport",
    "TemporalReplayTransition",
    "replay_temporal_evidence",
]
