"""Versioned runtime confidence calibrators selected by corpus and benchmark profile."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class CalibrationPoint:
    raw: float
    calibrated: float

    def __post_init__(self) -> None:
        for name, value in (("raw", self.raw), ("calibrated", self.calibrated)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")


@dataclass(frozen=True)
class CalibrationProfile:
    calibrator_id: str
    version: str
    corpus_profile: str
    benchmark: str
    points: Tuple[CalibrationPoint, ...]
    answer_threshold: float = 0.5

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.calibrator_id, self.version, self.corpus_profile, self.benchmark)):
            raise ValueError("calibrator identity, version, corpus profile, and benchmark are required.")
        if not self.points:
            raise ValueError("at least one calibration point is required.")
        raw_values = [point.raw for point in self.points]
        calibrated_values = [point.calibrated for point in self.points]
        if raw_values != sorted(raw_values) or len(set(raw_values)) != len(raw_values):
            raise ValueError("raw calibration points must be strictly increasing.")
        if calibrated_values != sorted(calibrated_values):
            raise ValueError("calibrated values must be non-decreasing.")
        if not 0.0 <= float(self.answer_threshold) <= 1.0:
            raise ValueError("answer_threshold must be in [0, 1].")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def calibrate(self, raw_confidence: float) -> float:
        raw = float(raw_confidence)
        if not math.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise ValueError("raw_confidence must be finite and in [0, 1].")
        points = self.points
        if raw <= points[0].raw:
            return points[0].calibrated
        if raw >= points[-1].raw:
            return points[-1].calibrated
        for left, right in zip(points, points[1:]):
            if left.raw <= raw <= right.raw:
                width = right.raw - left.raw
                fraction = (raw - left.raw) / width
                return left.calibrated + fraction * (right.calibrated - left.calibrated)
        raise RuntimeError("calibration interpolation failed.")


class RuntimeCalibratorRegistry:
    """Persistent profile registry with explicit active selection and no silent fallback."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._profiles: dict[tuple[str, str, str, str], CalibrationProfile] = {}
        self._active: dict[tuple[str, str], tuple[str, str]] = {}
        if self.path.exists():
            self._load()

    def register(self, profile: CalibrationProfile) -> None:
        key = (
            profile.corpus_profile,
            profile.benchmark,
            profile.calibrator_id,
            profile.version,
        )
        existing = self._profiles.get(key)
        if existing is not None and existing != profile:
            raise ValueError("a different calibrator already exists at this exact identity.")
        self._profiles[key] = profile
        self._persist()

    def activate(
        self,
        *,
        corpus_profile: str,
        benchmark: str,
        calibrator_id: str,
        version: str,
    ) -> CalibrationProfile:
        key = (corpus_profile, benchmark, calibrator_id, version)
        try:
            profile = self._profiles[key]
        except KeyError as exc:
            raise KeyError("requested calibrator is not registered.") from exc
        self._active[(corpus_profile, benchmark)] = (calibrator_id, version)
        self._persist()
        return profile

    def selected(self, *, corpus_profile: str, benchmark: str) -> CalibrationProfile:
        try:
            calibrator_id, version = self._active[(corpus_profile, benchmark)]
            return self._profiles[(corpus_profile, benchmark, calibrator_id, version)]
        except KeyError as exc:
            raise KeyError(
                "no active calibrator exists for the requested corpus/benchmark profile."
            ) from exc

    def calibrate(self, raw_confidence: float, *, corpus_profile: str, benchmark: str) -> float:
        return self.selected(corpus_profile=corpus_profile, benchmark=benchmark).calibrate(raw_confidence)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "profiles": [
                asdict(profile)
                for _, profile in sorted(self._profiles.items(), key=lambda item: item[0])
            ],
            "active": [
                {
                    "corpus_profile": corpus,
                    "benchmark": benchmark,
                    "calibrator_id": value[0],
                    "version": value[1],
                }
                for (corpus, benchmark), value in sorted(self._active.items())
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for raw in payload.get("profiles", []):
            profile = CalibrationProfile(
                calibrator_id=raw["calibrator_id"],
                version=raw["version"],
                corpus_profile=raw["corpus_profile"],
                benchmark=raw["benchmark"],
                points=tuple(CalibrationPoint(**point) for point in raw["points"]),
                answer_threshold=float(raw.get("answer_threshold", 0.5)),
            )
            key = (
                profile.corpus_profile,
                profile.benchmark,
                profile.calibrator_id,
                profile.version,
            )
            if key in self._profiles:
                raise ValueError("duplicate calibrator identity in registry.")
            self._profiles[key] = profile
        for raw in payload.get("active", []):
            selector = (raw["corpus_profile"], raw["benchmark"])
            identity = (raw["calibrator_id"], raw["version"])
            profile_key = selector + identity
            if profile_key not in self._profiles:
                raise ValueError("active calibrator points to a missing profile.")
            self._active[selector] = identity
