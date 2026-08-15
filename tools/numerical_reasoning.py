"""Unit-aware numerical reasoning with source-linked operands.

The engine performs deterministic conversions/arithmetic and first-order independent
uncertainty propagation.  It refuses dimensionally invalid operations and preserves
citation/source identifiers for every derived quantity.  It is not a symbolic algebra
system and does not infer missing units from prose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_BASE_DIMENSIONS = ("length", "mass", "time", "temperature", "amount", "current", "luminosity")


def _text(value: Any, label: str, maximum: int = 200, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class Dimension:
    powers: tuple[int, int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0, 0)

    def __post_init__(self) -> None:
        if len(self.powers) != len(_BASE_DIMENSIONS) or any(isinstance(value, bool) or not isinstance(value, int) or abs(value) > 20 for value in self.powers):
            raise ValueError("dimension powers are invalid")

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.powers, other.powers)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.powers, other.powers)))

    def power(self, exponent: int) -> "Dimension":
        if isinstance(exponent, bool) or not isinstance(exponent, int) or abs(exponent) > 20:
            raise ValueError("dimension exponent is invalid")
        return Dimension(tuple(value * exponent for value in self.powers))


DIMENSIONLESS = Dimension()
LENGTH = Dimension((1, 0, 0, 0, 0, 0, 0))
MASS = Dimension((0, 1, 0, 0, 0, 0, 0))
TIME = Dimension((0, 0, 1, 0, 0, 0, 0))
TEMPERATURE = Dimension((0, 0, 0, 1, 0, 0, 0))
AREA = LENGTH.power(2)
VOLUME = LENGTH.power(3)
VELOCITY = LENGTH / TIME
DISCHARGE = VOLUME / TIME
PRECIPITATION_RATE = LENGTH / TIME


@dataclass(frozen=True)
class UnitDefinition:
    symbol: str
    dimension: Dimension
    scale_to_si: float = 1.0
    offset_to_si: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _text(self.symbol, "unit symbol", 64))
        if not isinstance(self.dimension, Dimension):
            raise ValueError("dimension must be Dimension")
        scale = _finite(self.scale_to_si, "scale_to_si")
        if scale == 0:
            raise ValueError("scale_to_si may not be zero")
        object.__setattr__(self, "scale_to_si", scale)
        object.__setattr__(self, "offset_to_si", _finite(self.offset_to_si, "offset_to_si"))

    def to_si(self, value: float) -> float:
        return (_finite(value, "value") + self.offset_to_si) * self.scale_to_si

    def from_si(self, value: float) -> float:
        return _finite(value, "value") / self.scale_to_si - self.offset_to_si


class UnitRegistry:
    def __init__(self) -> None:
        self._units: dict[str, UnitDefinition] = {}

    def register(self, unit: UnitDefinition, *aliases: str) -> None:
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition")
        keys = (unit.symbol, *aliases)
        for key in keys:
            normalized = _text(key, "unit alias", 64).casefold()
            existing = self._units.get(normalized)
            if existing is not None and existing != unit:
                raise ValueError("unit alias already maps to another definition")
            self._units[normalized] = unit

    def resolve(self, symbol: str) -> UnitDefinition:
        return self._units[_text(symbol, "unit symbol", 64).casefold()]

    def convert(self, value: float, source: str, target: str) -> float:
        left, right = self.resolve(source), self.resolve(target)
        if left.dimension != right.dimension:
            raise ValueError("cannot convert across different dimensions")
        return right.from_si(left.to_si(value))


def default_unit_registry() -> UnitRegistry:
    registry = UnitRegistry()
    registry.register(UnitDefinition("1", DIMENSIONLESS), "dimensionless")
    registry.register(UnitDefinition("m", LENGTH), "meter", "metre")
    registry.register(UnitDefinition("km", LENGTH, 1000.0), "kilometer", "kilometre")
    registry.register(UnitDefinition("mm", LENGTH, 0.001), "millimeter", "millimetre")
    registry.register(UnitDefinition("ft", LENGTH, 0.3048), "foot", "feet")
    registry.register(UnitDefinition("m2", AREA), "m^2")
    registry.register(UnitDefinition("km2", AREA, 1_000_000.0), "km^2")
    registry.register(UnitDefinition("m3", VOLUME), "m^3")
    registry.register(UnitDefinition("MCM", VOLUME, 1_000_000.0), "million_cubic_metres")
    registry.register(UnitDefinition("s", TIME), "second")
    registry.register(UnitDefinition("min", TIME, 60.0), "minute")
    registry.register(UnitDefinition("h", TIME, 3600.0), "hour")
    registry.register(UnitDefinition("day", TIME, 86400.0), "d")
    registry.register(UnitDefinition("m/s", VELOCITY), "mps")
    registry.register(UnitDefinition("m3/s", DISCHARGE), "cumec", "cumecs", "cms")
    registry.register(UnitDefinition("mm/h", PRECIPITATION_RATE, 0.001 / 3600.0))
    registry.register(UnitDefinition("degC", TEMPERATURE, 1.0, 273.15), "C", "celsius")
    registry.register(UnitDefinition("K", TEMPERATURE), "kelvin")
    return registry


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    uncertainty: float = 0.0
    source_ids: tuple[str, ...] = ()
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, "value"))
        uncertainty = _finite(self.uncertainty, "uncertainty")
        if uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "unit", _text(self.unit, "unit", 64))
        if len(self.source_ids) > 100:
            raise ValueError("source_ids exceed the item limit")
        object.__setattr__(self, "source_ids", tuple(dict.fromkeys(_text(item, "source_id", 500) for item in self.source_ids)))
        object.__setattr__(self, "label", _text(self.label, "label", 500, allow_empty=True))

    def convert(self, target_unit: str, registry: UnitRegistry) -> "Quantity":
        source_def, target_def = registry.resolve(self.unit), registry.resolve(target_unit)
        if source_def.dimension != target_def.dimension:
            raise ValueError("cannot convert quantity across dimensions")
        value_si = source_def.to_si(self.value)
        converted = target_def.from_si(value_si)
        # Offsets do not affect absolute uncertainty; only scale does.
        uncertainty_si = self.uncertainty * abs(source_def.scale_to_si)
        target_uncertainty = uncertainty_si / abs(target_def.scale_to_si)
        return Quantity(converted, target_def.symbol, target_uncertainty, self.source_ids, self.label)


def _merge_sources(*quantities: Quantity) -> tuple[str, ...]:
    return tuple(dict.fromkeys(source for quantity in quantities for source in quantity.source_ids))


def add(left: Quantity, right: Quantity, registry: UnitRegistry) -> Quantity:
    right_converted = right.convert(left.unit, registry)
    uncertainty = math.hypot(left.uncertainty, right_converted.uncertainty)
    return Quantity(left.value + right_converted.value, left.unit, uncertainty, _merge_sources(left, right), "sum")


def subtract(left: Quantity, right: Quantity, registry: UnitRegistry) -> Quantity:
    right_converted = right.convert(left.unit, registry)
    uncertainty = math.hypot(left.uncertainty, right_converted.uncertainty)
    return Quantity(left.value - right_converted.value, left.unit, uncertainty, _merge_sources(left, right), "difference")


@dataclass(frozen=True)
class DerivedQuantity:
    value_si: float
    dimension: Dimension
    uncertainty_si: float
    source_ids: tuple[str, ...]
    operation: str


def multiply(left: Quantity, right: Quantity, registry: UnitRegistry) -> DerivedQuantity:
    ldef, rdef = registry.resolve(left.unit), registry.resolve(right.unit)
    lv, rv = ldef.to_si(left.value), rdef.to_si(right.value)
    value = lv * rv
    lu, ru = left.uncertainty * abs(ldef.scale_to_si), right.uncertainty * abs(rdef.scale_to_si)
    variance = (rv * lu) ** 2 + (lv * ru) ** 2
    return DerivedQuantity(value, ldef.dimension * rdef.dimension, math.sqrt(variance), _merge_sources(left, right), "multiply")


def divide(left: Quantity, right: Quantity, registry: UnitRegistry) -> DerivedQuantity:
    ldef, rdef = registry.resolve(left.unit), registry.resolve(right.unit)
    lv, rv = ldef.to_si(left.value), rdef.to_si(right.value)
    if rv == 0:
        raise ZeroDivisionError("cannot divide by a zero quantity")
    value = lv / rv
    lu, ru = left.uncertainty * abs(ldef.scale_to_si), right.uncertainty * abs(rdef.scale_to_si)
    variance = (lu / rv) ** 2 + ((lv * ru) / (rv * rv)) ** 2
    return DerivedQuantity(value, ldef.dimension / rdef.dimension, math.sqrt(variance), _merge_sources(left, right), "divide")


def weighted_mean(values: Sequence[Quantity], weights: Sequence[float], registry: UnitRegistry) -> Quantity:
    if not values or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and aligned")
    target = values[0].unit
    converted = [item.convert(target, registry) for item in values]
    normalized_weights = [_finite(weight, "weight") for weight in weights]
    if any(weight < 0 for weight in normalized_weights) or sum(normalized_weights) <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    total = sum(normalized_weights)
    value = sum(weight * item.value for weight, item in zip(normalized_weights, converted)) / total
    uncertainty = math.sqrt(sum((weight / total * item.uncertainty) ** 2 for weight, item in zip(normalized_weights, converted)))
    return Quantity(value, target, uncertainty, _merge_sources(*values), "weighted_mean")


__all__ = [
    "AREA", "DIMENSIONLESS", "DISCHARGE", "Dimension", "DerivedQuantity", "LENGTH",
    "MASS", "PRECIPITATION_RATE", "Quantity", "TEMPERATURE", "TIME", "UnitDefinition",
    "UnitRegistry", "VELOCITY", "VOLUME", "add", "default_unit_registry", "divide",
    "multiply", "subtract", "weighted_mean",
]
