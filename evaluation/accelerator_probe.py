"""Accelerator measurement contract for real provider-supplied observations."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AcceleratorSample:
    backend: str
    device: str
    allocated_bytes: int | None = None
    peak_bytes: int | None = None
    power_watts: float | None = None


class CallbackAcceleratorProbe:
    def __init__(self, reader: Callable[[], AcceleratorSample]) -> None:
        if not callable(reader):
            raise ValueError("reader must be callable")
        self.reader = reader

    def sample(self) -> AcceleratorSample:
        value = self.reader()
        if not isinstance(value, AcceleratorSample):
            raise RuntimeError("accelerator reader returned an invalid sample")
        return value


__all__ = ["AcceleratorSample", "CallbackAcceleratorProbe"]
