"""Lease-guarded execution for singleton distributed maintenance jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from tools.distributed_coordination import Lease, LeaseCoordinator

T = TypeVar("T")


class LeaseLostError(RuntimeError):
    """Raised when a running job can no longer prove lease ownership."""


@dataclass
class JobLeaseContext:
    coordinator: LeaseCoordinator
    lease: Lease
    ttl_seconds: float

    @property
    def fencing_token(self) -> int:
        return self.lease.token

    def heartbeat(self) -> Lease:
        renewed = self.coordinator.renew(self.lease, ttl_seconds=self.ttl_seconds)
        if renewed is None:
            raise LeaseLostError("distributed job lease was lost.")
        self.lease = renewed
        return renewed


@dataclass(frozen=True)
class JobResult(Generic[T]):
    job_name: str
    acquired: bool
    fencing_token: int | None
    value: T | None = None


class FencedJobRunner:
    """Run a job only while this worker owns a fenced lease.

    Long-running jobs should call ``context.heartbeat()`` at safe checkpoints and propagate the
    fencing token into writes where the destination supports compare-and-swap semantics.
    """

    def __init__(self, *, coordinator: LeaseCoordinator, holder: str, ttl_seconds: float = 30.0) -> None:
        self._coordinator = coordinator
        self._holder = holder
        self._ttl = ttl_seconds

    def run(self, job_name: str, task: Callable[[JobLeaseContext], T]) -> JobResult[T]:
        lease = self._coordinator.acquire(
            name=f"job:{job_name}", holder=self._holder, ttl_seconds=self._ttl
        )
        if lease is None:
            return JobResult(job_name=job_name, acquired=False, fencing_token=None)
        context = JobLeaseContext(self._coordinator, lease, self._ttl)
        try:
            value = task(context)
            context.heartbeat()
            return JobResult(
                job_name=job_name,
                acquired=True,
                fencing_token=context.fencing_token,
                value=value,
            )
        finally:
            self._coordinator.release(context.lease)


__all__ = ["FencedJobRunner", "JobLeaseContext", "JobResult", "LeaseLostError"]
