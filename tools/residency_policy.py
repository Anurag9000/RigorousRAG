from dataclasses import dataclass


@dataclass(frozen=True)
class ResidencyPolicy:
    regions: frozenset[str]
    providers: frozenset[str] = frozenset()

    def allows(self, *, region: str, provider: str) -> bool:
        if region not in self.regions:
            return False
        return not self.providers or provider in self.providers


@dataclass(frozen=True)
class ResidencyTarget:
    provider: str
    region: str
    service: str


def target_allowed(policy: ResidencyPolicy, target: ResidencyTarget) -> bool:
    return policy.allows(region=target.region, provider=target.provider)


__all__ = ["ResidencyPolicy", "ResidencyTarget", "target_allowed"]
