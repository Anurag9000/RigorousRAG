from evaluation.accelerator_probe import AcceleratorSample, CallbackAcceleratorProbe
from tools.key_management import CallbackKeyProvider, WrappedKey
from tools.residency_policy import ResidencyPolicy, ResidencyTarget, target_allowed


def test_accelerator_probe_uses_supplied_observation():
    sample = AcceleratorSample("provider", "device-0", 10, 20, 30.0)
    assert CallbackAcceleratorProbe(lambda: sample).sample() == sample


def test_residency_policy_restricts_region_and_provider():
    policy = ResidencyPolicy(frozenset({"in-south-1"}), frozenset({"provider-a"}))
    assert target_allowed(policy, ResidencyTarget("provider-a", "in-south-1", "vectors"))
    assert not target_allowed(policy, ResidencyTarget("provider-a", "us-east-1", "vectors"))
    assert not target_allowed(policy, ResidencyTarget("provider-b", "in-south-1", "vectors"))


def test_external_key_provider_contract_round_trip():
    wrapped = WrappedKey("provider-a", "key-1", "v2", "in-south-1", b"wrapped")
    provider = CallbackKeyProvider(
        lambda value, context: wrapped,
        lambda value, context: b"plain-key",
    )
    assert provider.wrap(b"plain-key", {"owner": "owner-a"}) == wrapped
    assert provider.unwrap(wrapped, {"owner": "owner-a"}) == b"plain-key"
