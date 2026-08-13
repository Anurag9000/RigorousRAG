from tools.adapter_registry import AdapterRegistry
from tools.retrieval_model_contracts import RetrievalModelSpec


def active_retrieval_spec(registry: AdapterRegistry, name: str, mode: str,
                          model_name: str, revision: str) -> RetrievalModelSpec:
    record = registry.active(name)
    if record is None:
        raise RuntimeError("active retrieval artifact is missing")
    if record.kind not in (mode, "retrieval-" + mode):
        raise ValueError("retrieval artifact kind mismatch")
    return RetrievalModelSpec(
        mode=mode,
        model_name=model_name,
        revision=revision,
        checksum_sha256=record.checksum_sha256,
    )


__all__ = ["active_retrieval_spec"]
