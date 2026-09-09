#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v37.

v37 keeps the complete v34 scientific/source/retained-trainable closure contract
and advances only the accepted estate wiring to the canonical rate-limit-safe v37
bootstrap and preservation adapter. Every live owner repository except OPF_ADP
must expose a root launcher wired to one of those immutable authorities. Local
reports must still satisfy the full v34 closure: no retained trainer, model,
dataset/task/loss/optimizer/scheduler/sampler/augmentation/config/registry/
combination/ensemble/workflow or other trainable scientific source may be hidden
behind broad ignore/dynamic coverage or an unsupported exemption.

This is a source certificate only. It intentionally does not download datasets,
run training, or claim empirical/runtime correctness.
"""
from __future__ import annotations

import account_wide_training_control_audit_v34 as prior

CANONICAL_BOOTSTRAP_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CANONICAL_BOOTSTRAP_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CANONICAL_ADAPTER_COMMIT = "ab4ad585a3cfd0b0350a163983ead9ad7a51d559"
CANONICAL_ADAPTER_BLOB = "046ef85de71ac2d9852cd43d615d4db8c8ee5f1c"
CERTIFICATE_SCHEMA = 37


def main() -> int:
    # Reuse the complete v34 remote/local scientific closure implementation,
    # changing only the immutable authority identities and certificate schema.
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    prior.CERTIFICATE_SCHEMA = CERTIFICATE_SCHEMA
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
