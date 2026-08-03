"""Canonical live-path and chronology integrity for custody chain exports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tools import evidence_graph_set_signed_retirement_restore_custody_export as _base
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_reconcile import (
    artifact_path_digest,
)

_ORIGINAL_BUILD = getattr(
    _base,
    "_unfiltered_build_restore_chain_of_custody",
    _base.build_restore_chain_of_custody,
)
_ORIGINAL_VERIFY = getattr(
    _base,
    "_unchronologized_verify_restore_chain_of_custody",
    _base.verify_restore_chain_of_custody,
)
_base._unfiltered_build_restore_chain_of_custody = _ORIGINAL_BUILD
_base._unchronologized_verify_restore_chain_of_custody = _ORIGINAL_VERIFY


def _validate_chronology(manifest: Any) -> None:
    if not (
        manifest.pre_bound_at
        <= manifest.restore_completed_at
        <= manifest.post_bound_at
        <= manifest.generated_at
    ):
        raise ValueError("custody chain chronology is invalid.")
    if any(
        artifact.completed_at > manifest.pre_bound_at
        for artifact in manifest.artifacts
    ):
        raise ValueError("artifact publication follows pre-bound custody.")


def _with_artifacts(manifest: Any, artifacts: tuple[Any, ...]):
    payload = asdict(manifest)
    payload["artifacts"] = [asdict(value) for value in artifacts]
    payload.pop("chain_digest", None)
    for key in (
        "contains_source_text",
        "contains_assertion_secrets",
        "contains_raw_paths",
        "mutation_performed",
    ):
        payload.pop(key, None)
    stable = {
        "scope": "rigorousrag-external-restore-chain-of-custody-v1",
        **payload,
    }
    return _base.RestoreChainOfCustodyManifest(
        owner_id=manifest.owner_id,
        restore_id=manifest.restore_id,
        snapshot_digest=manifest.snapshot_digest,
        target_path_digest=manifest.target_path_digest,
        snapshot_record_count=manifest.snapshot_record_count,
        restore_target_verification_digest=(
            manifest.restore_target_verification_digest
        ),
        restore_completed_at=manifest.restore_completed_at,
        custody_id=manifest.custody_id,
        custody_manifest_digest=manifest.custody_manifest_digest,
        pre_receipt_digest=manifest.pre_receipt_digest,
        backup_sha256=manifest.backup_sha256,
        backup_size_bytes=manifest.backup_size_bytes,
        pre_actor_id_digest=manifest.pre_actor_id_digest,
        pre_binding_method=manifest.pre_binding_method,
        pre_binding_digest=manifest.pre_binding_digest,
        pre_bound_at=manifest.pre_bound_at,
        post_receipt_digest=manifest.post_receipt_digest,
        post_target_verification_digest=(
            manifest.post_target_verification_digest
        ),
        post_actor_id_digest=manifest.post_actor_id_digest,
        post_binding_method=manifest.post_binding_method,
        post_binding_digest=manifest.post_binding_digest,
        post_bound_at=manifest.post_bound_at,
        legal_hold_status=manifest.legal_hold_status,
        artifacts=artifacts,
        generated_at=manifest.generated_at,
        chain_digest=_base._canonical_digest(stable),
    )


def build_restore_chain_of_custody(**kwargs: Any):
    manifest = _ORIGINAL_BUILD(**kwargs)
    backup_digest = artifact_path_digest(
        kwargs["backup_path"],
        label="backup_path",
    )
    receipt_digest = artifact_path_digest(
        kwargs["pre_receipt_path"],
        label="pre_receipt_path",
    )
    matching = tuple(
        artifact
        for artifact in manifest.artifacts
        if artifact.backup_path_digest == backup_digest
        and artifact.receipt_path_digest == receipt_digest
    )
    if len(matching) != 1:
        raise RuntimeError(
            "chain requires exactly one completed artifact intent for live paths."
        )
    if matching != manifest.artifacts:
        manifest = _with_artifacts(manifest, matching)
    _validate_chronology(manifest)
    return manifest


def verify_restore_chain_of_custody(path):
    manifest = _ORIGINAL_VERIFY(path)
    _validate_chronology(manifest)
    return manifest


_base.build_restore_chain_of_custody = build_restore_chain_of_custody
_base.verify_restore_chain_of_custody = verify_restore_chain_of_custody


__all__ = [
    "build_restore_chain_of_custody",
    "verify_restore_chain_of_custody",
]
