"""Owner-scoped signing and verification routes for resolved human-review decisions."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.attestation_keyring import RotatingManifestKeyring
from tools.manifest_attestation import canonical_manifest_bytes, verify_attestation
from tools.review_attestation import attest_review_record, review_manifest
from tools.review_attestation_store import ReviewAttestationStore, StoredReviewAttestation
from tools.review_store import ReviewStore
from tools.runtime_providers import RuntimeProviderRegistry
from tools.security import Principal
from tools.trust_provider_bootstrap import ATTESTATION_KEYRING_PROVIDER_ID


def _public(item: StoredReviewAttestation, *, include_manifest: bool = True) -> dict[str, Any]:
    signed = asdict(item.signed)
    output: dict[str, Any] = {
        "attestation_id": item.attestation_id,
        "owner_id": item.owner_id,
        "request_id": item.request_id,
        "lease_token": item.lease_token,
        "reviewer_id": item.reviewer_id,
        "resolution": item.resolution,
        "signed": signed,
        "created_at": item.created_at,
    }
    if include_manifest:
        output["captured_manifest"] = dict(item.captured_manifest)
    return output


def _keyring(providers: RuntimeProviderRegistry) -> RotatingManifestKeyring:
    try:
        provider = providers.require(ATTESTATION_KEYRING_PROVIDER_ID)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Review attestation signing/verification provider is unavailable.") from exc
    if not isinstance(provider, RotatingManifestKeyring):
        raise HTTPException(status_code=503, detail="Configured review attestation provider has an invalid contract.")
    return provider


def _owned_review(review_store: ReviewStore, owner_id: str, request_id: str):
    item = review_store.get(owner_id=owner_id, request_id=request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Review request not found.")
    return item


def _owned_attestation(
    store: ReviewAttestationStore,
    owner_id: str,
    request_id: str,
    attestation_id: str,
) -> StoredReviewAttestation:
    try:
        item = store.get(owner_id=owner_id, attestation_id=attestation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if item is None or item.request_id != request_id:
        raise HTTPException(status_code=404, detail="Review attestation not found.")
    return item


def build_review_attestation_router(
    *,
    principal_dependency: Callable[..., Any],
    review_store: ReviewStore,
    attestation_store: ReviewAttestationStore,
    providers: RuntimeProviderRegistry,
) -> APIRouter:
    router = APIRouter(tags=["governance", "review-attestation"])

    @router.post("/reviews/{request_id}/attest")
    async def attest(
        request_id: str,
        principal: Principal = Depends(principal_dependency),
    ) -> dict[str, Any]:
        current = _owned_review(review_store, principal.owner_id, request_id)
        if current.state != "resolved" or not current.reviewer_id or not current.resolution:
            raise HTTPException(status_code=409, detail="Only a resolved reviewer-bound review may be attested.")
        keyring = _keyring(providers)
        try:
            manifest = review_manifest(current)
            signed = attest_review_record(current, keyring.signer())
            stored = attestation_store.put(
                owner_id=principal.owner_id,
                captured_manifest=manifest,
                signed=signed,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="Review attestation could not be signed or persisted.") from exc
        return _public(stored)

    @router.get("/reviews/{request_id}/attestations")
    async def list_attestations(
        request_id: str,
        limit: int = Query(default=100, ge=1, le=10000),
        principal: Principal = Depends(principal_dependency),
    ) -> list[dict[str, Any]]:
        _owned_review(review_store, principal.owner_id, request_id)
        try:
            items = attestation_store.list(owner_id=principal.owner_id, request_id=request_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return [_public(item, include_manifest=False) for item in items]

    @router.get("/reviews/{request_id}/attestations/{attestation_id}")
    async def get_attestation(
        request_id: str,
        attestation_id: str,
        principal: Principal = Depends(principal_dependency),
    ) -> dict[str, Any]:
        _owned_review(review_store, principal.owner_id, request_id)
        return _public(_owned_attestation(attestation_store, principal.owner_id, request_id, attestation_id))

    @router.get("/reviews/{request_id}/attestations/{attestation_id}/verify")
    async def verify(
        request_id: str,
        attestation_id: str,
        principal: Principal = Depends(principal_dependency),
    ) -> dict[str, Any]:
        current = _owned_review(review_store, principal.owner_id, request_id)
        stored = _owned_attestation(attestation_store, principal.owner_id, request_id, attestation_id)
        keyring = _keyring(providers)
        signature_valid = verify_attestation(stored.signed.attestation, stored.captured_manifest, keyring)
        current_manifest_sha256 = ""
        matches_current = False
        current_manifest_available = current.state == "resolved" and bool(current.reviewer_id) and bool(current.resolution)
        if current_manifest_available:
            try:
                current_manifest = review_manifest(current)
                current_manifest_sha256 = hashlib.sha256(canonical_manifest_bytes(current_manifest)).hexdigest()
                matches_current = current_manifest_sha256 == stored.signed.review_manifest_sha256
            except (TypeError, ValueError):
                current_manifest_available = False
        return {
            "attestation_id": stored.attestation_id,
            "request_id": stored.request_id,
            "signature_valid_for_captured_state": bool(signature_valid),
            "matches_current_review_state": bool(matches_current),
            "current_review_manifest_available": bool(current_manifest_available),
            "captured_review_manifest_sha256": stored.signed.review_manifest_sha256,
            "current_review_manifest_sha256": current_manifest_sha256,
            "key_id": stored.signed.attestation.key_id,
            "algorithm": stored.signed.attestation.algorithm,
            "signed_at": stored.signed.attestation.signed_at,
        }

    return router


__all__ = ["build_review_attestation_router"]
