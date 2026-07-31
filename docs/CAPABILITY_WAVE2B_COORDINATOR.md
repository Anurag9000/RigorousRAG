# Capability Wave 2B coordinator foundation

## Implemented contracts

- strict owner/document vector-generation snapshots;
- complete row/document/metadata validation and exact batch restoration;
- fixed-stripe owner/document locks without an unbounded lock registry;
- vector-first, sparse-second replacement under one coordinator;
- prior vector and sparse generation capture before mutation;
- compensation after partial vector failure or sparse failure;
- explicit `IndexCoordinationError.rollback_errors` when restoration is incomplete;
- coordinated vector+sparse deletion with restoration after either-store failure;
- immutable generation manifests containing owner, document, content hash, profile
  fingerprint, vector row count and sparse generation;
- owner-scoped vector-only/sparse-only/aligned reconciliation scans.

## Focused local verification

The coordinator tests cover exact capture/restore, cross-owner filtering, successful
replacement, sparse failure, partial vector failure, rollback failure, deletion failure,
absent deletion, reconciliation and pre-mutation validation. The combined capability suite
covers Wave 1, Wave 2A and this coordinator foundation.

## Deliberate non-authoritative status

The coordinator is not yet called by `tools.document_service.index_document`, API/batch
ingestion or API deletion. Existing vector-only behavior therefore remains authoritative
until the integration commit wires all entrypoints, retry/recovery, registry/source cleanup
and cross-store reconciliation together. This avoids publishing a partially dual-written
production path.
