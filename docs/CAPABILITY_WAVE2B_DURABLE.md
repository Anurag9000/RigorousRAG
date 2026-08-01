# Capability Wave 2B — authoritative durable index generations

Last updated: 2026-08-01

## Purpose

Wave 2B turns the persistent sparse index from an optional side store into part of the document lifecycle. A successful indexed document now has three aligned states:

1. owner-scoped vector rows;
2. one owner-scoped sparse document generation;
3. one durable current-generation pointer with immutable history.

The current pointer is the authority used by generation-validated retrieval. It records the finalized content hash, embedding-profile fingerprint, vector-row count and sparse generation.

## Implemented components

- `tools/vector_generation.py`
  - bounded complete vector snapshots;
  - owner/document metadata validation;
  - exact batch restoration;
  - raw vector deletion that cannot recurse into public lifecycle deletion;
  - cleanup after partial restoration failure.
- `tools/sparse_fields.py`
  - deterministic title, heading, abstract, body, caption, table and reference fields;
  - stable field IDs;
  - page, section, field and metadata provenance.
- `tools/generation_store.py`
  - append-only history;
  - optimistic current pointers;
  - active, deleted and restored states;
  - strict finite JSON and database/path identity checks.
- `tools/three_store_coordinator.py`
  - one reentrant owner/document lock across vector, sparse and manifest operations;
  - compensation after index or manifest failure;
  - tombstoned deletion;
  - bounded drift classification.
- `tools/raw_index_coordinator.py`
  - internal vector+sparse deletion that bypasses the public lifecycle hook.
- `tools/authoritative_document_index.py`
  - privacy-finalized commit boundary;
  - authoritative snapshot and restoration API;
  - reload-idempotent public `RAGLayer.delete_document` installation.
- `tools/sparse_runtime.py`
  - path-keyed sparse, generation and coordinator factories.
- `tools/index_reconciliation.py` and `tools/index_reconciliation_cli.py`
  - dry-run repair plans;
  - exact-confirmation cleanup for deleted-generation residue;
  - bounded public JSON without source paths or document contents.

## Production wiring

- `tools.document_service.index_document` is the only final ingestion write boundary and commits all three stores.
- API ingestion and durable retry paths use the shared document service.
- Public RAG deletion coordinates vector, sparse and durable state before registry/source cleanup.
- `ingest_docs.py` captures an authoritative pre-index snapshot and restores all three stores if registry publication fails.
- `.env.example` defines independent vector, sparse and generation locations.
- exact-head CI isolates every durable store under the runner temporary directory.

## Transaction sequence

### Replacement

1. acquire the stable owner/document `RLock`;
2. capture vector, sparse and current-generation state;
3. write the new vector generation;
4. write the new sparse generation;
5. append the durable active record and move the current pointer;
6. on failure, restore vector, sparse and current-generation state;
7. report every incomplete compensation explicitly.

### Deletion

1. capture all prior state under the same lock;
2. remove raw vector rows and sparse state;
3. append a durable deleted record;
4. restore prior state if tombstoning fails;
5. only then may caller-owned registry/source cleanup proceed.

## Drift categories

- vector only;
- sparse only;
- vector+sparse without manifest;
- active manifest without both stores;
- deleted manifest with residual store rows;
- content/profile/generation metadata mismatch;
- inspection failure;
- healthy aligned generation.

Only deleted-generation residue is currently eligible for automatic repair, and only with the exact confirmation phrase. Every other category requires retained-source inspection and reindexing or an explicitly reviewed adoption workflow.

## Verification boundary

Focused tests exist for snapshots, restoration, append-only history, optimistic sequences, transaction compensation, public/raw deletion seams, batch rollback, path identity, corrupt JSON, deterministic sparse fields and reconciliation output. The complete exact-head Linux, Windows, container and release-lock matrix is still the release authority and is not claimed green for the current head.

## Residual limitations

- SQLite/process-local locking is not distributed exactly-once infrastructure.
- A process crash between external-store operations can still create drift; startup/operator reconciliation must detect it.
- Existing pre-Wave-2B documents do not have authoritative manifests and require migration/reindexing.
- The retained-document registry is not yet included as a fourth transaction participant.
- Automatic repair intentionally remains narrower than detection.
