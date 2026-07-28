# Continuation Audit — Pass Three

## Scope

This pass continued the regression audit after `CONTINUATION_AUDIT_PASS2.md`. It focused on the two implementation risks explicitly left open in the draft pull request, then followed the same filesystem and visual-evidence invariants into adjacent upload, retention, cleanup, and rendering paths.

This record describes source changes and regression contracts. It does **not** claim that the exact branch head has passed compilation, lint, tests, coverage, or container execution. The pull request remains draft until executable checks run against the exact release commit.

## 1. Classic generation cleanup could race a concurrent reader

### Finding

Classic crawl state, sparse index, and PageRank were correctly committed by publishing `snapshot_manifest.json` last. However, the writer immediately deleted all prior generation files after publishing the new manifest. A separate process could read the previous manifest, pause, and then lose one of its referenced generation members during writer cleanup.

### Correction

- added `.snapshot.lock` in the classic storage directory;
- combined the existing process-local `RLock` with an operating-system advisory file lock;
- use `fcntl.flock` on POSIX and `msvcrt.locking` on Windows;
- reject a symbolic-link lock path;
- hold the lock across the complete manifest/member read;
- hold the same lock across generation writes, manifest publication, and old-generation cleanup;
- preserve manifest-last atomicity while preventing reader/cleanup interleaving.

### Regression contract

A reader deliberately pauses after reading the old manifest. A writer then attempts to publish and clean up a replacement generation. The test requires the writer to remain blocked until the reader finishes loading every old member, after which the writer publishes normally and a later reader observes the new generation.

## 2. Document routes could initialize embeddings and access Chroma on the event loop

### Finding

`/docs/list` and `/docs/{doc_id}` called `get_rag_layer()` before their worker offload. First-use embedding initialization, Chroma scans, vector deletion, registry joins, and retained-file cleanup could therefore block FastAPI's async event loop.

### Correction

- added synchronous owner-scoped list/delete service helpers;
- moved RAG initialization, vector access, registry joins, and cleanup inside the shared bounded research executor;
- applied the existing running-plus-pending admission ceiling and whole-route deadline to document routes;
- saturation returns generic `503` with `Retry-After`;
- timeouts return the existing generic `504`;
- upload copying, queue persistence, and ingestion submission were also moved off the event loop.

### Regression contract

Tests record thread names for RAG initialization, document listing, vector lookup/deletion, registry operations, and source cleanup. Every operation must run in a `rigorousrag-query` worker. Separate tests require both document routes to fail closed when the shared executor is saturated.

## 3. Owner upload directories could be swapped after path validation

### Finding

Random upload names and `Path.resolve()` containment checks prevented ordinary traversal, but path validation and final create/unlink were separate operations. A same-host actor capable of renaming or replacing an owner directory could redirect a later path-based write or deletion.

### Correction

Added `tools/upload_storage.py` with descriptor-anchored primitives:

- open the upload root and owner directory with no-follow directory descriptors on POSIX;
- create an owner directory relative to the already-opened root;
- create random files relative to the already-opened owner descriptor with `O_EXCL` and `O_NOFOLLOW`;
- use private file mode `0600`, bounded streaming, flush, `fsync`, and partial-write cleanup;
- delete only regular final entries through descriptor-relative lookup;
- reject owner-directory symbolic links and paths outside the exact `UPLOAD_DIR/<owner>/<file>` shape;
- provide a conservative Windows fallback with repeated symlink and directory-identity checks.

The HTTP upload endpoint now uses these primitives and returns a generic storage-unavailable `503` for unsafe local storage while preserving `413` for byte-limit violations.

### Regression contract

Tests cover:

- randomized private regular files;
- oversized partial-write cleanup;
- owner-directory symlink refusal without outside writes;
- out-of-root deletion refusal;
- owner-directory replacement after an upload path was recorded;
- public endpoint fail-closed behavior and generic error text.

## 4. CLI retention and orphan cleanup used separate path-based operations

### Finding

`DocumentStore.copy_source()`, `remove_source()`, and orphan cleanup still used path resolution followed by a later open or unlink. They therefore did not share the HTTP upload path's new descriptor-anchored invariant.

### Correction

- `copy_source()` opens the original with no-follow semantics, validates the opened inode as a bounded regular file, and writes it through the same randomized owner storage primitive;
- `remove_source()` uses descriptor-relative regular-file deletion;
- orphan cleanup delegates deletion to the same primitive;
- active and retained path resolution uses anchored owner-file validation;
- genuine renamed/unreferenced files may still be removed, but replacement owner symlinks are never followed.

### Regression contract

Tests cover symlinked CLI sources, source size ceilings, random retained filenames, owner-directory swaps, and orphan cleanup that leaves an outside symlink target untouched.

## 5. Visual verification and rendering reopened a mutable pathname

### Finding

The registry verified owner scope, source hash, document ID, page count, and render geometry, returned a `Path`, and the visual tool reopened that path. A host-side replacement between verification and rendering could make the vision model inspect bytes different from those used for identity verification.

### Correction

- added bounded descriptor-relative retained-file reads;
- `DocumentStore.source_bytes()` returns one immutable byte snapshot only after owner/document SHA-256 identity verification and PDF complexity preflight;
- PDF preflight opens the same bytes with `fitz.open(stream=..., filetype="pdf")`;
- the visual renderer consumes those same bytes and never reopens a retained path;
- preallocation geometry, actual rendered pixels, and exact encoded bytes remain independently bounded;
- the original scientific module is preserved byte-for-byte in `tools/integrity_legacy.py`;
- `tools/integrity.py` is a compatibility shim that overrides only visual extraction while retaining the original module object and monkeypatch behavior for every other scientific function.

### Regression contract

Tests require:

- an identity-verified PDF to return its exact retained bytes;
- mutation to make `source_bytes()` unavailable;
- owner-directory symlink replacement to make anchored reads fail closed;
- figure extraction to accept bytes rather than a path;
- visual entailment to pass the exact registry snapshot to the renderer while a fake `source_path()` raises if called.

## Resolved risks from the previous PR description

The following previously listed implementation risks are now addressed in source and regression tests:

1. classic snapshot cleanup versus concurrent readers;
2. synchronous RAG initialization/vector access on document-route event-loop paths;
3. upload and retained-source owner-directory swap races;
4. retained-PDF verification-to-render pathname reopening.

## Remaining boundaries

- The exact current head still has no observed GitHub Actions run or commit status through the available connector.
- Python threads and third-party provider calls cannot be forcibly terminated safely after they begin; bounded admission and transport deadlines limit impact.
- Final-path robots policy is checked before crawler indexing and expansion, but only after the HTTP redirect response has been fetched.
- Application-level SSRF controls still require deployment network egress policy as defense in depth.
- A privileged host can mutate process storage or memory; filesystem anchoring prevents path redirection but is not a substitute for host isolation, read-only evidence stores, or encryption at rest.
- Parser checks are not malware scanning or sandboxing.
- OCR, PDF reading order, formula/table interpretation, scanned-caption localization, and multi-panel figure semantics remain heuristic.
- Citation provenance remains structural and does not prove semantic entailment.

## Verification boundary

CI configuration still defines Python 3.10–3.12 compilation, fatal Ruff checks, pytest with branch coverage, and a Docker build. No passing result may be inferred from test source, static inspection, commit creation, or the existence of workflow configuration. The pull request must remain draft until executable checks run against the exact release head and every failure is corrected.
