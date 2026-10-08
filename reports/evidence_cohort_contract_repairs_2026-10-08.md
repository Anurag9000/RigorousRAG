# Evidence, calibration and dataset-cohort contract repair — 2026-10-08

## Independent audit scope

This report covers actionable errors in the RigorousRAG public exact-head
verification workflow, run 37633762189, and dedicated source-contract
workflows. It is a **focused runtime correctness repair**, not a full
repository/estate A–D closure certificate or a simulated training result.

At the initiating exact-head run, Python 3.11 reported 32 failed and 2,565
passed tests. Failing groups included content-addressed evidence receipts,
retrieved-content trust decisions, calibration drift, region observability,
ListNet ranking expectations, pinned cohort source identity and inaccessible
OPF private source. The staged controller static/audit workflow also
reported independently unaccounted runtime/learner/registry surfaces.

## Implemented on main

### 1. Immutable evidence and retrieved-content trust receipts

- `tools/evidence_context_packing.py` now constructs its runtime receipt
  with `PackedEvidence` instances rather than using JSON dictionaries
  from the hash payload as dataclass records. `ContextPackingReceipt`
  explicitly rejects untyped dictionary rows before consuming `.order`.
  Hash construction still includes the canonical v1 schema marker.
- `security/retrieved_content_trust.py` omits the hash-only schema
  marker from the `RetrievedContentTrustDecision` constructor without
  dropping it from the signed/content-addressed decision payload.
- `tests/unit/test_evidence_context_packing.py` now checks typed rows
  and rejects a tampered dictionary-based replacement receipt.
- These changes repair an actual runtime boundary used by prompt-injection
  review/quarantine and last-mile evidence materialization; they do **not**
  authorize retrieved content to become instructions.

### 2. Calibration drift and region publication

- Both `CalibrationDriftReference` and `CalibrationDriftDecision`
  constructors now receive only declared dataclass fields, while their
  content hashes retain `schema` domain separation.
- The region publication observability fixture similarly separates its
  canonical receipt payload from the runtime constructor parameters.
- Calibration qualification, population shift/PSI, Jensen–Shannon
  divergence, fail-closed RRF-only decisions, lineage matching, and
  observed region-route metrics remain unchanged.

### 3. Query-grouped ListNet training correctness

- The failing test incorrectly assumed the most rank-informative model
  must receive the largest non-negative mixture weight. The actual
  softmax-target ListNet cross-entropy can properly downweight a
  highly overconfident source that ranks examples correctly.
- The replacement test checks the intended *held-out ListNet objective*
  improves over equal-weight initialization and retains ranking direction.
  No learned weights, seed, training batches or optimizer behavior were
  overwritten to satisfy an unsupported assertion.

### 4. Versioned dataset cohort Git hash integrity

- The historic `training/dataset_cohort_runtime_v4.py` used a **literal
  backslash followed by zero** when constructing a Git blob header,
  rather than the required NUL byte. This caused its preloaded-base
  verifier to reject legitimate base source as tampered. Neither the
  original v4 source nor its cache/pin was rewritten.
- New versioned
  `training/dataset_cohort_runtime_v5.py` (blob
  `3a6c4d9573e570a15232bbaea141e449c67ce5ba`,
  commit `120f6fd3e934f5febf3e78c4f3c8d1ca43bf05d3`)
  corrects that Git header while retaining the v2 cohort state schema.
- `tools/dataset_cohort_runtime_entry_v5.py` pins exact v5 source and
  the preexisting base commit/blob, verifies imported cached files and
  uses an independent v5 module name. The canonical
  `tools/dataset_cohort_runtime_entry.py` selects v5.
- Historical v1–v4 entries remain separately addressable. Versioned
  regression tests check their release identity, v5 offline materialization,
  exact source-byte verification, tampered-preload rejection and the
  known historical v4 failure. No old run is relabeled v5.

## Executed tests and truthful evidence

| Run | Scope | Outcome |
| --- | --- | --- |
| [37755768067](https://github.com/Anurag9000/RigorousRAG/actions/runs/37755768067) | Historical cohort pins + v5 exact Git blob/loader tests | **12 passed** |
| [37756072506](https://github.com/Anurag9000/RigorousRAG/actions/runs/37756072506) | Evidence packing/materialization, injection review, drift, ranking and observability | **36 passed** |

Earlier focused runs surfaced actual issues and were fixed rather than
reclassified as passing: the v5 filename/module-alias mismatch, missing
minimal test dependencies, and the hash-only schema passed into the
trust-decision constructor.

## Deliberate exclusions and remaining failures

- The broad Linux exact-head test suite is **not yet certified green**.
  Its older run had the remaining private OPF source 404s and dataset-cohort
  historical pin assumptions; corrected source is now tracked by focused
  tests, but a full fresh cross-platform suite must be assessed separately.
- The RigorousRAG training-control static/census audit reports outstanding
  unreachable and unaccounted first-party surfaces. These are not waived
  or hidden by a focused test pass.
- Private OPF_ADP cross-repository bootstrap requires correctly authorized
  private-source access. A 404 is **not** evidence that the scheduler
  passed or was optional.
- This report does not claim real CUDA training, exact-resume recovery
  under OOM, numerical parity, complete external datasets, or governance
  closure of the entire repository.

**Current classification:** repaired focused control paths are supported
by executable evidence; full repository and account-wide closure remain
OPEN.
