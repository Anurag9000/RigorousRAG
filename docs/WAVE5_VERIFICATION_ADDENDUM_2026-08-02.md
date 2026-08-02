# Wave 5 verification addendum — 2026-08-02

This addendum supersedes earlier focused-test counts in `docs/WAVE5_IMPLEMENTATION_STATUS_2026-08-02.md`. It records the additional resumable-run and historical-baseline governance slices and the newest exact-current Linux verification boundary.

## Additional completed capabilities

### Resumable live benchmark runs

Implemented:

- [x] Append-only completed-run storage keyed by exact plan fingerprint and run ID.
- [x] Separate benchmark and execution-plan fingerprints.
- [x] Per-run contract, result, report and stored-run digests.
- [x] True interruption recovery: completed seeds are reused without resolving their queries or invoking their selectors.
- [x] Selector-configuration isolation without losing benchmark comparability.
- [x] Text-free SQLite payloads containing only aggregate metrics, counts and digests.
- [x] Row/payload/database identity and symlink/reparse defenses.
- [x] Exact-confirmation plan cleanup.
- [x] Privacy-safe status/removal CLI.

### Governed historical baselines

Implemented:

- [x] Append-only benchmark baseline records.
- [x] One current pointer per benchmark fingerprint and regression policy ID.
- [x] Explicit no-current requirement for first activation.
- [x] Exact current-baseline expectation for replacement.
- [x] Eligible-regression-only replacement.
- [x] Exact baseline-report, candidate-report, benchmark, policy-ID and policy-digest binding.
- [x] Blocked, forged, stale-current and policy-mismatched replacement refusal.
- [x] Previous-baseline and activation-regression lineage.
- [x] Strict nested report reconstruction and digest verification.
- [x] Independent pointers for distinct policy IDs.
- [x] Privacy-safe initialize/promote/status/history CLI.
- [x] No runtime selector or serving-policy mutation.

## Exact-current focused verification

A fresh archive was downloaded from the live `main` after the baseline-registry commits. On that one unchanged archive:

- **114/114 evidence-graph focused tests passed**;
- the repository’s complete pytest suite passed;
- `python -m pip check` passed;
- whole-tree `python -m compileall -q .` passed.

The 114 focused contracts include:

- graph types, builders, storage, retrieval and analysis;
- derived graph jobs, reconciliation, authority and operations;
- cross-document graph sets and read-time authority;
- reviewed relation proposals, decisions and supersession;
- compensating graph-set publication;
- bounded authoritative GraphRAG selection;
- per-case and macro evaluation;
- strict benchmark fixtures and atomic report CLI;
- historical non-inferiority regression gates;
- live authoritative benchmark execution and immediate text reduction;
- resumable completed-run storage;
- governed historical baseline activation and replacement.

## Repository process state

- Development remains direct to `main`.
- No feature branch or pull request was created for this work.
- No force push or history rewrite was performed.
- Historical pull-request records remain separate from the active main-only workflow.

## Verification still open

The following are not claimed green by this addendum:

- Ruff and the repository’s complete configured lint policy;
- Windows Python 3.10/3.12 compatibility;
- Docker Compose validation;
- Docker image build and readiness smoke tests;
- connected web, scholarly, embedding and model-provider tests;
- multi-process/distributed graph publication and baseline-pointer coordination;
- filesystem, SQLite, process-crash and disk-failure injection across every new phase;
- agent/API/browser citation publication for GraphRAG;
- one final unchanged-head line-by-line release audit with all platform jobs green.

## Remaining Wave 5 implementation priorities

1. Mechanically verified conversion from selected graph evidence into the repository’s existing server-owned citation/evidence registry.
2. Agent tool registration only after citation conversion, abstention and deduplication contracts pass.
3. API/browser propagation and safe-DOM tests.
4. Durable graph-set publication-attempt journaling and crash recovery.
5. Multi-process leadership or database-scoped leases.
6. Reviewer authorization and separation of duties.
7. Measured latency, memory, backend I/O and monetary/resource accounting.
8. Dataset cards, annotation guidance, checksums, versions, splits and licenses.
9. Bootstrap/permutation inference and multiple-comparison controls.
10. Baseline/run archival export, legal hold, retention, backup and restore.

## Permanent non-claims

- Evidence-graph provenance is not scientific truth.
- Reviewer approval is not proof of entailment.
- A selected graph path is not causal proof.
- Retrieval and regression metrics are only as valid as their labels, fixtures and statistical assumptions.
- A governed historical baseline does not change runtime behavior.
- Passing Linux tests is not the complete cross-platform release matrix.
- Release readiness is not claimed.
