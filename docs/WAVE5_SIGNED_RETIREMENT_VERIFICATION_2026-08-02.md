# Wave 5 signed retirement verification boundary

Last updated: 2026-08-02

This ledger records the exact execution boundary for the signed publication retirement, operations, snapshot and restore-preflight work.

## Live repository audit before this ledger commit

- audited code/document head: `989f95c79cd32226a65a5299be59f0f5239cc8b4`;
- only branch returned: `main`;
- open pull requests returned: none;
- combined status checks returned for the audited head: none;
- workflow runs returned for the audited head: none.

Absence of returned checks is not evidence that CI passed.

## Executed reconstructed retirement-core evidence

The reconstructed workspace contains the committed retirement contracts, SQLite journal, exact weaker-publication lease mutation, recovery executor, failure-normalizing boundary and isolated runtime. Unrelated older repository services are represented by API-faithful stubs.

Executed command family:

```text
python -m compileall -q tools tests
python -m pytest -q
```

Result:

```text
12 passed
```

The executed checks cover:

1. deterministic retirement identity and scope binding;
2. journal lifecycle, lease claim and terminal completion;
3. exact weaker lease takeover without weaker retry-count inflation;
4. normal signed-pointer restoration and weaker retirement;
5. recovery after signed-pointer commit before phase persistence;
6. recovery after weaker cancellation before phase persistence;
7. post-intent external-pointer preservation;
8. pre-intent external-pointer refusal;
9. post-claim raw failure normalization;
10. isolated third-journal path selection;
11. late signed-authority drift after weaker cancellation;
12. weaker saga-lease renewal before cancellation.

Focused compilation passed for the reconstructed module tree.

This is not an exact-current complete repository checkout.

## Committed repository-native retirement contracts

### Saga contracts: 38

- retirement identity and journal: 8;
- exact weaker lease mutation: 4;
- third-journal runtime isolation: 4;
- recovery executor and crash matrix: 10;
- seed/preflight binding: 2;
- failure-normalizing boundary: 3;
- operator CLI: 5;
- late-phase authority/lease faults: 2.

### Operational audit and retention contracts: 5

- complete state/lease classification;
- bounded and duplicate refusal;
- latest-terminal/legal-hold/default-completed protection;
- old terminal duplicate candidates;
- text-free non-mutating CLI.

### Snapshot contracts: 8

- deterministic construction and owner scope;
- atomic no-overwrite export and round trip;
- checksum and duplicate-key refusal;
- path redirection and bounded-result refusal;
- verify without live journal;
- descriptor-safe valid verification;
- growth-during-read refusal;
- path replacement after descriptor acquisition.

### Restore-preflight contracts: 5

- empty initialized target eligibility;
- exact terminal idempotence and nonterminal refusal;
- state collision, partial and additional history refusal;
- initialized read-only target and query-only write refusal;
- byte-preserving non-mutating CLI.

Total newly committed retirement-family contracts represented above: **56**.

These 56 repository-native contracts have not been executed together against a fresh exact-current checkout.

## Earlier focused evidence retained separately

Earlier reconstructed evidence remains tied to its recorded code boundaries:

- 12/12 signed actor assertion/binding/use checks;
- 33/33 signed publication, journal isolation, transition and retirement-preflight checks;
- the historical unchanged archive with 114/114 evidence-graph checks and a full repository pytest pass predates the newest governance, agent, retirement, snapshot and restore commits.

Those results must not be promoted to exact-current evidence.

## Newly implemented boundaries

- isolated third retirement database;
- durable retirement phases and leases;
- exact weaker-publication lease takeover;
- compare-and-swap signed-pointer restoration;
- crash recovery without weaker compensation;
- read-only retirement operational audit;
- conservative no-delete retention planning;
- deterministic text-free snapshot export;
- descriptor-safe offline snapshot verification;
- SQLite read-only target view;
- read-only snapshot restore preflight;
- current Wave 5 backlog superseding stale historical checkboxes.

## Exact-current verification still required

- complete repository pytest;
- coverage;
- Ruff;
- full-tree compilation;
- live agent, FastAPI and frontend integration;
- Docker/Compose persistence and restart;
- Windows path, permissions and reparse points;
- independent-process lease and pointer contention;
- process-kill injection at every retirement phase;
- SQLite busy/locked, WAL, I/O and disk-full injection;
- snapshot export interruption and directory-fsync failures;
- restore-target concurrent mutation tests;
- backup/restore disaster-recovery exercise.

## Non-claims

- No exact-current CI success is claimed.
- No full exact-current pytest success is claimed.
- Snapshot integrity is not a digital signature.
- Restore-preflight eligibility is not restore authorization.
- Retention candidates are not deletion authorization.
- Signed retirement does not relabel weaker graph sets as signed provenance.
- Release readiness is not claimed.
