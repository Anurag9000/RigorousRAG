from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch anchor missing in {path}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    ".github/workflows/release-locks.yml",
    '    branches: [main, "agent/**"]\n',
    "    branches: [main]\n",
)

replace_once(
    "tests/unit/test_deployment_parity.py",
    '''    assert "  pull_request:\\n" in workflow\n    assert "  merge_group:\\n" in workflow\n    assert "paths:" not in workflow.split("permissions:", 1)[0]\n''',
    '''    assert "  pull_request:\\n" in workflow\n    assert "  merge_group:\\n" in workflow\n    assert "    branches: [main]\\n" in workflow\n    assert "agent/**" not in workflow\n    assert "paths:" not in workflow.split("permissions:", 1)[0]\n''',
)

Path("docs/CONTINUATION_AUDIT_PASS13_TO_15.md").write_text(
    '''# RigorousRAG continuation audit — Passes 13–15

Date: 2026-08-01  
Branch: `main`  
Development policy: direct commits to `main`; no feature branches or pull requests.

## Scope

These passes audited compatibility layers that replace or wrap legacy modules, exact
numeric boundaries in public constructors, stateful singleton/class identity, and retained
storage identity. The central defect class was reimport drift: a wrapper could capture a
previous wrapper as its base/original function, stack validation layers recursively, reset
a singleton cache, or replace a live state object.

Every source change below was published only after its focused GitHub Actions suite
compiled the affected modules and passed.

## Pass 13 — security compatibility-layer idempotency

The strict security boundary now:

- stores the unwrapped upload-suffix and public-URL functions once on the implementation
  module instead of recapturing patched wrappers on reimport;
- remains nonrecursive across repeated imports;
- contains exceptions raised by hostile domain iterators and hostile header mappings;
- converts malformed request-header containers to a generic security error without
  leaking private exception text;
- drops oversized response-header names and values instead of truncating names into
  collision-prone aliases;
- retains canonical authentication, URL, domain, upload-name, exact-limit, peer-validation,
  redirect, and provider-response contracts introduced in pass twelve.

The focused security/network suite passed before commit
`45fb4c26d52675539156d4d9c0a841914fdcc93c` was published.

## Pass 14 — scientific and RAG compatibility chains

Scientific-integrity layers now preserve two explicit call chains:

1. the legacy scientific functions beneath `integrity_boundary`;
2. the first strict boundary functions beneath the final `integrity` layer.

Repeated imports no longer capture the final wrapper as the next layer's original
implementation. The visual-entailment monkeypatch seam remains public and testable.

The RAG boundary now:

- preserves the original legacy `RAGLayer` base across reimports;
- preserves the singleton instance dictionary and lock;
- rejects fractional, floating, `Decimal`, and `Fraction` pseudo-integers through the
  integer index protocol;
- accepts explicit objects implementing `__index__`;
- rejects POSIX links and Windows reparse points in every existing Chroma path component.

After two workflow-only test corrections, the focused integrity/RAG suite passed all 56
selected tests before commit `d6970f2598cfee4668e57d9a0262d26160e1f9b3` was published. Temporary failure diagnostics
were removed from the successful head.

## Pass 15 — stateful public class and root identity preservation

Classic storage, the retained-document registry, RAG, and the research agent now preserve
one public wrapper class per process as well as one legacy base class. Reimporting these
modules therefore does not deepen method-resolution chains, reset public class identity,
or invalidate singleton type checks.

Additional retained-document registry controls:

- database parent, database file, and upload root are bound to initial device/inode
  identities;
- ordinary directory/file replacement is detected even when no symbolic link is used;
- POSIX links and Windows reparse points fail closed;
- cached stores revalidate identities before reuse;
- boolean orphan-cleanup clocks are rejected.

Additional search-agent controls:

- `SearchAgent`, `ToolExecution`, and schema-validator originals are preserved once;
- public class identities remain stable across repeated imports;
- turn, tool-call, and token counts require the integer index protocol;
- boolean timeouts are rejected;
- model, API-key, and provider fields reject every ASCII control and DEL.

The first runner stopped during discovery because it named a nonexistent historical test
file. After correcting only that runner path, the complete selected storage, registry,
agent, RAG, and compatibility suite passed before commit
`3130208e7957fda91e1e480672d02ef332778237` was published. The temporary diagnostic and
runner were removed.

## Current verification boundary

Observed focused execution now covers passes 10, 11, 13, 14, and 15. The pass-thirteen
suite also executed the pass-twelve security-boundary regressions. These runs establish the
listed component contracts but are not a full release certificate.

The authoritative workflow must still succeed on one unchanged final `main` head across:

- Linux Python 3.10–3.12 complete dependency, compilation, fatal Ruff, pytest, and branch
  coverage gates;
- Windows Python 3.10 and 3.12 classic-storage gates;
- Docker Compose validation and container build;
- Linux, Windows, and macOS Python 3.10–3.12 release-lock generation, verification,
  hash-required dry installation, and artifact publication.

`docs/LATEST_EXACT_HEAD_RESULT.md` is authoritative only when generated by the freshness-
bound reporter for the still-current `main` SHA.

## Residual non-claims

These passes do not convert process-local SQLite, locks, executors, schedulers, rate
limits, or compensating vector writes into distributed exactly-once infrastructure. They
do not provide host isolation, encryption at rest, malware sandboxing, certified
de-identification, semantic entailment proof, or scientific replication. OCR, reading
order, formula/table extraction, captions, and multi-panel interpretation remain
heuristic and require source inspection and expert review.
''',
    encoding="utf-8",
)

Path("docs/REMEDIATION_STATUS.md").write_text(
    '''# Exhaustive remediation status

This is the authoritative status index for the RigorousRAG repository-wide remediation
started on 2026-07-27 and continued through **fifteen** audit/regression passes.

## Repository state

- Default and only branch: `main`
- Open pull requests: none
- Development policy: coherent commits directly to `main`; no feature branches or PRs
- Authoritative verification workflow: `.github/workflows/release-locks.yml`
- Configured gate: 16 jobs
- Freshness-bound result reporter: `.github/workflows/exact-head-report.yml`
- Current final-head certificate: not established unless
  `docs/LATEST_EXACT_HEAD_RESULT.md` exists for the exact current `main` SHA with a
  `success` conclusion

Prior PRs #1–#4 are preserved only as closed/merged history. All previous implementation
commits and every continuation pass are contained in `main`.

## Implemented product and safety surfaces

The current source covers:

- classic crawling, lexical indexing, PageRank, immutable generation persistence, and
  internal search;
- PDF/DOCX/text ingestion, bounded OCR, semantic sections, stable owner-scoped document
  identity, retained sources, visual evidence, and vector retrieval;
- durable jobs, retries, scheduling, startup recovery, registry lifecycle, and
  fingerprint-bound corrupt-row operator recovery;
- scholarly, web, page, handbook, uploaded-document, comparison, limitation, debate,
  protocol, conflict, figure, and BibTeX tools;
- request-scoped agent orchestration, bounded tool execution, authoritative citations,
  provider/network controls, and privacy-safe telemetry;
- FastAPI authentication, uploads, request framing/body limits, throttling, deadlines,
  readiness, frontend lifecycle, container deployment, release locks, and test contracts.

## High-value enforced contracts

| Boundary | Current contract |
|---|---|
| Tenant identity | Server-owned API-key mapping or configured single-user identity controls vector, registry, scientific, and lifecycle operations. Caller owner headers cannot select another tenant. |
| Request/work admission | HTTP framing, bodies, identifiers, models, running-plus-pending work, deadlines, tool calls, evidence, citations, metadata, warnings, and responses are bounded. |
| Upload/retained files | Random owner-scoped names, exact byte limits, descriptor-relative POSIX operations, root/file identity binding, private modes, `fsync`, and symbolic-link/reparse refusal. |
| Durable ingestion | SQLite state machine, atomic claims, bounded attempts, durable backoff, keyed scheduling, startup replay, immutable parser snapshots, source re-hashing, and compensating vector restoration. |
| Corrupt rows | Normal reads fail closed. Operator scans are sanitized, bounded, keyset-paginated, high-water-marked, fingerprint-bound, and never implicitly delete retained sources/vectors/registry records. |
| Parsing/OCR | PDF page/text/pixel ceilings, DOCX member/expansion/ratio ceilings, text limits, bounded OCR and rendering, strict models, and non-text-control refusal. |
| Privacy | Best-effort masking covers text, OCR, filenames, metadata, summaries, jobs, scientific objects, CLI/telemetry output, paths, credentials, contact data, IPs, and non-finite values. |
| Retrieval/provenance | Uploaded results require canonical owner/document/chunk metadata and exact page provenance; malformed, cross-owner, incomplete, or non-finite rows are discarded. |
| Network/provider | Canonical credentials/configuration, duplicate-key rejection, public DNS plus connected-peer validation, redirect revalidation, proxy suppression, cross-origin secret stripping, strict headers/MIME/body/deadline limits, and strict provider JSON. |
| Stateful compatibility | Security, scientific, RAG, classic storage, document registry, and search-agent wrappers preserve original/public identities across reimports instead of stacking wrappers or resetting singleton state. |
| Browser/deployment | No untrusted `innerHTML` or third-party runtime assets; session-only credentials/history; lexical asset identity; non-root read-only container; dropped capabilities; named volumes; loopback publishing by default. |
| Release reproducibility | Immutable requirements snapshots, public-PyPI authority, ambient resolver-authority removal, exact pins/hashes, no-follow identity-stable verification, atomic publication, and immutable action pins. |

## Observed execution

Focused successful workflow execution is recorded for:

- pass 10: operator repair, service paths, trusted sources;
- pass 11: telemetry publication and rotation;
- pass 13: strict security/network boundary, including pass-twelve regressions;
- pass 14: scientific-integrity and RAG compatibility/exact-input suites;
- pass 15: classic storage, document registry, search agent, RAG, and stateful reimport
  suites.

Historical full-suite evidence remains nonfinal: an earlier Linux Python 3.12 run passed
711/713 tests with 76.25% branch coverage and exposed the subsequently corrected
sentence-final email-masking defect. Two historical nine-platform lock matrices passed.

## Required final gate

One unchanged final `main` SHA must complete all 16 authoritative jobs. Every failure must
be corrected directly on `main`, followed by a complete rerun. A success record for an
older SHA, another branch, pull request, or merge queue is rejected by the reporter and is
not a release certificate.

## Residual architectural and scientific limitations

- Redirect-target robots policy cannot prevent the redirect response itself from being
  fetched.
- Python thread work already executing cannot be forcibly killed safely.
- Application SSRF controls still require deployment DNS and egress policy.
- Filesystem identity checks are not host isolation or encryption at rest.
- Parser limits are not malware scanning or sandboxing.
- Process-local admission, scheduling, rate limiting, SQLite, locks, and vector
  compensation are not distributed exactly-once infrastructure.
- OCR, reading order, formulas, tables, scanned captions, and multi-panel interpretation
  remain heuristic.
- Regex masking is not certified de-identification.
- Readiness does not prove model availability or representative semantic retrieval.
- Structural provenance does not prove semantic support or scientific truth; source
  inspection, expert assessment, and replication remain required.

Detailed records are in `CONTINUATION_AUDIT.md`, passes 2–9, passes 10–12, and passes
13–15.
''',
    encoding="utf-8",
)

Path("docs/EXECUTABLE_VERIFICATION.md").write_text(
    '''# Executable verification ledger

This ledger records observed execution separately from committed source/test contracts.
A focused successful suite certifies only its selected components. A complete release
certificate requires all authoritative jobs to succeed for the exact current `main` SHA.

## Historical broad evidence

### Release-lock matrices

- Run `30547701731`: all nine Linux/Windows/macOS, Python 3.10–3.12 lock jobs passed.
- Run `30603463220`: all nine platform/Python lock jobs passed again.

Each passed lock generation, verification, hash-required dry installation, and artifact
publication. Later generator/verifier hardening means these are historical evidence, not a
certificate for current `main`.

### First complete Linux Python 3.12 suite

A superseded run:

- passed dependency installation, `pip check`, compilation, and fatal Ruff checks;
- collected 713 tests;
- passed 711 and failed 2;
- measured 76.25% branch coverage, above the configured 50% floor.

Both failures were the same sentence-final email masking defect in OCR and semantic
sections. The shared privacy primitive and direct punctuation regressions were corrected
after that run.

## Observed focused continuation execution

| Pass | Published commit | Observed gate |
|---|---|---|
| 10 | `8d81a1a9778f5a1224517ad5bcfa7956596e9f9e` | Operator repair, server path configuration, and trusted-source suites compiled and passed. |
| 11 | `522ed5eb9e709a2cb8f4093d7cb083bdaa607bfc` | All 22 selected telemetry identity, append, lock, and rotation tests passed. |
| 13 | `45fb4c26d52675539156d4d9c0a841914fdcc93c` | Security-boundary activation, reload, authentication, URL/domain/header, safe-download, peer, redirect, and response tests passed. |
| 14 | `d6970f2598cfee4668e57d9a0262d26160e1f9b3` | All 56 selected scientific-integrity, RAG, reimport, exact-integer, and reparse tests passed. |
| 15 | `3130208e7957fda91e1e480672d02ef332778237` | Selected classic storage, registry, search-agent, RAG, and stateful compatibility suites passed after correcting one stale workflow test path. |

Pass twelve's strict security contracts were exercised by the pass-thirteen suite.
Temporary one-shot workflows, patch scripts, and failure diagnostics were removed from
each successful published head.

## Authoritative exact-head workflow

`.github/workflows/release-locks.yml`, named `Exact-head verification and release locks`,
contains 16 jobs:

- exact-checkout registration smoke;
- Linux Python 3.10–3.12 dependency consistency, whitespace, compilation, fatal Ruff,
  complete pytest, and measured branch coverage;
- Windows Python 3.10 and 3.12 classic-storage compilation and regressions;
- Docker Compose validation and container build;
- Linux/Windows/macOS Python 3.10–3.12 release-lock generation, verification,
  hash-required dry installation, and artifact publication.

It runs for `main`, version tags, manual dispatch, pull requests, and merge queues. The
repository development policy nevertheless uses direct commits to its only branch,
`main`.

`.github/workflows/exact-head-report.yml` publishes
`docs/LATEST_EXACT_HEAD_RESULT.md` only when:

1. the completed run's branch is `main`;
2. its SHA equals the checked-out current `main` SHA;
3. `origin/main` still equals that SHA immediately before publication.

Stale, branch, PR, and merge-queue results cannot overwrite the ledger.

## Current release boundary

No current-head success is claimed unless `docs/LATEST_EXACT_HEAD_RESULT.md` exists and
records `success` for the exact current `main` commit. The available local execution
container cannot clone GitHub because DNS resolution for `github.com` fails; GitHub
Actions is therefore the executable source of truth.

Every authoritative failure remains blocking and must be corrected directly on `main`,
then rerun across the complete matrix.
''',
    encoding="utf-8",
)
