# Exact-head verification repairs — 2026-08-06

This record captures defects exposed only after the exact-head workflow was repaired enough to register real jobs.

## Workflow registration

The release matrix previously referenced the `runner` context from job-level environment expressions, where that context is unavailable. Runtime paths now enter `GITHUB_ENV` from runner steps, and a registration-smoke job makes future zero-job startup failures visible.

## Custody signer lint/correctness

A misspelled local variable in the Ed25519 custody signer-key record contract (`redired_actor`) was corrected to `retired_actor`. This restores both fatal-name lint and retired-key record construction.

## Release lock authority

`pip-tools` recorded its invocation, including the explicitly selected public PyPI resolver URL, in the generated comment header. The lock generator now passes `--no-header`; resolved packages and required hashes remain unchanged while generated lock files no longer embed package-index authority text.

## Windows classic storage authority

The Windows pathname fallback correctly detected swapped roots and parents, but its broad member-I/O exception handler converted the resulting root-integrity `OSError` into a normal missing/corrupt-member result. The handler now revalidates the bound root: authority failures propagate, while ordinary member-level I/O failures retain conservative quarantine behavior.

## Verification rule

These repairs do not establish release readiness by themselves. Release readiness requires the complete Linux, Windows, container and cross-platform hashed-lock matrix to pass on one unchanged `main` SHA.
