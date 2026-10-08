# Private OPF reference testing boundary — 2026-10-08

## Observed evidence (not inferred)

The public RigorousRAG exact-head Linux job at commit
`8e9258d5e8d0698cec2678e0f373b4fee6b5c099`, workflow run
`37756254926`, executed 2,596 passing tests on Python 3.11 and failed
six literal OPF scheduler tests with `urllib.error.HTTPError: HTTP Error
404: Not Found`. These tests called
`base._prepare_opf_runtime(tmp_path)`, which retrieves byte-pinned
source from `Anurag9000/OPF_ADP`. That source repository is private,
so its `raw.githubusercontent.com` URLs cannot be read by an
unauthenticated public runner. The 404 was not a behavioral counterexample
for the scheduler.

**A second, independent failure remains:** total reported branch coverage
was 49.04%, below the required 50%. This document and fixture change do
not lower the threshold, omit code from coverage or certify the remainder.

## Behavior of the amended test boundary

`tests/test_training_controller_literal_opf_scheduler_behavior.py`
now supports a deliberately explicit private source input:
`OPF_LITERAL_VERIFIED_CACHE=/absolute/path/to/private/opf-source`.
Every runtime source path is required to be a regular file within that
trusted root and to match the exact `OPF_RUNTIME_BLOBS` Git blob hash
before Python imports the scheduler. A missing, symlinked, escaped, or
tampered file fails; no fallback to the active workspace version or
different OPF pin is allowed.

If a private URL returns **404** and no verified cache is supplied,
the six literal behavioral tests are **skipped as unverified**, with an
explicit reason. A server error, unexpected HTTP status, bad blob or
invalid supplied cache remains a hard failure. An authorized CI runner
may set `OPF_LITERAL_REQUIRE_PRIVATE=1` to treat even private-source
404 as a hard failure. Do not interpret skipped tests as passes, and do
not copy private OPF source into the public RigorousRAG tree or CI logs.

`tests/test_training_controller_literal_opf_private_cache_contract.py`
regresses all boundaries: expected unauthenticated 404, opt-in strict
failure, other HTTP errors, valid/tampered blob, and a symlink.

## Correct execution workflow

For a runner with authorized OPF_ADP source access:

1. Check out RigorousRAG at its exact desired commit.
2. Materialize the **literal pinned OPF commit**, not a floating `main`,
   inside a secure/private runner directory.
3. Verify the manifest and every blob against the literal active
   `tools/universal_training_controller_opf_reference_v2.py`.
4. Execute:
   ```bash
   OPF_LITERAL_VERIFIED_CACHE=/secure/pinned-opf-tree \
   OPF_LITERAL_REQUIRE_PRIVATE=1 \
     python -m pytest -q -o addopts='' \
       tests/test_training_controller_literal_opf_scheduler_behavior.py \
       tests/test_training_controller_literal_opf_private_cache_contract.py
   ```
5. Run the full matrix including native OPF tests and physical CUDA
   admission/recovery scenarios separately, retaining logs and evidence.

The public source-contract test suite can verify pin and cache integrity
without possessing the private bytes, but it cannot execute their
scheduler implementation. The strict account-wide training audit also
remains incomplete for separate source/registry/config closure reasons.
No release, physical-GPU, coverage or private-runtime certification is
claimed here.
