# CUDA-first account audit — source and pin verification

Snapshot date: 2026-09-27. Scope: 44 connected repositories under Anurag9000, inspecting live `main` root launcher content and the shared controller's Git-blob manifest.

**Truth boundary:** This is an account-wide launcher/pin inventory, not a complete per-function execution audit or a CPU/GPU parity certificate. Repository-specific scientific/job completeness and hardware execution remain independently unverified unless explicitly stated.

## Verified shared-controller chain

- 38 root launchers reference RigorousRAG commit `72d37dd0ce06123153e10ee660141df5c6a46ef6` **and** validate its entrypoint Git blob `fb60c41db6ab3922166aab6a976fe3221c939ef8`.
- The v38 source bundle pins OPF_ADP commit `1d3dea055736b1261a621436626ce89d69242a14`; `utils/ml_backends.py` is pinned as Git blob `33108a3e20e982188ebc089399c682b11f202c4c`.
- Pinned scheduler source identity is not proof of execution on CUDA. CUDA-capable workloads must additionally prove usable framework CUDA runtime, selected child device, GPU library activation, numerical output and fallback behavior.

## Repository matrix

| Repository | Root controller | This audit's implementation finding |
|---|---|---|
| Text-and-Emotion-Analysis-Tool-with-Visualization | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| CO-project | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Resume | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Making-LLMs-fill-reimbursement-form | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Companies-institutes-and-Internships | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| OPF_ADP | Backend/scheduler source repository | Targeted GPU/CPU source fix committed; runtime verification still open |
| dragonball-chess | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| ERP_College | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Breaking-the-Neural-Barrier | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Gram-Connect | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Traffic | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| ERP_Web | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Signal-Prophet | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| RigorousRAG | Shared-controller source repository | GPU/CPU feature-level coverage not yet certified |
| modular-grokking-adp | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| hallucination-resistant-llm-framework | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| NutriFlavorOS | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open; synthetic trainer vs no-trainable runner inconsistency |
| PlaceMate-AI | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| HydroGraph-Delhi | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified; default branch remains `master` |
| Silicon-Pilot | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| existential-coordination-games | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Forest_Run | Separate source-classification authority; no retained ML training | No retained trainable surface per repository-owned authority; numerical simulation applicability separately assessed |
| ESD_Project | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| End-to-End-Digital-Communication-System | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Smart-Glasses | Verified v38 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Continual-Learning | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| VaaniNoise-SED | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| VaaniEventClean | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| TutorMistakeLens | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| TutorGuidance-Eval | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| CatalogPathForge | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| ReceiptTupleForge | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| MASSIVE-IntentForge | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndoIntent-150 | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| DocVision | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndoDocFusion | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| HinglishKnowledge-NLI | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndicRelationForge | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| STM32n6AI | Separate native central-runner authority | Targeted GPU/CPU source fix committed; runtime verification still open |
| continual-learning-with-rl | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Railguard-AI | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| False-News-Interpretability | Verified v38 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Last-War | No root training launcher | Requires independently reconstructed trainable/simulation applicability |
| Project-Foundry | Empty repository | Not applicable until repository contains source |

## Unclosed technical requirements

1. Pin and checksum propagation is source-verified; bootstrap download/cache, runtime import and scheduler parity need executable tests on each supported deployment environment.
2. For every repository, trace workload registry → scheduler admission → child environment → model/tensors/optimizer → preprocessing/augmentation/metrics → output/checkpoints. Record actual devices and operations, not mere GPU availability.
3. Probe usable CUDA memory allocation independently for PyTorch, CuPy and applicable JAX/TensorFlow backends. `nvidia-smi` alone is not an execution certificate.
4. Install `cudf.pandas` and `cuml.accel` before pandas/sklearn imports, verify supported coverage and CPU fallbacks, and preserve estimator/metric numerical semantics.
5. CPU-only workers must mask CUDA for all accelerators, not merely set PyTorch device to CPU. Ensure GPU worker isolation maps assigned physical GPU to its local `cuda:0`.
6. Test multi-GPU assignment, missing/CPU-only framework wheel, GPU driver mismatch, insufficient VRAM, GPU OOM recovery, checkpoint resume, deterministic seeds, cross-library RNG isolation, and per-dataset augmentation consistency.
7. Preserve zero-training/CPU-only workloads as such; never manufacture ML job records or claim meaningless CuPy conversion is an acceleration.
8. Run CPU and real CUDA smoke, unit, integration, and end-to-end tests. Current GitHub Actions failures before any job steps do not count as a runtime test.

## Known actionable gaps

- **STM32n6AI:** Native worker now validates central assignment; compact gaze/liveness/audio/L2CS/RT-GENE routes have explicit device propagation. Many other native routes remain unaudited and may still default to CPU.
- **NutriFlavorOS:** Root no-trainable declaration conflicts with retained synthetic/demo PyTorch training scripts; these must be classified and wired, or consciously excluded with evidence.
- **Gram-Connect:** cuML acceleration is optional and estimator-specific; LightGBM CUDA requires a compatible build and must not be reported as GPU without an actual run.
- **Traffic:** Its physics backend and explicit CPU admission were corrected, but end-to-end simulation parity and performance have not been measured.
- **Historic bundles:** Frozen versions remain in history for reproducibility; do not rewrite historical hashes or claim earlier experiments used the updated implementation.

**Closure status: source-pin propagation verified; account-wide complete CUDA-first implementation and execution certification remain OPEN.**

## Follow-up implementation — worker/runtime contracts

Source commits, all direct to `main` (not a CUDA-hardware test certificate):

| Repository | Commit | Verified source-level change | Remaining qualification |
|---|---|---|---|
| STM32n6AI | `696fa2525aa21060efc5f8d133b8ef2827ccb64b` | Native request hash/device assignment and CPU/GPU policy checks | Actual trainer output needed |
| STM32n6AI | `4f1093dbb97fddca526d04655be400f6f63fc226` | Gaze, liveness, audio minibatches/models dispatched to selected Torch device | Other native routes not yet qualified |
| STM32n6AI | `a36dc4f4644e40b10cdce825cd0bf4c8794c2a4c` | GPU-admitted jobs without trainer execution-device evidence fail closed | Device report is not a performance benchmark |
| STM32n6AI | `c5963fc6a3687e73c263b5d578787cd7ba23af4b` | Direct launcher requires explicit admitted placement; failed spawn restores retryable state | Multi-GPU/restart integration still open |
| Smart-Glasses | `67a08bfec50f8f57fa575fd05fa6b786b075496e` | CuPy/cuDF/ONNX/OpenCV optional probes respect forced CPU and CUDA mask | Dataset/model family traces still open |
| End-to-End-Digital-Communication-System | `c8be2f4c7a2b2957b71458a5fdd0ffb52f26d478` | NumPy/CuPy selector honors central CPU policy; invalid modes rejected | Full CUDA BER/sweep parity still open |
| Last-War | `67a324aeabe802d250787af705591f4bd863063c` | Optional batch observation bridge honors CPU admission and probes usable Torch CUDA | C++ simulation itself remains CPU/host-parallel; no native GPU simulation claim |
| NutriFlavorOS | `7c9261a270796b4dc409b465576567a8683cf07a` | Synthetic model script propagates partial failure and no longer labels fake training as production | No-trainable root declaration vs retained synthetic optimizer loops remains unresolved |

The shared v38 source pin remains unchanged; downstream fixes do not silently mutate the historical scheduler or experiment caches. CI run status is not a runtime proof: several project workflows report failure without executing job steps. On supported machines, complete both CPU-only and real CUDA execution tests with exact checkpoint and metric provenance.

**Status remains OPEN:** GPU detection and pin identity do not imply every relevant model or library was accelerated, and no per-repository comprehensive CPU/GPU numerical-parity certificate has been issued.


## 2026-09-28 continuation — v39 main-branch census and corrective commits

This is an incremental source-level follow-up; the original 2026-09-27 v38
snapshot above is retained as historical evidence, not overwritten.

- Of the original 44 repositories, 38 use the byte-pinned shared root launcher.
  A fresh read of the 38 **main-branch** launchers found the current v39
  controller commit `274d9d71663a675359b9ea89d7259995a2821e60` and
  entry blob `b97604e12b0c95294be44652ece0d8ab59942109`.
  This confirms source-pinning, not successful bootstrap or hardware execution.
- `continual-learning-with-rl` was the remaining v38 root pin and was moved
  to v39 at `db0891a13d93188d7e50e35427edae4a2b480a35`, preserving its
  frozen v89 catalog and scientific-authority blobs. Its root loader had
  supplied the **root launcher** as the frozen authority's `__file__`, even
  though that authority computes ROOT with `Path(__file__).parents[1]`.
  The module-location contract was repaired at
  `92b8688a5c9112a26c3f9186036af97ce0179e44` and a CPU-safe
  provenance/profile regression was added at
  `28263208f86731c019007de3d6955a3ee0374fca`.
  Real controller bootstrap, training, and CUDA execution remain unverified.
- `Traffic` recognized its explicit OPF CPU accelerator flag but not the
  independent `CPU_ONLY` and `TRAINING_CONTROL_CPU_ONLY` aliases.
  `0aa859c6b4fa518d8e77dd9e7b5c892be70ab929` now enforces all four
  CPU-admission flags and explicit empty/-1 CUDA visibility masks before
  importing CuPy. `dddc9b2d8d9cec9e183981091060cecec1916287` adds
  fake-usable-CuPy regression cases to the already selected CPU CI test
  module. An isolated CPU policy fixture passed eight cases locally; a
  full-repository CI pass and physical CUDA run have not been established.
- **Branch divergence:** `HydroGraph-Delhi/main` already has the correct
  v39 root launcher (Git blob
  `53f9e93318c9df52cc564f2d0941da8bb93f1d51`), but GitHub still
  reports `master` as the default branch; its master launcher is the older
  blob `25f49babbc7348f65bd0d74c74cf43128ee1b7a1` with controller
  commit `fd34a95d18892df7fb14d1efbb99076a7810fb91`.
  Default-branch migration and preservation/reconciliation of master-only
  history are OPEN; do not count this as an all-default-branch v39 rollout.

The 44-repository actual CUDA execution, per-function backend integration,
CPU/GPU numerical parity, and failure/recovery certification remain OPEN.


### Continuation verification details

- CLRL's original v89 verifier pinned the historical root file at Git blob
  `4af396331e6628cb28d6e44695ac2c0045a1c404`, so the live wrapper
  change also required a source-verifier update. Commit
  `a6697122fc36a4f95bd7b37ec72b9eb1d5f11807` now pins the current
  wrapper blob `20947194224b1de77aeaf0d302856b5411253e8b` and asserts its
  v39 controller identity and corrected module location; all frozen v89
  scientific files retain their historical hashes. Commit
  `d32d8442dbcdf2f4a489535a561d44fe6db557a0` wires the new regression
  into the v89 source CI path and test command.
- For that CI commit, the registry-meta source-contract, general CI and
  main-only workflow runs reported failure **with zero recorded job steps**;
  workflow-job logs were unavailable. This is explicitly not a test failure
  attributable to a particular source line, and not a test pass.
- HydroGraph branch comparison reports `master...main` ahead by six
  commits, behind by zero, with changes in its launcher, estate workflow,
  and dataset-cohort applicability authority. The reverse comparison reports
  no master-only commits. Thus `main` contains the recorded master history;
  the default-branch change remains unperformed and must be verified before
  retiring `master`.


### NutriFlavorOS retained training classification and CPU isolation

Source-level investigation of the current main branch found four explicit synthetic trainer loops in
`scripts/train_all_models.py` and retained optimization in
`backend/ml/online_learning_manager.py`, `backend/ml/meal_planner_rl.py`,
and `backend/ml/taste_predictor.py`. Its root no-trainable classification is therefore
not true as a whole-software statement; the existing scanner remains fail-closed
rather than misrepresenting synthetic runs as production. The full findings and
open device/data contracts are recorded in
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md` at
`077310579616d993de52df1fb2d8616babcb6345`.

The device selector now honors `CPU_ONLY`, `TRAINING_CONTROL_CPU_ONLY`, the
OPF disable flag and the repository-specific override, including stripped explicit
CUDA masks: `43752935641ca0da83ddce8ac8fc9d4d264eac12`.
Torch-dependent regression cases were extended in
`0878bc8d43e0d52c10956b5b5d84a0ef67845a31`.
Since the default backend test requirements do not include Torch, an isolated
fake-Torch regression executable without that optional dependency was added at
`7c3161e667dffe4122ef2273d8164b9625d1893f`.
The existing validation workflow includes the test directory; its most recent
jobs again reported failure with zero recorded steps, so neither a full test pass
nor a physical CUDA run can be inferred.

Outstanding: make retained real training and user-interaction updates first-class
separately classified scientific/operational paths; fix model/input placement and
feature-shape contracts; run actual CPU and CUDA integration tests.

The NutriFlavor root and audit scanner documentation were additionally corrected
in `9207bc4eaf55da57d410ccd75608154945589809` and
`6bd6b4dd942e8cb6a0fb2d9c62d10e6f8685646c`. Their original
no-trainable gate is intentionally still fail-closed: documentation corrections
do not imply that the retained optimizer workloads have been catalogued.


### 2026-09-28 next continuation — NutriFlavor source-level training-path work

The repository-specific audit was extended in
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md`
(commit `9eee84412a0364b9b9c7bfa97ff17e8cb019fb32`).
Online taste and health minibatches now have explicit shapes, shared inference
normalization, selected Torch placement and missing-label masking. The synthetic
PPO script previously ran 20 transitions into a 32-minimum trainer and silently
saved a no-op; the demo now collects 40 and requires explicit update evidence.
These are source fixes and synthetic/CPU-contract evidence, **not** real-data
or CUDA training certificates.

Further inventory identified a distinct retained governed PyTorch research
trainer under `backend/research/training`, with architecture registries,
optimizers, stateful loaders, checkpointing and experimental adapters. Its
auto-device selector originally bypassed CPU admission. Commits
`bb1720ca0cb803af2263ed2b112d8ba7ce11d906`,
`7e90fe8c39adc8323871d9a582f83954dfe6f678` and
`6fb71059b81720c6eacec7ac0b49b572f4b840f9` establish CPU admission
and usable-CUDA checks in training/evaluation, while
`307b552d72405c46f4763360ad5e3026a5ca6e46` prevents CPU-admitted
checkpoint RNG handling from accessing CUDA. Focused simulated-device tests
are committed. Real training, checkpoint/resume and CPU/GPU numerical parity
remain OPEN. The prior account-level “no-trainable” description must not be
used as a whole-software closure statement.


### Additional governed CPU seed isolation

The NutriFlavor governed runtime used `torch.manual_seed`, which itself
also seeds accelerator generators even if the explicit
`torch.cuda.manual_seed_all` call is skipped. Commit
`6d72903f0308717e3570d9900f52011f35628e7a` now uses the Torch CPU
default generator under central CPU admission and leaves admitted-GPU seeding
unchanged; `af63b9796442db4bc81dde698dbbc47e5bc4b166` adds focused
source-function contracts. Documentation and open evidence are in
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md`
at commit `8ec477fa073199e56bd62d6909d4aee53cf6b5b7`.
The latest CI jobs still contain zero recorded steps, so no project-level
test or real CUDA certification is claimed.


### NutriFlavorOS PPO and online-feedback closure work

The source-level follow-up in
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md`
(commit `a72ac02c5c0c5d47d96b56867b7d03aeda3ffe24`) now records
two distinct paths rather than treating all learning code as one GPU job.
The legacy observational feedback helper cannot fabricate PPO log-probabilities
or a critic baseline, discard unsupported feedback as successful training, or
claim an optimizer-less grocery update. The authenticated HTTP feedback
route remains offline-only.

The actual synthetic on-policy PPO route now preserves behavior action masks,
captures sampled critic baselines, compares masked sampling/training
distributions, computes returns on the selected tensor device, and quarantines
agents after a possibly partial optimizer failure. CPU-safe method guards and
optional real Torch PPO regression tests were committed; an isolated CPU Torch
mask/return algebra check passed. Neither real CUDA execution nor complete
PPO/real-world training validation has been established. Exact rollout and
optimizer/RNG checkpoint-resume, dataset/source governance, and account-wide
library/device-path coverage remain OPEN.


### Synthetic PPO batch-boundary follow-up

A subsequent source review found that the initial 40-transition synthetic loop
would optimize at 32 and leave eight pre-update-policy observations in the
next update's buffer. The epoch and standalone demonstration now both use
exactly one 32-transition rollout per update; the source contract test was
updated accordingly. Evidence:
`NutriFlavorOS` commits `44b62301b12e1f2603687a17cbaf3ddcb838364d`,
`ebce2f8cfc234876307272ee06474f1f22323d3a`,
`7b5241dafa9e891a7b4459d281a53a7bce42591d`.
This is a synthetic on-policy contract correction, not real CUDA execution
or generalized exact-resume certification.


### PPO behavior-version and checkpoint boundary

`NutriFlavorOS/backend/ml/meal_planner_rl.py` now guards against mixing
behavior-policy versions within a rollout. Its model save is explicitly a
weights-only artifact, stores policy version and rejects restoring weights
into a pending or compromised rollout. This is source-level correctness,
not optimizer/buffer/RNG exact-resume certification. Evidence:
`7d36be670f10c1b1aa7241dd4d207288519fec8d`,
`56aa569e2f614855572cef4901c5a296f72339e5`.
The repository audit at `29500c10b16be130aa5ff7db30811c1e6d45039a`
records the remaining exact-resume and real CUDA gaps.


### Latest NutriFlavor PPO source gate

A new lightweight regression source contract at
`NutriFlavorOS/backend/tests/test_rl_ppo_source_contract_without_torch.py`
(`8256b839ecc884be33a5bbc42302934a169ccd16`)
parses the real module sources without requiring optional PyTorch, ensuring
the default backend CI will fail on lost action-mask, retained fake-value
or removed feedback guard structures once runners execute. The optional
real Torch numerical tests still require actual runtime execution.
The repository audit is updated at
`bf950ca95767781155fe01333d7dc978635b8a2b`.
The estate CUDA audit remains OPEN.


### 2026-09-28 explicit-model and scheduler-backend admission correction

A subsequent NutriFlavorOS source audit found that six retained model
constructors accepted explicit devices without checking central CPU
admission. All six now route explicit/default device placement through
the same verified Torch resolver. It checks a requested CUDA device's
allocation and kernel completion; cached device/memory helpers mask an
imposed CPU policy. Both the legacy ML stack and the separate governed
research trainer recognize `TRAINING_CONTROL_BACKEND=gpu/cpu` from
the shared OPF scheduler: GPU-admitted workers fail closed when CUDA
cannot execute instead of claiming a CPU fallback as GPU training,
and CPU workers do not access accelerators. A failed GPU singleton
initialization is not retained.

Source/test receipts and remaining limitations are recorded in
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md`,
commit `775d6ade0a871d1b3dbb6361bedb6ed1ceed8878`.
This is source-level integration, not an actual six-model CUDA run or
full estate certification. Other repositories' independent GPU-library,
model-placement and scheduler routes remain OPEN.


### Shared OPF_ADP backend audit — independent CPU flags and immutable-pin boundary

Retained OPF_ADP `utils/ml_backends.py` was independently investigated.
It previously recognized only the literal OPF disable flag on some paths,
allowed independent CuPy/metric imports under `CPU_ONLY` and the scheduler's
CPU backend, used framework visibility without a completed kernel probe, and
seeded accelerator generators from CPU-admitted Torch workers. Its cache
could report stale GPU activation after environment changes.

The shared source now checks `CPU_ONLY`,
`TRAINING_CONTROL_CPU_ONLY`, `OPF_ADP_DISABLE_GPU_ACCELERATORS`,
`TRAINING_CONTROL_BACKEND` and stripped CUDA visibility in all backend
helper routes. CUDA proof requires actual Torch/CuPy allocation, operation
and synchronization. Cached status and successful direct CUDA probes bind
to an immutable process admission/device-mask context; later changes require
a fresh worker. CPU-only seeding uses the Torch CPU default generator, and
CPU-admitted array/metric helpers reject already-existing accelerator arrays.

The literal scheduler was also corrected to prevent nested GPU admission
from overriding a parent `TRAINING_CONTROL_BACKEND=cpu` or padded CUDA
mask. CPU-safe fake-CUDA and scheduler regressions were added and wired
to the scheduler workflow. Documentation, commit receipts and open tests
are recorded in
`OPF_ADP/docs/shared_cuda_backend_audit_2026-09-28.md` at
`d3687061296a4cfc063bce149096dcc1d973716b`.

**Distribution boundary:** The 38 main-branch root launchers still pin
RigorousRAG v39, whose inner bootstrap pins the older immutable OPF_ADP
commit `0e9e4eef90903dd1769b54b18941a8bb6e1716f8`, including the old
`utils/ml_backends.py` blob
`33108a3e20e982188ebc089399c682b11f202c4c`.
The main-branch OPF_ADP correction has **not** been retroactively propagated
to those pinned launchers. A new audited controller/reference generation
and explicit downstream repins are required; historical v39 blobs must
not be rewritten. GitHub Actions currently fails before any recorded job
steps, so a full test pass, physical CUDA execution and estate closure
remain OPEN.


## 2026-09-28 verified v40 release and main-branch repin

The active `tools/universal_training_controller_entry.py` is now the exact
versioned v40 source Git blob
`68a6f30f6ef77c2f512a0fecd6731bc35b444fa4`, with v39 preserved
byte-for-byte as `tools/universal_training_controller_entry_v39.py`
(blob `b97604e12b0c95294be44652ece0d8ab59942109`).
The new v40 bootstrap pins the separate inner commit
`7ed16bade494d2191962756a1bf1ca87d8693245` and inner blob
`046d990addec395cfeb1c4ca820be7d332853d29`, preserves 59
historical controller/scientific blobs, and changes only the verified OPF
reference overlay. It pins OPF_ADP commit
`d3687061296a4cfc063bce149096dcc1d973716b`, six runtime source
blobs plus three CPU-safe test blobs.

**Real GitHub Actions evidence:** the active v40 static workflow at
https://github.com/Anurag9000/RigorousRAG/actions/runs/36350065371
completed successfully: seven versioned source/bundle tests passed,
archived v39 cold bootstrap passed, active v40 cold bootstrap passed
(`controller_files=60`, `opf_files=9`, correct OPF commit), and 57
public controller safety tests passed. Earlier stale-version test fixtures
and the obsolete source-hash expectation were repaired without changing
frozen v39 scientific/OPF artifacts.

All 38 shared-launcher repositories had their **main-branch**
`run_all_training.py` repinned directly to RigorousRAG release commit
`29b4167737ee5f0958927ce0ad50040d17b708d1` and exact active
entrypoint blob `68a6f30f6ef77c2f512a0fecd6731bc35b444fa4`.
After the commits, every one of the 38 was independently reread from
`main`: all 38 contain precisely one new commit and blob and neither
of the old v39 identities. No native scientific job/profile bodies
were rewritten by this controller-only replacement.

| Main-branch repo | Root repin commit |
|---|---|
| Text-and-Emotion-Analysis-Tool-with-Visualization | `b07f1cf0930191ad334e8cd4af4ac6e667793df8` |
| CO-project | `3b4ba38380b62a84762631216edaf9e2688163b7` |
| Resume | `fc8e8b0bd8e2197ed547fc7660b10d7c486d2514` |
| Making-LLMs-fill-reimbursement-form | `482689cfe7adb3f9d962cb7b5fac0a3fd31f8141` |
| Companies-institutes-and-Internships | `096eee64e8ef7d532d535abab5b9af161baed731` |
| dragonball-chess | `fe3fd432dbc87788abdf189eaa1aaa2e3c7cb2d6` |
| ERP_College | `a706d0aa72aa9b801362c45759beb6f4b45b21f8` |
| Breaking-the-Neural-Barrier | `dcd4fe46cd9cef77a940827b95ee394952d07006` |
| Gram-Connect | `e4a78a977639cee487c4afd5e2601791c6b9800b` |
| Traffic | `599720d7a41e317caebb2029e3cf0842c8603b31` |
| ERP_Web | `9c2d8b22090e45b33929da8a463b09fce215d5ab` |
| Signal-Prophet | `d67d50bc6923c473da325aeb02efa5167e9b8f00` |
| modular-grokking-adp | `38326716513fff4bbbfb2155f2d4d3f934702dcc` |
| hallucination-resistant-llm-framework | `f0c9b2c1d9579bb7c4023a117d9ca22f8875dcd5` |
| NutriFlavorOS | `e04df979dd719adb8daab1058c97f05f2296a396` |
| PlaceMate-AI | `cd8d853786458c4b358e90a10f4f319a9a34fc39` |
| HydroGraph-Delhi (`main` only; default `master` remains older) | `8d030e3bf7b5f1c79c5c2e2a9e7c4b8578de92b5` |
| Silicon-Pilot | `3cbc6e9d78c5c7c4df7fba8daa88bd6f00f5a625` |
| existential-coordination-games | `2d50396c60052c32a03ea4c0bac8fb0423daff05` |
| ESD_Project | `d9b23a580c34eddb17b1fc2a8683fe04f20d6f2a` |
| End-to-End-Digital-Communication-System | `1851711c49cf07e80459735412bcaf714d219950` |
| Smart-Glasses | `ce768ee599d30faab2c30cf39aeb51fd60f568ee` |
| Continual-Learning | `7c42b6aa735bbceb7ae9262510c58ecea16d6f3f` |
| VaaniNoise-SED | `c5944f08dfe5fb8116c6c598496bfb4c74609205` |
| VaaniEventClean | `6e801123f533387c4315072280ceb57b8979713a` |
| TutorMistakeLens | `e6bf1d33c5dd036566e2d1a5ee970ecf8fd8c8ea` |
| TutorGuidance-Eval | `205e5803b7281fd033fc6d4fd0910538aab46bcf` |
| CatalogPathForge | `9b871679e7cfcc3a74d681055c79bca11fc3c839` |
| ReceiptTupleForge | `a29b3a97835964639be2ad047486361397300dd5` |
| MASSIVE-IntentForge | `ac54641c673b1c7beb6ab4917fb894d0265cf9f6` |
| IndoIntent-150 | `ada1a90d3fc78b02cd1826830b428196b9dd63b2` |
| DocVision | `556469f7c281942e20c469729c460a252dd2a336` |
| IndoDocFusion | `3fbba4b29a7456cef4940b31c9548d7281ba7136` |
| HinglishKnowledge-NLI | `9aedc81eb2dc13140679df378cd023157fd5f7af` |
| IndicRelationForge | `51a822a753789f7573e4af02c1b81b01e7c17e85` |
| continual-learning-with-rl | `41eb34a55c00fd55ef6149defb4700265cb69065` |
| Railguard-AI | `9fab7e92d69d042a8f47744d381a5c5d0018aca3` |
| False-News-Interpretability | `582276d95cf2ad33464c182ec276107d05a8bfe0` |

The CLRL v89 source verifier was additionally updated at
`f0d7b69cf8e55b17ce89b80d4e7abadea6d08fde` to pin its new
wrapper blob `4a0cb3f31d61ca43d4bb87b258bfc6714bae2bd1`
without changing the frozen v89 scientific-authority and catalog blobs.
Its controller regression was updated at
`14c88963850264529d517cfcd52f15aa931ea46c`.

**Important limitations:** the OPF_ADP private-repository Actions workflow
has thus far failed with zero recorded job steps, and v40 cold bootstrap
does not download the private OPF source unless
`TRAINING_CONTROL_SELF_TEST_OPF=1` is set. The nine OPF paths/hashes were
verified against the pinned Git tree through the authorized GitHub connector;
this is not the same as runtime import on a clean GPU server. Per-repository
full scientific/job coverage, physical CUDA execution, library substitution,
OOM-recovery and CPU/GPU numerical parity remain independently OPEN.
Six repositories have separate native/no-trainable or source-specific
orchestration cases; the 38 shared-root repin count does not imply all
44 are execution-certified. HydroGraph's default-branch migration is OPEN.


### Post-rollout source-contract and six-special-case check

The CLRL current v40 controller regression now resides at
`tests/test_cuda_controller_pin_v40.py` (creation
`8a6d8fe5fd374e52dcd25b2583c68da5e0b6bdd0`);
the v89 source workflow was rewired at
`472e9921c18d2baca52ba9f1dd6d4f81aefd45c4`, and the misleading
v39-named duplicate was removed from `main` at
`606a5da750b8b72c9165cdc8e63b909c7df7f401`.
Its scientific source hashes and row-integrity requirements remain intact.
The CLRL GitHub Action still had no assigned runner or recorded steps;
no remote test pass is claimed.

The six non-shared-root cases require independent applicability judgments:
`OPF_ADP` is the actual scheduler/backend source (no shared launcher);
`RigorousRAG` now uses its local active v40 controller;
`STM32n6AI` has its own `stm32n6ai.training.central_runner`;
`Forest_Run` intentionally runs a fail-closed no-retained-ML-surface audit;
`Last-War` has no `run_all_training.py` at the inspected `main`;
and `Project-Foundry` was empty at the inspected `main`.
These are **not** counted as repinned shared launchers. No special-case
repository has inherited an unverified training PASS.

A retry of OPF_ADP Actions run
https://github.com/Anurag9000/OPF_ADP/actions/runs/36349422266
also reported failed with no assigned runner and zero executed steps.
The controller's v40 validation in RigorousRAG is independent of an actual
private-OPF runner test or physical CUDA hardware test.


### 2026-09-28 v41 release and all-38 main-branch source-pin census

The first shared backend isolation fixes were implemented on OPF_ADP main,
but active RigorousRAG v39 still pinned the old immutable backend. v40
introduced the corrected Torch/CuPy allocation-and-synchronization checks,
independent CPU flags, cache/admission isolation and literal scheduler parent
mask. A separate review discovered that the retained OPF scheduler regression
still asserted the **old** v21 source blob; this assertion was corrected on
OPF_ADP main at `a361b49ac24ffc7de87440538c88994faed017c4`.
The corrected regression is Git blob
`0fc13aa7d5f5bc0cdef447c8d792968115f00b47`.

To avoid rewriting v40 historical manifests or invalidating earlier cache
identities, RigorousRAG v41 was issued as a distinct versioned release.
Its active outer entry at commit
`e2acf14bb06c4d72a343024d6e3146a82756c1df` is Git blob
`2f34cf0a1319c00d04bcfa99972f6e209534a096`. It pins OPF_ADP
`a361b49ac24ffc7de87440538c88994faed017c4`, the corrected
scheduler test blob and eight other exact OPF sources/tests. The
historical v39 and v40 outer entries remain available by their original
Git blobs, and all 59 unchanged scientific/controller files retain their
historical hashes. Only the OPF reference overlay differs in the v41
controller manifest.

A source-level reread of all **38** regular `main/run_all_training.py`
launchers found the exact v41 outer commit/blob pair once each, without
the previous v40 pair. These changes were made directly to each
repository's `main` and preserved its native scientific profile and
catalog. In `continual-learning-with-rl`, the live-wrapper pin and
v89 source-verifier hash were reconciled; its frozen scientific-authority
blob `083c4aa3f19e585caafbec6f6007866447a3a6e7` and v89
catalog blob `df504ca1417d7005d6df50c921f740ee176286b1` remain
unchanged. Its controller regression was renamed and wired as
`tests/test_cuda_controller_pin_v41.py` at
`2c3af6109fe0d913664455bf70a1130cf42b7ebd`.

**Executed public CI:** RigorousRAG training-control-static run
`36351742143` passed 12 release/source tests, archived v39/v40
cold bootstraps, the active v41 cold bootstrap and 57 controller
safety tests. The OPF dependency's private GitHub Actions workflow
`v41-shared-cuda-reference` explicitly checks the nine exact source
files against a local OPF checkout and runs the CPU-safe backend and
scheduler tests, but run `36351824190` ended with no runner assigned
and zero recorded steps. The CLRL source workflow also ended before
runner assignment. Therefore private OPF test execution, real CUDA,
full 38-repository bootstrap and per-model numeric parity are **OPEN**.

The other six account repositories retain their separate authority
classification; v41's 38-root pin census does not certify them.
`HydroGraph-Delhi/main` is included among the 38, but its GitHub
default branch remains older `master`; the main-only/default-branch
migration is still OPEN, not disguised by the main-branch census.


#### HydroGraph default-content reconciliation and NutriFlavor fail-closed metadata

Following the v41 38-launcher census, `HydroGraph-Delhi` still had
default `master` eight commits behind `main`, with zero master-only
commits. A non-force, fast-forward branch-ref update moved the existing
`master` ref to `498ddff2f0f60a004f4b04f0b46f73c5bb4c5a55`, exactly
equal to `main`. The GitHub default view now exposes the v41 launcher
Git blob `0c5bee6cea6f9a6ce5837e9dbbacf721b86b3260`, and all
master history is preserved. **The default branch is still named
`master`; the desired main-only topology requires changing the GitHub
default and retiring master via repository administration.** No such
change or branch deletion is claimed.

`NutriFlavorOS/run_all_training.py` retained a false audit-job exemption
reason claiming it had no optimizer even after its source scanner
correctly detected synthetic and online training. The exception reason
was corrected at `fbf665f1afd72412af4cda6f4ed771fcf6f054ce` to
identify that node as a source-classification audit with unresolved
optimizer paths. A CPU-safe regression was added at
`15418a6f998e1d2a52395be973dcb02730267ee6` asserting the
scanner remains fail-closed and detects the synthetic, online and
governed trainer surfaces. It does **not** fabricate production jobs
or certify completion of the retained training DAG; the source
inventory, execution and parity remain OPEN.


#### Smart-Glasses native GPU helper correction

Independent review found that
`Smart-Glasses/src/smart_glasses_eye_tracking/gpu_compat.py`
could activate optional GPU modules despite generic `CPU_ONLY`,
`TRAINING_CONTROL_CPU_ONLY` or the scheduler's declared CPU backend;
it also cached CuPy imports across subsequent admission changes and
previously checked CUDA visibility/allocation without a completed
operation. Native source commits
`da8449062a6500aef0d7f7fecb49df255a71d2ee` and
`ef64a5abf2401a2aaff783e1406a729b695b8bd1` now enforce independent
CPU admission before public helper probes/conversions, reject
preexisting accelerator-array conversion in a CPU child, and run
allocation + operation + synchronization for Torch/CuPy capability.
A scoped test extension at
`d0c71b2bb1e487831588e7fe55cab1e05e56abcc` and new
`gpu-compat-cpu-contract` workflow at
`44a3d07dea9983e20a9cce2e75183ce16e42e112`
cover the admission aliases and mockable probe contract.
Source-specific limitations and receipts are in
`Smart-Glasses/docs/gpu_acceleration.md` at
`6ff25d0cc4ccaa670ad3174e303c90084d08e9d8`.
The private workflow again reported no runner/steps, so this is not
a source-test or real-GPU pass. ONNX/OpenCV/cuDF advertised providers
still require actual task-execution proof; the full Smart-Glasses
functional/ML surface and other repositories remain independently OPEN.


#### Breaking-the-Neural-Barrier native backend correction

The BTN v4 scientific root already pins the audited v41 universal controller,
but a separate investigation of its retained native device/RAPIDS helpers
found a bypass of independent `CPU_ONLY`, `TRAINING_CONTROL_CPU_ONLY`
and explicit scheduler backend admission. `gpu_detected()` previously
cached hardware/enable-flag visibility; Torch device placement relied on
`is_available` without allocation/kernel completion; first-use
`default_device` caches could outlive a subsequent CPU mask; and direct
RAPIDS, RNG, cleanup and Torch seeding helpers could touch accelerators
within CPU-admitted workers.

Targeted `main` corrections in `utils/gpu_acceleration.py`,
`utils/device_utils.py` and `utils/seed_utils.py` enforce independent
CPU/GPU admission, check actual Torch/CuPy operations and synchronization,
probe explicit local CUDA indices, prevent GPU-admitted Torch CPU fallback,
key device caches to current policy, block direct RAPIDS/CUDA helpers
under CPU admission and seed only the Torch CPU generator there. Existing
tests were reconciled and `test_btn_gpu_admission.py` added.
The repository audit is
`Breaking-the-Neural-Barrier/docs/native_cuda_admission_audit_2026-09-28.md`
at `a929b257fe91b613dd6b4aa6dc828f1622ab4209`.

Dedicated CI attempt `36356977221` failed without a runner or executed
job steps; its source tests and physical CUDA remain unverified.
The source-level findings do not close the 7,679-file whole-software
functional/model/dataset investigation. Native source changes preserve
the scientific catalog and model/experiment authority.


#### STM32n6AI native nested-worker CUDA admission correction

The STM32n6AI retained central campaign is a distinct, stateful native
worker scheduler, not one of the 38 simple universal-controller wrappers.
Its prior `training/device_policy.py` did not reconcile parent OPF
backend admission, independent CPU-only aliases, or actual Torch
allocation/kernel execution. Its worker launcher inherited
`TRAINING_CONTROL_BACKEND=gpu` even when intentionally creating a
separate CPU fallback worker.

The native device resolver now checks native and parent admission, the
generic CPU flags and normalized CUDA mask, requires allocation,
operation and synchronization at the selected local CUDA index, and
does not turn an admitted GPU failure into CPU success. The executor
rejects GPU escalation from a CPU parent and GPU assignments outside
an inherited CUDA mask before route resolution. It explicitly records
the fresh child backend, local GPU index, and CPU-only alternate
library masks for legitimate CPU downgrade. The worker validates the
same policy before route resolution; existing source checksum and
actual trainer-device evidence remain active.

Source receipts and regression evidence are recorded at
`STM32n6AI/docs/native_cuda_admission_audit_2026-09-28.md`
(commit `82493cf810fde01f266873622c70602702030232`).
The exact Git blob of the standalone native device policy was verified
locally and its focused CPU/fake-CUDA suite passed 15 cases. The
full native executor/worker tests are checked in, but GitHub Actions
run `36357625830` ended with zero steps/no runner. Physical CUDA,
per-model training, datasets, pressure recovery, checkpoint exact
resume and firmware/export verification remain OPEN.


#### Breaking-the-Neural-Barrier native Torch/CuPy/RAPIDS and nested-child admission

The active shared v41 controller pin alone did not close BTN's independent
`utils/gpu_acceleration.py`, `utils/device_utils.py`,
`utils/seed_utils.py` and `utils/pressure_control.py` paths.
BTN's native GPU detection accepted a manual enable flag or Torch driver
visibility without a completed operation, omitted independent CPU-only
aliases, and its cached default device could remain CUDA after parent CPU
admission. Native fixes now require actual allocated Torch/CuPy operations
and synchronization; gate optional RAPIDS imports, CUDA seed/RNG/cleanup,
and default device selection on the parent CPU/GPU policy; and fail
GPU-admitted Torch work instead of silently relabeling CPU fallback.

A separate nested-child audit found that BTN's
`build_cuda_visible_devices_env` could override parent CPU masks and
remap to GPUs outside an inherited allocation. It now rejects such
requests and narrows already-assigned device masks by local index.
Focused fake-CUDA, CPU-only Torch, pressure, seed and visibility-ownership
regressions are checked in and routed through
`.github/workflows/btn-cuda-admission-contract.yml`.
Full source investigation, test map and remaining proof obligations:
`Breaking-the-Neural-Barrier/docs/btn_native_cuda_admission_audit_2026-09-28.md`
(commit `0df8279f4f5163b6583d5c583800e7b5fe552a80`).
The checked native helper/source blobs were independently reread from
`main`; scientific authority `training_catalog_v4.py` was not changed.

The recent focused BTN workflow likewise failed **before runner
assignment**, with zero recorded steps. This is not a test pass, a
demonstrated source failure, or actual CUDA verification. The retained
7,679-file BTN tree and hundreds of model/workflow families remain
independently OPEN for full device-path, numeric parity, dataset and
resume/pressure testing.


#### Forest_Run applicability proof and CL Executive native CUDA-admission correction

`Forest_Run` is a Kotlin Android game, not a training repository.
Its original no-trainable scanner returned a pass for an absent/empty
root (zero files and zero findings). The certificate authority now
requires the retained Android build files, manifest, MainActivity,
GameView, production Kotlin, and actual Android dependency declarations
before issuing the N/A classification. The regression test exercises
an empty root, each missing required file, an empty dependency
manifest, existing non-ML vocabulary, and injected ML markers.
An actual GitHub Actions run `36362901297` passed **7 tests** and
issued a source certificate with 614 scanned source/config files and
zero ML-training findings. CUDA/CuPy training is therefore correctly
N/A for this repo; Android production/device acceptance remains a
separate open gate.

`continual-learning-with-rl` required a separate fix despite its
successful v41 controller pin. Its OPF CLI bridge previously inferred
CUDA from visibility plus `torch.cuda.is_available()` and could
silently inject CPU for a GPU-admitted child. Its original
`src/cl_exec/runner.py` similarly defaulted to CPU after a failed
CUDA request and seeded accelerator generators without respecting
independent CPU-only aliases. New
`src/cl_exec/device_policy.py` requires Torch allocation+operation+
synchronization, validates the parent backend, masks CPU workers,
and fails GPU-admitted jobs when CUDA cannot execute. The bridge
and core/resilient runner now share the policy; failed GPU-admitted
seeding rejects before advancing Python/NumPy/Torch RNG. Focused
fake-Torch/AST source tests and a dedicated CI workflow were added.
Details: `continual-learning-with-rl/docs/cl_native_cuda_admission_audit_2026-09-28.md`,
commit `423bbbb080ab91ee5a5105aab370e8c86cd2145b`.

The CL private Actions run `36363304137` failed before runner
assignment with zero recorded steps, so its new regression result,
per-route model and array placement, actual CUDA, OOM/exact-resume
and numerical parity remain OPEN. The frozen v89 scientific
authority/catalog were not modified by the native helper changes.


##### CL Executive additional native CRL and Stage-13 device routes

A subsequent source audit identified four independent CRL/Stage-13
execution paths that bypassed the just-added CL Executive/OPF bridge
and directly used `torch.cuda.is_available()` plus global Torch/CUDA
seeding. `crl_runtime.py`, `stage13_runtime.py`,
`stage13_ppo_runtime.py` and
`resilient_stage13_runtime.py` now call the common
`prepare_torch_execution` authority with strict explicit-CUDA
semantics. CPU-only children seed only Torch's CPU generator; failure
of a GPU-admitted CUDA execution probe raises before touching RNG
state. The scientific objectives, catalog, and checkpoint layouts
were not edited. Focused fake-Torch source tests and workflow compile
coverage were expanded. Detailed source evidence:
`continual-learning-with-rl/docs/cl_native_cuda_admission_audit_2026-09-28.md`,
commit `f5c3cf83c203657a6c80369dd513cba7de51c420`.
No full family graph/numerical parity/physical CUDA or private CI pass
is claimed. The frozen v89 scientific authority and catalog retain
their original blob identities.


##### CLRL expanded native CUDA call-site tranche and exact RNG boundaries

An additional source-level literal scan of 96 distinct CLRL
runtime/runner/agent modules found direct Torch CUDA capability,
global seed and CUDA RNG checkpoint accesses outside the previously
corrected original runner and Stage-11/13 entrypoints.
Additional Stage-12, resilient Stage-11/12/13-PPO, named CL,
prompt CL and R20 replay routes now share device admission.
CPU-only workers do not touch CUDA generators. GPU-admitted
workers fail before making new RNG state on unusable CUDA.
R14/R15/R16 task-free and v59 structural-routing checkpoint
CUDA RNG is now bound to the actual selected device; a CUDA state
cannot silently be omitted on an apparent interruption-exact
CPU resumption. CRL agent initialization, vector learner private
CPU initialization and transformer PEFT metadata probing also
avoid unsanctioned global Torch CUDA seeding. The 96-module
pattern scan is not a full semantic audit of the repository's
1,512 files or all task/algorithm combinations.

All receipts and the remaining source/CI/real CUDA distinction
are in `continual-learning-with-rl/docs/cl_native_cuda_admission_audit_2026-09-28.md`,
commit `e0d344f32356045bf20aedff7b212acdbc18683f`.
The frozen v89 scientific authority/catalog are still the original
Git blobs `083c4aa3f19e585caafbec6f6007866447a3a6e7` and
`df504ca1417d7005d6df50c921f740ee176286b1`;
root wrapper remains pinned to verified v41. The private focused
workflow run `36364338763` had no assigned runner and no steps,
so **new test and real-GPU execution evidence remain OPEN**.


##### CLRL checkpoint preflight before learner-state mutation

The previous CLRL device/RNG restoration guards correctly rejected
mismatched CPU/CUDA generator states, but seven checkpoint readers
loaded model/optimizer/cursor or replay state **before** reaching the
RNG mismatch check. The common policy now provides a pure
`validate_checkpoint_cuda_rng` preflight, placed after schema,
identity and geometry verification but ahead of any learning-state
mutation in named, prompt, R20, R14, R15, R16 and v59 checkpoint
loaders. It rejects absent/invalid RNG fields, missing CUDA state
on a CUDA worker, and CUDA state in a CPU worker. Existing
restore-time checks remain; other malformed learner checkpoint
content is not yet transactionally certified.

Sources, seven per-loader receipts, source-order and fake-RNG
regressions are recorded in
`continual-learning-with-rl/docs/cl_native_cuda_admission_audit_2026-09-28.md`
at commit `95651caf6e39b0e8cc76dadfe0887a2ac8b6836f`.
No new scientific objective/catalog or root controller pin was
changed. Real CUDA/CPU-resume parity and private CI execution remain
OPEN.


#### Forest_Run no-training certificate continuity and source-enumeration hardening

Following the seven-test no-ML applicability proof, a direct audit of the
root certificate path found an independent stale-evidence bug: an exception
during a later source audit left the previous `status=pass` report on disk.
The root entrypoint now invalidates its chosen output *before* inspecting
source and, on an audit error, publishes a nonzero-exit
`status=fail/classification=unresolved` report with no ML-applicability
verdict or execution claim. A publication failure likewise cannot retain
the old PASS at that path. Its scanner now exempts only the root
`run_all_training.py`, not nested scripts of the same basename, and
includes shell/XML launch/configuration sources while excluding generated
audit artifacts that could poison future scans with historical failure text.

The actual public GitHub Actions run
`36366366530` on commit
`341abc26c21eaa26441f214020b42f050aa0d40d`
executed **12 applicability regressions successfully** and certified
**633 scanned source/configuration files with zero ML-training findings**;
the independent repository-local certificate run `36366366579` also
passed. Full receipts and source-specific limitations:
`Forest_Run/docs/audits/2026-09-28_training_applicability_fail_closed.md`,
commit `062452a74d24ebdf338b5e1d8dacbd80c0720b58`.
This extends the earlier seven-test source gate; it is not a new ML or
GPU-training workload and not a whole-game/release acceptance certificate.
The Android host/API35 connected run `36363796716` succeeded on an
**earlier** commit, and cannot be applied as if it tested a later source
candidate. Signed/store delivery, real-device/human approval, gameplay
quality, security and final release readiness retain their own gates.
