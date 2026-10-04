# CUDA-first account audit — source and pin verification

Original snapshot: 2026-09-27; reconciliation through 2026-09-28. Scope: 44 connected repositories under Anurag9000, inspecting live `main` root launcher content and the shared controller's Git-blob manifest.

**Truth boundary:** This is an account-wide launcher/pin inventory, not a complete per-function execution audit or a CPU/GPU parity certificate. Repository-specific scientific/job completeness and hardware execution remain independently unverified unless explicitly stated.

## Verified shared-controller chain

- All **38** regular root launchers were reread on `main` and pin RigorousRAG v41 activation commit `e2acf14bb06c4d72a343024d6e3146a82756c1df` together with entrypoint Git blob `2f34cf0a1319c00d04bcfa99972f6e209534a096`.
- This v41 source bundle pins OPF_ADP commit `a361b49ac24ffc7de87440538c88994faed017c4` with the audited `utils/ml_backends.py` blob `d8f808be88c19b32d82d66e31da1a45dd7b56e27`, the literal scheduler blob `30067d5aeb6c1cc3debad8be086055c110c0899c`, and the corrected scheduler-test blob `0fc13aa7d5f5bc0cdef447c8d792968115f00b47`.
- RigorousRAG public static CI at v41 activation passed 12 bundle tests, archived v39/v40 and live v41 bootstrap checks, and 57 controller safety tests; the separately private OPF_ADP integration suite did not execute because its job never received a runner.
- `Forest_Run` has a separate, applicable no-training authority with an executed passing certificate and Android host checks. `Last-War` remains CPU-native with optional CUDA observation transfers and no learned RL optimizer. `Project-Foundry` is empty; no trainable capability can be inferred.
- Pinned scheduler source identity is not proof of real CUDA training. Each trainable repository must independently establish usable framework runtime, selected child device, GPU-library activation, numerical output, legitimate training signals, checkpoint/restart and fallback semantics.


## Repository matrix

| Repository | Root controller | This audit's implementation finding |
|---|---|---|
| Text-and-Emotion-Analysis-Tool-with-Visualization | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| CO-project | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Resume | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Making-LLMs-fill-reimbursement-form | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Companies-institutes-and-Internships | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| OPF_ADP | Backend/scheduler source repository | Targeted GPU/CPU source fix committed; runtime verification still open |
| dragonball-chess | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| ERP_College | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Breaking-the-Neural-Barrier | Verified v41 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Gram-Connect | Verified v41 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Traffic | Verified v41 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| ERP_Web | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Signal-Prophet | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| RigorousRAG | Shared-controller source repository | GPU/CPU feature-level coverage not yet certified |
| modular-grokking-adp | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| hallucination-resistant-llm-framework | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| NutriFlavorOS | Verified v41 commit and entry blob | Targeted GPU/CPU source fixes; source-only audit root truthfully leaves retained optimizer workloads unresolved |
| PlaceMate-AI | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| HydroGraph-Delhi | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified; `master` fast-forwarded to `main` but remains named default |
| Silicon-Pilot | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| existential-coordination-games | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Forest_Run | Separate Android-game no-training authority | Executed source certificate and Android host validation passed; physical-device/store/human gates open; ML training not applicable |
| ESD_Project | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| End-to-End-Digital-Communication-System | Verified v41 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Smart-Glasses | Verified v41 commit and entry blob | Targeted GPU/CPU source fix committed; runtime verification still open |
| Continual-Learning | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| VaaniNoise-SED | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| VaaniEventClean | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| TutorMistakeLens | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| TutorGuidance-Eval | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| CatalogPathForge | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| ReceiptTupleForge | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| MASSIVE-IntentForge | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndoIntent-150 | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| DocVision | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndoDocFusion | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| HinglishKnowledge-NLI | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| IndicRelationForge | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| STM32n6AI | Separate native central-runner authority | Targeted GPU/CPU source fix committed; runtime verification still open |
| continual-learning-with-rl | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Railguard-AI | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| False-News-Interpretability | Verified v41 commit and entry blob | GPU/CPU feature-level coverage not yet certified |
| Last-War | Separate CPU-native simulator and Torch observation policy | C++ world/benchmark baselines exist; optional CUDA observation transfer, no neural learner or GPU-native simulation; runtime verification open |
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


#### Last-War native simulator versus CUDA observation and learner classification

The 1,448-file Last-War repository has a real authoritative C++ world,
CPU-parallel batch API, deterministic native AI baseline policies,
task/Gymnasium/PettingZoo adapters, public multiagent trajectories and
replay. It has **no retained optimizer-driven RL learner or root
`run_all_training.py`** in the inspected source. Optional
`research/lastwar_tensors.py` and `lastwar_graph.py` transfer
country-filtered CPU-generated observations to Torch CUDA; they do not
execute the C++ simulation on GPU, implement CuPy-native world mechanics
or prove trained-agent self-play. The OPF v41 38-launcher pin inventory
does not include this independent engine/research authority.

A focused audit found that both observer adapters wrote
`last_transfer_bytes` before tensor conversion/transfer returned,
so a failed C ABI observation or Torch copy could leave a stale/nonzero
reported transfer. They now reset this counter on a new observation
and record only successfully returned tensor bytes
(commits `fa9dad21a06cf62bcbca3672c34c1ef39ea76243`,
`09eaef2fe6c8b394aa75d31664fbd847a14448d1`).
The existing GPU allocation/fill/synchronization policy was also
hardened to reject backend/mask changes **during** device probing,
including the automatic CPU fallback, and to bind both observer
constructors to the admission signature captured *before* selection
(commits `4df29953045bb5331acb1aaf881ab7f3ddaf86d2`,
`5a1cd07bcaaa05a4160edec86eb94ebe7d9f5825`,
`6778cf05e35823d05ed97450c9f6ad7569c63682`).
Tests now cover failed native/copy paths and policy changes during
probes/constructors; obsolete docs describing mere `cuda.is_available()`
selection were reconciled.

Full sourced boundary, test map and open learner/native-device work:
`Last-War/docs/48-native-observation-vs-training-audit.md`,
commit `d6044958a21ec2abcd444fe33a2be894745aea6f`.
Private GitHub Actions runs `36367102015` (observation) and
`36367102009` (native engine) finished with no assigned runner and
zero executed steps: **no source-test, native build, physical CUDA,
training or numerical-parity PASS is claimed**. A legitimate learner
requires its own data/optimizer/masking/checkpoint/evaluation design;
deterministic self-play traces must not be classified as PPO training.


#### Forest_Run and Last-War independent non-ML / optional-observation classification

Forest_Run is a native Kotlin/Android game, not a retained ML trainer.
Its applicability authority previously could classify a wholly missing
Android tree as no-training. It now requires its actual build/manifest,
MainActivity/GameView and production Kotlin/dependency surface before
issuing a certificate. The root's training-control audit and estate-local
certificate both passed on `main` commit
`5cf8da8efe5d13d8284dbff50f6dac5a9d0721d0`
(Actions runs `36367701754` and `36367701811`). Android host compilation,
unit/lint, packaging and release hardening passed separately at
`36367701937`; its API 35 connected job was in progress when inspected.
No physical-device/store/human-release or GPU-learning evidence is
implied by these source and host passes.

Last-War is an independently classified native C++ CPU world with
batched C ABI and optional Torch CUDA **observation transfer**, not a
GPU-native simulator or gradient-based RL learner. The initial
observation policy already required an actual CUDA allocation, operation,
and synchronization and enforced CPU/GPU admission. Further inspection
found an admission-change window after native reads and during
Torch conversion/nonblocking transfer: both observer types could
publish a returned tensor and a nonzero byte count after the parent
mask changed. Main commits
`36ec68902177a63dea8b1ad5df46c8689c2e9859`,
`e3b5741f534f7a62bae48b0b1e0b05a392ef5750`
now recheck admission after the native/Torch operations and before
successful byte accounting. Native graph legal-action flags are checked
as **binary** at
`2ae9ceeda18408da5ad0427481f50c1117f38247`
so malformed truthy integers cannot create purported legal actions.
Regressions include native-read, CPU conversion and mock CUDA enqueue
races, plus non-binary action masks. Receipts and explicit unverified
training/hardware status are in
`Last-War/docs/48-native-observation-vs-training-audit.md`
(commit `854000e69b2d02c6ade3b010f81bc65ad8e762ad`).
Private Last-War workflow runs still fail before runner assignment:
neither native build nor observation tests are recorded as passing.
Its neural RL optimizer, GPU simulation, full game and exact learner
resume remain OPEN and must not be counted as present.


#### Forest_Run training-control applicability closure evidence

Forest_Run is a native Android/Kotlin game rather than an ML repository, so
GPU-first training, CUDA optimizer state, CuPy substitution, training parity,
model registries and training DAGs are correctly **not applicable**. The
fail-closed authority was strengthened at
`66cdfa55f703c2ce183d7e559ea5366fd3ee7bba` so an empty or
partial repository can no longer be certified merely because the scanner
finds no training markers. It now requires the real Gradle/app manifest,
MainActivity/GameView production source and a nonempty Android dependency
surface before issuing the no-training certificate. Regression coverage for
empty/partial/dependency-free trees was added at
`e86195a69278210fbb820ce2a267a0d692f92ddb` and wired into
the applicability workflow at
`2963386d5f40b5c7735fdd9d1de1e25ee969575d`.

Current `main` is `f2bc315622a16c4a444ca9cc21a665cb8b6737b0`.
At that exact head:
- training-control applicability run `36378248446` succeeded, including
  compilation, the new absent-tree/injected-marker regressions, and the final
  no-retained-trainable-surface certificate;
- estate local certificate run `36378248363` succeeded;
- Android validation run `36378248320` succeeded for the host/release/lint/
  packaging job and API-35 connected smoke/deterministic-evidence job.

This closes only the ML/training **applicability** classification and its
source/CI evidence. Forest_Run's own documentation still correctly keeps
external physical-device acceptance, human/artistic approval, production
signing, Play delivery/policy declarations and accountable release approval
outside source-only closure.


#### Forest_Run no-ML training applicability now executed and scoped

Forest Run is an Android/Kotlin game rather than an ML-training repository.
Its fail-closed authority was hardened so an empty/partial checkout can no
longer pass merely because no optimizer marker exists. It now requires the
actual Gradle/manifest/MainActivity/GameView application surface, production
Kotlin source and Android dependencies before issuing
`no_retained_trainable_surface`.

At commit `d461353aa0b7e4f7bf455eb001ebf8a7640fd4f6`,
GitHub Actions training-control run `36386648800` obtained a runner,
executed **12** applicability tests successfully and emitted a certificate
with zero findings across **635** scanned files. The estate-local certificate
run `36386648793` and broader Android validation run `36386648828`
also succeeded at that exact commit. A documentation-only follow-up is in
`Forest_Run/docs/audits/2026-09-28_training_control_no_ml_surface.md`
(commit `0b64e9512e0f67becd5347ecf5c996d03ab81f2b`).

Therefore CUDA-first **training**, model/optimizer/dataset training controls,
CPU/GPU *training* parity and ML checkpoint/resume are correctly NOT
APPLICABLE for the verified Forest Run source. This scoped closure does not
certify rendering GPU behavior, physical-device acceptance, Play delivery,
production signing, legal/creative approval or human release acceptance.

#### Forest_Run no-ML applicability closure — 2026-09-29

`Forest_Run` was audited as an independent native Android game rather than
being forced through an ML/GPU template. Its repository-level training-control
authority now refuses to certify an empty, partial or dependency-free Android
tree: it requires the concrete Gradle/application entry surface, production
Kotlin source and retained Android dependencies before issuing
`no_retained_trainable_surface`.

Candidate `4708779c14388e458921675c5869953db6b6bdec` passed the
strengthened Training-control applicability audit (`36523170411`), Estate
local certificate (`36523170375`) and Android validation (`36523170374`).
The Android workflow completed host/release/lint/packaging, unit tests, release
hardening, page-size/R8/source checks and the API 35 connected smoke job.

The same candidate closes the Hyacinth swept-brush regression exposed by the
previous Android run: the test now derives frame travel from live brush geometry
and current base world speed rather than assuming maximum scroll speed.

Forest Run therefore has no applicable CUDA model-training/CuPy/optimizer
surface to implement. Its remaining open work is product-release evidence
(production signing, store delivery, representative physical devices, human
visual/gameplay/accessibility approval and store/privacy governance), not a
fabricated ML training DAG. Detailed receipt:
`Forest_Run/docs/audits/2026-09-29_training_control_no_ml_closure.md`.

#### Continual-Learning deterministic CUDA/exact-resume correction — 2026-09-29

`Continual-Learning` already had a strong repository-local CUDA admission layer:
central CPU aliases, scheduler backend identity, worker-local `cuda:0` mapping,
inherited GPU-mask checking and an allocation + operation + synchronization
probe before Torch CUDA is accepted. The new audit found a different correctness
gap: seeded CUDA execution explicitly used nondeterministic cuDNN mode while
the repository/controller claimed deterministic exact resume.

Native changes now enforce deterministic Torch algorithms, cuDNN deterministic
mode with benchmarking disabled, deterministic cuBLAS workspace configuration
before CUDA use, 32-bit seed validation, and process-boundary Python hash seeding.
`scripts/pressure_aware_catalog_runner.py` passes the job seed to the scientific
child as `PYTHONHASHSEED`; multi-member cohort parents use structural hash seed
`0` while retaining per-member RNG virtualization. Standalone `train.py` and
direct cohort execution re-exec before scientific imports when necessary.

The audit also repaired a literal `\n` embedded in the
`continual-memory-science` workflow path list. The workflow now materializes a
real `cpu-science` job, and deterministic-boundary tests are wired into both
memory-science and training-control verification. Private Actions still report
no assigned runner and zero executed steps, so these tests are not recorded as
passing and no physical CUDA parity/exact-resume run is claimed.

NumPy was retained for host RNG/checkpoint/metrics semantics rather than blindly
replaced with CuPy; model tensors already use admitted Torch CUDA, and changing
host RNG/checkpoint state would change scientific semantics rather than provide
a safe drop-in acceleration. Detailed evidence and open obligations:
`Continual-Learning/docs/cuda_exact_resume_audit_2026-09-29.md` at commit
`83be1b864aa160f603080cd1a7014426a1079a3c`.


#### continual-learning-with-rl core exact-resume RNG/device correction

A follow-up to the previously recorded CLRL native CUDA work found one common
exact-resume bypass in `src/cl_exec/resume_support.py`: global RNG
capture/restore still queried `torch.cuda.is_available()` and CUDA RNG state
directly, independent of central CPU admission and selected model device.
The core Stage-1--10 resilient runner also seeded before resolving the
requested device and remapped CPU-cloned checkpoints to CUDA before resume
compatibility preflight.

CLRL main now binds common checkpoint CUDA RNG to the selected execution
device, resolves ordinary and resilient placement before mutating Python,
NumPy or Torch RNG, rejects CPU/CUDA RNG mismatches before learner-state
restore, and loads the CPU-cloned checkpoint on CPU until schema, identity and
backend-RNG compatibility are proven. Source/test receipts and limitations are
recorded in
`docs/cl_native_cuda_admission_audit_2026-09-28.md`, latest documentation
commit `5bb68443531b717328ffb7f7ba370524d6dafa63`.

The private CL device-contract workflow continues to fail before runner
assignment, so these are source-level fixes, **not** executed CUDA/resume
evidence. Full v89 family reachability, physical GPU exact-resume, OOM
recovery and CPU/GPU numerical parity remain OPEN.


#### Forest_Run executed non-ML authority and Android regression closure

Forest_Run's no-training authority was strengthened so an empty or partial
Android tree cannot be misclassified as a clean non-ML repository. The
certificate now requires canonical build/manifest/entrypoint source,
production Kotlin source, and Android dependencies in addition to the
training-marker scan. Regression tests cover empty/partial trees,
dependency-free fixtures, real ML markers and ordinary application vocabulary.

Executed evidence from the corrected source line:
- training-control applicability run `36531498981`: SUCCESS;
- local estate certificate run `36531498984`: SUCCESS;
- Android validation run `36531499165`: SUCCESS for the host/release/lint
  job and API-35 connected smoke/deterministic-evidence job.

The Android workflow initially exposed one stale unit assertion: after
`resetRun()`, the test expected the historical `Find The Stride` cue,
while the canonical authored opening now guarantees Duck first and teaches
`Duck The Low Flyer`. The test was corrected at
`76401c71cb7566e7c55e05010999fd9ad7e94d04`; gameplay behavior was
unchanged. Detailed evidence is recorded at
`Forest_Run/docs/audits/2026-09-29_training_control_and_opening_reset_validation.md`
(commit `573ffaa1428ca684c7a4d6762ffcbdfa217b469e`).

This closes the current repository's **ML-training applicability** question
and the discovered Android unit regression. It does not close physical-device
matrix, Play delivery, signing, human/accessibility, creative/legal/privacy or
production-release acceptance.


#### Forest_Run no-training applicability authority — executed closure for training-control scope

Forest_Run is not an ML/training repository: its retained product is a native
Kotlin Android `SurfaceView`/Canvas game. CUDA-first training, CuPy,
optimizer checkpoints, model/dataset registries and CPU/GPU training parity
remain **not applicable** and must not be fabricated. The repository-specific
authority was hardened so an empty or partial checkout cannot be certified as
"no retained trainable surface": required Android build/manifest/entrypoint
files, production Kotlin source and real Android dependencies must all be
present before the certificate can pass. Synthetic regressions now prove that
an empty tree, missing application/build files and an injected real ML marker
fail closed.

Current `Forest_Run/main` at
`3b6e584e9bc7669aaeebf44e2e62f1f0eb198e09` produced **executed**
GitHub Actions evidence:
- Training-control applicability audit run `36533481327`: success, including
  compile, absent-tree/injected-marker regressions and certificate emission.
- Estate local training-control run `36533481354`: success.
- Android validation run `36533481357`: success for host release/lint/
  packaging and API-35 connected deterministic-smoke jobs.

This closes Forest_Run's **training-control applicability classification**:
there is no retained ML training surface on the validated source candidate.
It does **not** certify external Play Console submission, production signing,
human artistic/accessibility acceptance, physical-device matrix completion,
or other release gates already documented by the repository.


#### 2026-10-01 Text/Emotion inference-only GPU paths and CO-project non-ML closure

**Text-and-Emotion-Analysis-Tool-with-Visualization** now has executed current-head
evidence for its inference-only classification. The retained Transformer models
remain pretrained inference only; the source authority scans Python, TXT and
Markdown fenced code while ignoring Markdown prose signatures. Both retained
sentiment application copies are byte-identical. In addition to the existing
Torch CUDA allocation/operation/synchronization selector, preprocessing now
optionally installs `cudf.pandas` before pandas import after a usable CuPy
probe and calls `spacy.prefer_gpu()` before loading `en_core_web_sm`, while
all CPU-admission aliases preempt those optional accelerator probes.
Strict v41 run `36879845905` and estate certificate run
`36879845917` both succeeded. Physical CUDA/cuDF/spaCy execution, model
output parity and dependency locking remain OPEN.

**CO-project** was independently classified as a pure Python assembler/simulator,
not an ML repository. Its no-training authority now requires both
`Assembler.py` and `Simulator.py` and fails on empty/partial source or
injected training primitives. Training-only native/exact-resume and generic
workload-registry requirements were removed from the non-training audit profile;
strict retained-source and semantic training checks remain. OPF strict audit
run `36880314345` and estate certificate run `36880314216` both
succeeded. This closes only the training/CUDA applicability question, not
ordinary assembler/simulator functional correctness.


#### Forest_Run non-ML training-control applicability — executed closure evidence

Forest_Run is a repository-specific **not-applicable** case for ML training:
it is a native Kotlin/Android SurfaceView game with no retained optimizer,
model-training, dataset or ML-framework surface. The previous authority was
strengthened so an empty/partial checkout can no longer be mistaken for
evidence of no training. Required Android build/application files, production
Kotlin source and retained Android dependencies must exist before a certificate
can pass; ML/training markers still fail closed.

Current-head GitHub Actions on
`3e1231b2bf3d7d1295fbb7beeba625812499f2b6` now provide actual executed
evidence rather than source-only intent:
- training-control applicability audit `36881190063`: **success**;
- estate local training-control certificate `36881190024`: **success**;
- Android validation `36881190034`: **success**.
The applicability job ran 16 tests successfully and emitted
`no_retained_trainable_surface` with zero findings over 639 scanned
source/config files. The certificate explicitly reports
`ml_training_applicable=false` and `execution_claim_emitted=false`.

Forest_Run's ML/CUDA-training applicability is therefore **CLOSED as not
applicable for the retained software**, not falsely counted as a CUDA training
implementation. This does not close physical-device, Play/store, human,
licensing or production-release acceptance. Detailed evidence:
`Forest_Run/docs/audits/2026-10-01_training_control_non_ml_closure.md`
(commit `eb8976383079c86084b56e9f1193321011950d2a`).


#### Forest_Run no-training authority hardened and executed — 2026-10-02

Forest_Run is correctly classified separately from the 38 ML/training launchers:
it is a native Kotlin Android game and must not receive fabricated CUDA/model
training jobs. Its repository-specific fail-closed authority was strengthened
to require the actual Android source/build surface; scan notebooks, shell and
dependency manifests; detect dynamic framework imports and retained model
artifacts; reject uninspected executable archives/native libraries except the
explicit Gradle wrapper JAR; and bind source bytes, relevant path inventory,
and the exact local certificate-authority code with independent SHA-256
digests. Product assets can no longer hide a future model artifact merely
because the historical artwork directory is excluded from text scanning.

GitHub Actions run `37015111339` on Forest_Run commit
`e65224fbf111b4319da7fbace28a0c7f6f86042b` executed the focused
applicability suite successfully: **30 tests, OK**. Estate-local certificate
run `37015110857` also succeeded on that commit. The detailed source and
evidence boundary is recorded in
`Forest_Run/docs/audits/2026-10-02_training_control_no_ml_authority.md`
(commit `b94b18dc957b72254e0e2898985c1db733d41b49`).

This closes the question "should account-wide CUDA training be added to
Forest_Run?" as **not applicable under the current retained source**, not as
a waiver for future ML additions. Android/release/device/store/human
acceptance remains independent, and the Android validation for the focused
certificate commit was still pending when the Forest_Run note was written.


#### Continual-Learning native OOM retry and transactional RNG restore

A deeper repository-local trace found two correctness gaps beyond the already
implemented CUDA allocation/admission policy.

First, `scripts/pressure_aware_catalog_runner.py` persisted
`force_cpu=1` after CUDA OOM whenever CPU fallback was configured. Under a
centrally GPU-admitted parent this created an unlaunchable queued job: CPU
execution was forbidden by admission while the persisted state suppressed
the intended GPU retry. Commit
`17999cae591fe73c77fb0bcfb551ea0a0b11ff13` now keeps GPU-admitted
OOM retries GPU-eligible under the reduced batch scale and clears stale
persisted CPU fallback after a scheduler restart under GPU admission.
`tests/test_pressure_runner_cuda_oom_admission.py` was added at
`35da470d0565554ddd4e9c710cc463f056b30fff` and wired into the focused
training-control workflow at
`c14d614ce778bf9350912c8cf3875d5be754e38b`.

Second, both the native BaseRunner and physical-cohort RNG restore paths
mutated Python/NumPy/Torch CPU RNG state before discovering a missing or
invalid CUDA RNG payload. Commits
`2dea5f9f1f206e784524812dd11d5491821a7841` and
`d09e0a348a1b7169e49ec1b4163eb90e2253d5b4` now validate CUDA state
before host mutation and roll back host/CUDA streams when a setter fails.
Regression coverage was expanded at
`383229bd506142f0b3b328ce8bf80b31c351b5a0`.
Repository-specific evidence and nonclaims are recorded in
`Continual-Learning/docs/cuda_exact_resume_audit_2026-09-29.md` at
`4d1446a01c39af6eee5414eab2482dbd553fd3b5`.

The most recent verification/audit jobs were queued or failed before an
assigned runner/recorded steps, so these new tests are **not** marked passed.
Physical CUDA OOM recovery, real interruption/resume under AMP and full
122,714-job execution remain OPEN.


#### continual-learning-with-rl transactional RNG restore closure extension

A deeper exact-resume trace found that the common CLRL RNG preflight validated
backend compatibility but still allowed a validly-shaped CUDA RNG payload to
fail during `torch.cuda.set_rng_state_all()` after Python/NumPy/Torch CPU
generators had already been changed. The shared core helper was made
transactional at `ffba5b113566c615c2f10a34aa0ba9d0ef235e45`, with focused
rollback regression at `e90212d262895fe3c892210cc05c536e67daa93a`.

The same non-transactional setter sequence existed independently in seven
specialized checkpoint loaders. New helper
`src/cl_exec/rng_transaction.py`
(`cf638835b5539c460f5b5d390cb41588f640a1f1`) centralizes snapshot/apply/
rollback semantics. Named CL, prompt CL, R20 replay augmentation, R14/R15/R16
task-free runtimes and v59 structural routing now retain their existing
device/cross-backend preflight but route actual RNG mutation through that
transaction. Integration commits:
`079d81ec5880615d3783ef519c5351c1befdddc0`,
`f1c0a729ae883f1746230858b503f285719fb2ac`,
`c2710127fe344adcf27998fd348171a87e0e8c04`,
`fb110207c882cb2b666fa8c7b95bc2eddfc3ddd7`,
`ae1c6fcab12f7238dc5301a30d087c04305952df`,
`eb601932b4083ca4bcee3fbfdc02e2839955ddcf`,
`3fc178e2836b92326f6a0291e6dc6665ece01e3f`.
Focused source/rollback coverage:
`0da6965a5044b9c849019293041aa98716b842c3`;
workflow wiring:
`10856af009f2c05ef920ee2bcfcb641eb383f753`.
Repository audit updated at
`2fa5c45dcca4fc96f8324e1efec0d08904e358a1`.

These are checkpoint transaction corrections, not evidence of physical CUDA
resume or whole-repository closure. The private focused workflows still have a
history of zero-step/no-runner failures and must be executed successfully
before claiming runtime verification.


#### VaaniNoise-SED transactional CUDA RNG checkpoint restore

A repository-local checkpoint trace found that VaaniNoise already had a sound
central Torch device policy and CPU/CUDA resume compatibility preflight, but
`training/checkpoint.py` still applied Python, NumPy and Torch CPU RNG state
before its final CUDA RNG setter. A structurally valid but unusable CUDA
generator payload could therefore reject resume after host RNG streams had
changed.

Commit `d0453a439d91fc4f4e9005426102ece4d1681840` now snapshots Python,
NumPy, Torch CPU and selected CUDA RNG state and rolls the complete snapshot
back when any setter fails. Focused regression
`8d471b25b2d3e2492634ba80bdfc2af06efc6195` covers zero-mutation missing
CUDA state, simulated CUDA setter failure/rollback and CPU restore with no CUDA
RNG access. It is wired into both runtime-hardening Python lanes at
`f6bf52ebbe0fbd4a9676badbf44694895bdf71e7`.
Repository evidence/nonclaims:
`VaaniNoise-SED/docs/cuda_checkpoint_rng_audit_2026-10-03.md`
(commit `26f5d28118b785abb3faa4c33d05f33a11484fa7`).

The latest runtime-hardening, strict training-control and estate jobs at the
checked head had no assigned runners and zero executed steps, so no test pass,
physical CUDA resume, numerical parity or catalog-wide execution is claimed.

#### VaaniNoise-SED whole learner + mixed-source resume transaction extension

The earlier VaaniNoise CUDA RNG repair still left a larger exact-resume
transaction gap: model/optimizer/scheduler/scaler/EMA could commit before a
later component or RNG/post-load state rejected the checkpoint. Commit
`00d1da1e44e1e64ecd5cbb93331a8cde1b96536c` added a
spooled pre-resume learner snapshot; `ae58c8d9905fd2169cefcfb41aaf6665165012fd`
and `8677b402b529e763d2e768b7243d3c045dda3892` retain and
restore the original RNG state through the post-load transaction. The normal
fit runner now commits TrainingState/Group-DRO inside that boundary at
`3e7aac68a6884efb52f6de6ca37a6a3fb7d40f9a`. Focused learner,
RNG and robust-state rollback coverage reached
`1adb5410ee6490f78cca2710c414e510bd8f2ed9` and
`38849453255ab2e251224e8933f1b5a91d898cde`.

A separate M61 mixed-source loader also validated metadata, adapter
cursor/exposure state and optimizer counters after core checkpoint mutation.
The adapter is now locally transactional
(`51dce6555808ce287c933293729f6479c2f29f9f`) and the full
mixed-source restore is committed via the outer checkpoint post-load
transaction at `203b151fb073fe87096e5b6b1f0a6a2e8a974ec6`.
Regression `59bbd57941bfad967c2d84563dc70953ef926b17` asserts a
late counter mismatch restores model, optimizer, trainer state, adapter and
Python/NumPy/Torch RNG. Both transaction suites are wired into the 3.11/3.12
runtime-hardening matrix at
`c3b3a45f38f7e4a734576f1e72505e8e395f54ea`.
Repository evidence is updated at
`VaaniNoise-SED/docs/cuda_checkpoint_rng_audit_2026-10-03.md`
(commit `91f2278f8bc80ba3652242508e3d52326823b875`).

The latest VaaniNoise runtime-hardening run
`37129144718` and companion strict/estate runs still had no assigned
runners and zero executed steps. Thus source-level transaction closure is
stronger, but physical CUDA interruption/resume, AMP/OOM recovery and
catalog-wide numerical execution remain OPEN.


#### Forest_Run no-training classification — executed and verified

Forest_Run was deliberately **not** given synthetic CUDA or optimizer jobs. It
is a native Kotlin/Android game and its repository-specific authority classifies
ML training, training datasets, optimizer/checkpoint state and GPU-first training
as not applicable unless real trainable source is later introduced.

The fail-closed authority was strengthened so an empty, partial or dependency-
free Android tree cannot be mis-certified merely because there are no ML tokens
to find. The live certificate now requires the retained Gradle/application
entrypoints, production Kotlin source, and actual Android dependencies before it
can emit `no_retained_trainable_surface`. Synthetic fixtures still use the
lower-level composable scanner and injected framework/training markers fail the
authority closed.

This is now **executed evidence**, not source-only:
- current Forest_Run `main`: `eeccc677107f69039dbf71053c18a9038d4b5ff6`;
- Training-control applicability audit run `37053783933`: success;
- the fail-closed regression step ran **30 tests** and reported `OK`;
- the emitted certificate reports `finding_count=0`,
  `scanned_file_count=646`, `complete=true`,
  `ml_training_applicable=false`, and
  `classification=no_retained_trainable_surface`;
- Estate local training-control run `37053784236`: success;
- Android validation run `37053784090`: success.

This closes Forest_Run's **training/CUDA applicability** question. It does not
claim physical Android-device/store acceptance or replace the game's separate
release-evidence gates.


#### 2026-10-03 BTN central fleet receipt/mask hardening and Nutri PPO/checkpoint continuation

Breaking-the-Neural-Barrier's newer 18K/18L central CUDA layer was reviewed
against the exact pinned OPF scheduler. Two additional source contracts were
tightened. First, central GPU admission now validates a complete successful
fleet receipt: positive device_count, checked logical indices exactly
0..N-1, and no failure records. The bounded CUDA training-cell independently
rejects a forged success receipt with failures or missing reason metadata.
Second, the inherited CUDA visibility mask is checked against the pinned
scheduler's actual selection semantics. Single-device UUID masks remain valid,
while duplicate or multi-device non-decimal masks cannot be called GPU-usable:
automatic mode falls back to CPU before probing and explicit GPU admission
fails closed. Multi-device numeric masks continue only after full Torch fleet
proof. Principal commits: 125f3b6b036c77656c326a89c176ea4da1e4d169,
f9ab528dd4db46e12b6085526f1b1dcd16cd446c,
f8a0ff756de68f575aa3b945eb5dda01849405f6,
3230a9f85238aaf00a576da9fd2bb3e3375d5cc0,
392d76e75bdc9df3a60ed8b038b286f240823b48,
3ab95316b3242f0640b54f6020b8c2c772190f45.
The matching 18K/18L workflow jobs still end before runner assignment, so
these new regressions are source-only evidence and not a physical CUDA pass.

NutriFlavorOS's retained PPO path now bootstraps non-terminal truncated tails
from an explicit next state before optimizer mutation, caps on-policy batches
at exactly 32 samples, uses process-stable BLAKE2b state features, validates
the documented encoder dimension, and rolls actor/critic weights back after a
failed checkpoint application. Torch model restoration was centralized behind
a weights-only loader that refuses unsafe compatibility fallback. Governed
research checkpoints moved to schema v2 with primitive Python/NumPy RNG state
and save-time safe-tree validation, allowing optimizer/scheduler/scaler state
to use restricted Torch deserialization as well. Detailed receipts are in
NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md at
ef41641f0fe9193dd324505f6d8608aa6695f793. Current Nutri validation again
created zero-step jobs with no assigned runner, so remote tests, live exact
resume, physical CUDA and numerical parity remain OPEN.


#### NutriFlavorOS governed device-first RNG and exact step-resume hardening

A deeper governed-runtime review found that NutriFlavorOS still seeded Python,
NumPy and Torch before the requested training device had been admitted/probed.
An explicit CPU run could therefore touch CUDA RNG merely because CUDA was
visible. Commits
`16374c8c2e28d1881b02120bbfb56f5168815620` and
`6971237b04b5d5d3f4a1ddaca31d7c2af71a2675` now resolve the device
first and make the seed helper independently reject central admission
mismatches before any RNG mutation.

Checkpoint RNG state is now bound to the resolved device. CPU checkpoints do
not query CUDA RNG, device-type mismatches and missing CUDA RNG state fail
before learner mutation, and RNG restore is transactional across Python,
NumPy, Torch CPU and CUDA setters. Exact-resume preflight occurs before
model/optimizer/scheduler/scaler load. Principal commits:
`15f61cdd3598f0fa835ae2d396eba1fa5d7c9ab2`,
`35da4ba80c4b561559c63b3f08dbb0e6215a4a2b`,
`4d7d57973a60ee79a07a918fbc49e104d0c33613`.

Step checkpoints also now persist and restore partial-epoch loss, sample and
metric accumulators before the remaining loader batches. This prevents
restart-induced drift in epoch schedulers, best-checkpoint selection and early
stopping. Absolute completed-epoch reporting across resumes was corrected at
`261ac3fdecda6d2a92377644fb79d70d9174b20c`; source contract
`4f056406e09d0835e312ffdbc9d16bc441c5a93c` guards the resume
ordering and accumulator fields. Detailed evidence:
`NutriFlavorOS/docs/cuda_trainable_surface_audit_2026-09-28.md`
(commit `959d8cf2ce7aa5cc890d0bd8b722b85e000cc9ab`).

The private Nutri validation remains a zero-step/no-runner failure, so
physical CUDA resume, AMP/GradScaler restart, CUDA topology equivalence, real
stateful-loader interruption and uninterrupted-versus-resumed numerical parity
remain OPEN.


### 2026-10-04 executed Forest_Run applicability proof and BTN 19A CUDA receipt integrity

**Forest_Run.** The strengthened no-training authority is now backed by
executed current-main evidence rather than source-only reasoning. Current
`main` `eeccc677107f69039dbf71053c18a9038d4b5ff6` completed:
- Training-control applicability audit run `37053783933`: success, including
  compilation, the absent/partial Android-tree and injected-training-marker
  regressions, and the final no-retained-trainable-surface certificate.
- Estate local training-control certificate run `37053784236`: success.
- Android validation run `37053784090`: success for host/release/lint/package
  checks and API-35 connected smoke/deterministic evidence.

This closes the **ML-training applicability classification** for the current
Forest_Run source: inventing CUDA/optimizer jobs remains incorrect. It does not
replace the repository's documented physical-device, Play delivery, signing,
human/accessibility, or production-release evidence gates.

**Breaking-the-Neural-Barrier 19A runtime correction.** A later audit of the
18K/18L central CUDA proof found two evidence-integrity gaps. The policy receipt
was not cryptographically bound to the exact post-admission
`CUDA_VISIBLE_DEVICES` value, and a bounded multi-GPU smoke run could complete
an optimizer step on an earlier device then fail later while the CLI incorrectly
reported `training_executed=false`. Main now fingerprints the exact mask
without publishing device identifiers, rejects visibility drift before the
second fleet proof, and carries truthful per-device partial execution through
failure receipts. Machine contract:
`Breaking-the-Neural-Barrier/configs/central_cuda_receipt_integrity_19a.json`;
narrative:
`docs/BTNB_RUNTIME_CUDA_RECEIPT_INTEGRITY_19A_2026-10-04.md`.

Affected Git blobs are `2ee64faa6d8cf8fa591bc57665653ed081f424a0`
(policy), `2ebf9d0bbf3237cf572f7c98e54a29d8187c6ec2`
(training cell), `5d3dba523102929723e8374ab5c2d01be2fa43fc`
(policy tests), and `d49033c237a99771d5c8c9c6efd97dcb973659b6`
(fleet/cell tests). Actions run `37178663177` still had no assigned runner
and zero recorded steps, so these new BTN tests are not claimed as executed.
Physical CUDA, real multi-GPU pressure recovery, exact killed-process resume,
canonical neural training and paper parity remain OPEN.


### 2026-10-04 Forest_Run training-control applicability closure

Forest_Run remains a native Kotlin/Android game rather than an ML-training
repository, so GPU-first training, optimizer-state, training datasets and
CPU/GPU model parity are correctly **not applicable** rather than missing
features. The repository-specific authority has since been hardened beyond
the earlier empty-tree fix: it binds source/scope/authority SHA-256 manifests,
requires the real Android application/build entrypoints, scans executable
source/dependency formats and notebook code, catches literal and common
dynamic ML imports, detects retained serialized model formats and opaque
runtime artifacts, and fails on source symlinks or incomplete source trees.

Latest Forest_Run `main`:
`32ff32a36564ef3b184a7e7cc8410e159851c54b`.
On that exact SHA:
- Training-control applicability audit run `37183522969`: **success**,
  including compilation, absent-tree/injected-training regressions, live
  no-training certification and artifact upload.
- Estate local training-control certificate run `37183522981`: **success**.
- Android validation run `37183523012`: **success** for the host
  release/lint/packaging job **and** the API-35 connected emulator smoke and
  deterministic-evidence job.

Accordingly, Forest_Run's **training-control applicability classification is
CLOSED/PASS** on this source revision. This does not claim physical-device,
Play Store, human/artistic, signing, privacy-policy hosting or final release
acceptance; those are separate product/release gates documented by the repo.


#### 2026-10-04 Smart-Glasses checkpoint and exact-RNG CUDA follow-up

The native Smart-Glasses follow-up found that the generic
`checkpointing.py` utility and the exact-edge RNG engine still had
independent CUDA-admission logic after the earlier array/RAPIDS fixes.
Generic checkpoints now bind RNG to the model's exact resolved device,
prevalidate stochastic topology before model mutation, reject GPU-admitted
CPU models, and restore Python/NumPy/Torch RNG transactionally. The exact
engine now reuses `gpu_compat.gpu_admission_requested` and the executable
requested-device Torch probe rather than raw
`torch.cuda.is_available()`. The shared Torch probe itself accepts
`cuda:N`; ONNX/OpenCV capability caches are keyed to the scheduler/device
mask so they cannot reuse a stale provider result after a mask change.

Focused runtime/source regressions were added and wired to the existing
CPU-only-Torch GPU-admission workflow. Detailed bounded evidence is in
`Smart-Glasses/docs/gpu_acceleration.md`, commit
`7f2130287b6234a6b955a60e4b8423aeaeba0a8c`.
Latest Smart-Glasses workflow records still show no assigned runner and zero
steps for these jobs, so this is source-level closure only: no physical CUDA,
multi-GPU exact-resume, ONNX/OpenCV execution or numerical parity PASS is
claimed.


#### Forest_Run no-training classification — current-main closure

Forest_Run is an Android/Kotlin game rather than a retained ML-training
repository. Its current fail-closed applicability authority was re-audited
against the live `main` tree and strengthened beyond the earlier
required-file check. Current source
`training_control/forest_no_trainable_authority.py` (Git blob
`c7d92c79fd0d1e0cc7f566cc0c8609411de4baf9`) now binds source,
scope and authority manifests; inspects dependency manifests, shell/XML,
notebook code, dynamic Python import aliases and serialized-model formats;
and fails closed on unapproved opaque executable artifacts. The root
certificate clears stale PASS output before each audit and cross-checks
dataset/applicability certificates against the same bound manifests.

At Forest_Run commit
`32ff32a36564ef3b184a7e7cc8410e159851c54b`, GitHub Actions
successfully executed:
- training-control applicability audit run `37183522969`, including
  the source regressions and certificate publication;
- estate local training-control certificate run `37183522981`;
- full Android validation run `37183523012`.

Accordingly, **Forest_Run's training-control applicability question is
closed on that main commit**: no retained ML training surface is present,
so GPU-first model training, optimizer/resume, ML dataset cohorts and
CPU/GPU numerical training parity are not applicable and must not be
invented. This closure is limited to the training-control classification.
Physical Android-device acceptance, store delivery, signing, human
acceptance and other product-release gates remain governed separately by
the repository's existing evidence contracts.

#### BTN 20Q scheduler-mask alignment and bounded GPU-receipt authentication

A current-main BTN review found a composition defect between the local 18K
executable-CUDA admission and the literal vendored OPF v41 scheduler. 18K
could accept/prove a multi-device `CUDA_VISIBLE_DEVICES` mask, while the
pinned scheduler has one `--gpu-device-index` (default 0) and only accepts
a multi-token inherited mask when that decimal physical token is present.
The central layer could therefore emit a successful GPU receipt for a mask
the scheduler would later reject.

BTN now resolves the actual forwarded scheduler index in
`run_all_training.py`, narrows inherited or unrestricted visibility to
exactly one scheduler-usable token *before* the disposable Torch probe, and
binds both initial and selected visibility in the receipt. Multi-token masks
use local-index selection first with a legacy numeric physical-token
fallback. Multi-UUID/MIG masks are safe after one-token narrowing because
the pinned scheduler already accepts a single inherited token verbatim.
A successful selected-device proof must observe exactly one logical CUDA
device; forged multi-device receipts are rejected.

The 18L bounded CUDA preflight was also tightened. GPU mode now requires an
authentic `btnb.central_cuda_runtime.18k` receipt with executable Torch
CUDA, one bound visible token, one logical device, no failures, a valid
scheduler index, an explicit narrowing disposition, and an unchanged
visibility fingerprint before any second probe or optimizer step.

This source work is recorded as BTN four-regime tranche **20Q**, extending
the separate chain 310→311. The chain index now contains 221 unique deltas
with terminal 20Q; a direct current-source check confirmed the 20P
309→310 and 20Q 310→311 edge and
`tranche_source_governance_closed=false`. The chain auditor was corrected
to permit source-only regime closure while continuing to reject runtime-test,
strong-reference-parity, or repository-complete claims.

Exact source binding is in
`Breaking-the-Neural-Barrier/configs/central_cuda_scheduler_alignment_20q.json`;
design/limitations are in
`docs/BTNB_FOUR_REGIME_20Q_CENTRAL_CUDA_SCHEDULER_ALIGNMENT_2026-10-04.md`.
Focused workflow run `37187948525` again ended before runner assignment
with zero steps. Therefore 20Q remains verification-pending: no physical
CUDA run, real training, CPU/GPU parity, multi-GPU throughput, or
checkpoint/resume execution is claimed.


### 2026-10-04 Forest_Run no-training authority self-scan closure

Forest_Run remains a native Kotlin/Android game with **no retained ML training
surface**. Its correct estate behavior is a fail-closed applicability authority,
not synthetic CUDA/optimizer jobs.

A later audit found that the earlier authority semantically skipped the root
`run_all_training.py` and the entire `training_control/` directory. Known
authority files were hash-bound, but a newly introduced source in that
directory could evade semantic ML scanning while receiving a fresh digest.
Forest_Run commit `32f502d5852d632b981f6b9aef82b6e18efc42bc`
removed those exemptions while retaining Python string/comment masking so
policy prose may name frameworks without producing false positives. Commit
`511958d79d26f4b215ad0c53f4a00755061f0074` added regressions for
root-entrypoint PyTorch, hidden training-control TensorFlow, model artifacts
under the authority directory and non-executable policy strings.

**Executed evidence:** GitHub Actions run `37190394158` passed on the exact
commit. Its applicability step ran 39 tests and reported OK; certificate
generation and artifact upload succeeded. The emitted certificate reported
649 scanned source/configuration files, zero findings, eight Android dependency
declarations and `classification=no_retained_trainable_surface`.
Estate-local certificate run `37190394188` also passed on the same commit.
The exact source/scope/authority digests and nonclaims are recorded in
`Forest_Run/docs/audits/2026-10-04_training_control_self_scan_closure.md`
(commit `0ef76e7bb418e408dbbd82bd773fd3897cb3fe94`).

Accordingly, **Forest_Run repository-level ML training applicability is CLOSED
as not applicable for that source snapshot**. This does not close gameplay,
physical-device, store, signing, human-acceptance or Android release gates,
and does not grandfather future source changes.


#### Forest_Run no-training applicability closure — executed 2026-10-04

Forest_Run is now closed for the **training-control applicability** question.
It is a native Kotlin/Android game and the correct result is an executed,
fail-closed `no_retained_trainable_surface` certificate rather than fabricated
ML/GPU training jobs.

The authority now scans the root launcher and repository-owned
`training_control/` code, binds source/scope/authority manifests, rejects
missing Android app/build authority, opaque executable/model artifacts and
out-of-tree source symlinks, and emits a failure certificate instead of leaving
an earlier PASS behind after a later failed audit.

Exact-current-main evidence:
- Forest_Run commit `0ef76e7bb418e408dbbd82bd773fd3897cb3fe94`;
- Training-control applicability run `37190475490`: SUCCESS;
- 39 fail-closed applicability regressions: SUCCESS / `OK`;
- 649 scanned source/configuration files, zero findings;
- eight Android dependency declarations;
- source manifest SHA-256
  `54307acf7765c6a8f7c70316d0598ebd64e8af7993b01cac7a8415cbd73ca2cf`;
- scope manifest SHA-256
  `cfc7419d605cebd60d59b97293ea76fb2cffc8791094ff1fba670f177d8aac71`;
- authority manifest SHA-256
  `d304ecb7e5b80124063b1524451b10169d5ec9d89dc9d614152fb3ba05b77099`;
- Estate local training-control run `37190475487`: SUCCESS;
- Android validation run `37190475482`: SUCCESS.

Thus GPU-first ML training, optimizer/loss/early-stopping state, training
checkpoint/resume, CPU/GPU training parity, distillation and architecture
matrices are **not applicable** to the current Forest_Run source and should
not be added. This does not certify release readiness, physical-device/store
evidence or human acceptance; those remain governed by Forest_Run's separate
Android/release evidence system.
