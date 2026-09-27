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
