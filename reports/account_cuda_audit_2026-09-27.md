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
