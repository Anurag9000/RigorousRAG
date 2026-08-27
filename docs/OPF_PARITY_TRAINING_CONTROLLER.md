# OPF-parity cross-repository training controller

`tools/universal_training_controller.py` deliberately does **not** implement an independent pressure scheduler. It downloads the scheduler/runtime files from the immutable `Anurag9000/OPF_ADP` commit `a3c41f7c25f21977f1ff33e94a65b6450afabee9`, verifies each downloaded file against its Git blob ID, imports the original `utils/opf_massive_suite_runner.py`, and replaces only `build_suite_jobs(...)` with a repository job-catalog adapter.

The consequence is intentional: OPF's live scheduling semantics—including its quirks—remain the scheduling oracle. Admission gates, RAM/swap/VRAM sensing, latest-child predictive packing, victim ranking, launch cadence, dynamic concurrency, retry behavior, CUDA-OOM CPU fallback, process-tree supervision, persistent state, manifest handling and runtime tuning are executed by the pinned OPF source rather than a reimplementation.

Repository wrappers must provide an explicit `TRAINING_CONTROL_PROFILE`. Prefer `jobs` for audited repositories. Each job has an `id` and `entrypoint` or `command`, and may set `device_capable`, `phase`, `family`, and `repeat_index`. The controller also supports the previous `preferred_training_entrypoints`/`extra_jobs` profile shape while repositories are migrated.

Before training, the controller scans executable scripts for training signatures and writes `.training_control/coverage_report.json`. Strict coverage is on by default. An unaccounted executable trainer blocks a normal run rather than being silently omitted. `--audit-training-coverage` performs the audit without downloading/importing OPF or launching training. `--list-training-jobs` prints the compiled catalog. `--allow-uncovered-training` is diagnostic-only.

Dataset/setup entrypoints are executed once before the model scheduler when present. Model jobs themselves are scheduled only by the pinned OPF scheduler.

The generic controller defaults to OPF's live wrapper policy of `gpu_first` and a 60-second post-launch sampling delay unless the caller explicitly supplies the corresponding OPF knob. All other scheduler knobs are parsed and enforced by the original OPF parser and loop.
