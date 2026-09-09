#!/usr/bin/env python3
"""Cross-repository training controller using the literal OPF_ADP scheduler.

The training job catalog is repository-specific. Scheduling is not: after the
catalog is compiled into OPF JobSpec objects, the original pinned
utils/opf_massive_suite_runner.py executes unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

OPF_REFERENCE_REPOSITORY = "Anurag9000/OPF_ADP"
OPF_REFERENCE_COMMIT = "a3c41f7c25f21977f1ff33e94a65b6450afabee9"
OPF_RAW_ROOT = f"https://raw.githubusercontent.com/{OPF_REFERENCE_REPOSITORY}/{OPF_REFERENCE_COMMIT}"
OPF_RUNTIME_BLOBS = {
    "utils/opf_massive_suite_runner.py": "314dc390955e54c7ca35589e3008068155f9fb44",
    "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
    "utils/ml_backends.py": "2fe2b24e530cab3d747c983c4457f4080703512f",
    "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
    "utils/opf_shared_defaults.py": "76ad434ecef1f708c835210d4bc86e0717999d99",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "dacb9a2c44d611c045fbb7512ba5327343f79a85",
}
OPF_RUNTIME_FILES = tuple(OPF_RUNTIME_BLOBS)
CONTROLLER_SCHEMA = 2
TRAINING_NAME_RE = re.compile(
    r"(?:^|[_\-.])(train|training|finetune|fine[_-]?tune|fit|experiment|experiments|sweep|search|optimi[sz]e|continual|adapt|ppo|dqn|reinforce|policy|rl)(?:[_\-.]|$)",
    re.IGNORECASE,
)
TRAINING_BODY_RE = re.compile(
    r"(optimizer\s*=|torch\.optim|\.backward\s*\(|\.fit\s*\(|\bTrainer\s*\(|model\.train\s*\(|loss\.backward\s*\(|\bPPO\s*\(|\bDQN\s*\(|\bREINFORCE\b|\blearn\s*\(|train_one_epoch|training_step)",
    re.IGNORECASE,
)
SKIP_PARTS = {".git", ".training_control", "__pycache__", ".venv", "venv", "env", "node_modules", "results", "result", "outputs", "output", "checkpoints", "artifacts", "dist", "build", "docs", "doc"}
SCRIPT_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd"}


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _verify_reference_file(relative: str, data: bytes) -> None:
    expected = OPF_RUNTIME_BLOBS[relative]
    actual = _git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(
            f"Pinned OPF reference blob mismatch for {relative}: "
            f"expected {expected}, got {actual}"
        )


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-")
    return text or "job"


def _load_profile() -> Dict[str, Any]:
    raw = os.environ.get("TRAINING_CONTROL_PROFILE", "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise SystemExit("TRAINING_CONTROL_PROFILE must be a JSON object")
    return value


def _repo_root() -> Path:
    explicit = os.environ.get("TRAINING_CONTROL_REPO_ROOT", "").strip()
    return (Path(explicit) if explicit else Path.cwd()).resolve()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-parity-training-controller/2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _prepare_opf_runtime(root: Path) -> Path:
    cache = root / ".training_control" / "opf_reference" / OPF_REFERENCE_COMMIT
    marker = cache / "REFERENCE.json"
    expected_marker = {"repository": OPF_REFERENCE_REPOSITORY, "commit": OPF_REFERENCE_COMMIT, "files": OPF_RUNTIME_BLOBS}
    if marker.is_file():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            current = None
        if current == expected_marker and all((cache / rel).is_file() for rel in OPF_RUNTIME_FILES):
            cache_valid = True
            for relative in OPF_RUNTIME_FILES:
                try:
                    _verify_reference_file(relative, (cache / relative).read_bytes())
                except Exception:
                    cache_valid = False
                    break
            if cache_valid:
                return cache
    for relative in OPF_RUNTIME_FILES:
        destination = cache / relative
        data = _download(f"{OPF_RAW_ROOT}/{relative}")
        _verify_reference_file(relative, data)
        _write_atomic(destination, data)
    for relative in (
        "utils/__init__.py",
        "DNN/__init__.py",
        "DNN/VANILLA/__init__.py",
        "DNN/VANILLA/Dyn_DNN4OPF/__init__.py",
        "DNN/VANILLA/Dyn_DNN4OPF/utils/__init__.py",
    ):
        path = cache / relative
        if not path.exists():
            _write_atomic(path, b"")
    _write_atomic(marker, (json.dumps(expected_marker, indent=2, sort_keys=True) + "\n").encode())
    return cache


def _import_opf_scheduler(cache: Path):
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils.") or name == "Dyn_DNN4OPF" or name.startswith("Dyn_DNN4OPF."):
            del sys.modules[name]
    sys.path.insert(0, str(cache))
    sys.path.insert(0, str(cache / "DNN" / "VANILLA"))
    path = cache / "utils" / "opf_massive_suite_runner.py"
    spec = importlib.util.spec_from_file_location("opf_reference_massive_suite_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import OPF reference scheduler from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _command_for_path(root: Path, relative: str) -> List[str]:
    path = (root / relative).resolve()
    suffix = path.suffix.lower()
    if suffix == ".py":
        return [sys.executable, str(path)]
    if suffix == ".sh":
        return [shutil.which("bash") or shutil.which("sh") or "bash", str(path)]
    if suffix == ".ps1":
        return [shutil.which("pwsh") or shutil.which("powershell") or "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)]
    if suffix in {".bat", ".cmd"}:
        return ["cmd.exe", "/d", "/c", str(path)]
    return [str(path)]


def _normalize_command(root: Path, value: Any) -> List[str]:
    if isinstance(value, str):
        parts = shlex.split(value, posix=(os.name != "nt"))
    elif isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        raise ValueError(f"Unsupported command value: {value!r}")
    if not parts:
        raise ValueError("Training job command may not be empty")
    first = parts[0]
    candidate = root / first
    if not os.path.isabs(first) and candidate.exists():
        if len(parts) == 1:
            return _command_for_path(root, first)
        if candidate.suffix.lower() == ".py":
            return [sys.executable, str(candidate.resolve()), *parts[1:]]
        if candidate.suffix.lower() == ".sh":
            return [shutil.which("bash") or "bash", str(candidate.resolve()), *parts[1:]]
        if candidate.suffix.lower() == ".ps1":
            return [shutil.which("pwsh") or shutil.which("powershell") or "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(candidate.resolve()), *parts[1:]]
    return parts


def _dedupe_jobs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    answer: List[Dict[str, Any]] = []
    seen: set[Tuple[str, ...]] = set()
    for record in records:
        key = tuple(str(part) for part in record["command"])
        if key not in seen:
            seen.add(key)
            answer.append(record)
    return answer


def _job_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = profile.get("jobs")
    records: List[Dict[str, Any]] = []
    if isinstance(explicit, list):
        for index, item in enumerate(explicit):
            if not isinstance(item, dict):
                raise SystemExit(f"profile.jobs[{index}] must be an object")
            command_value = item.get("command") or item.get("entrypoint")
            if not command_value:
                raise SystemExit(f"profile.jobs[{index}] has no command/entrypoint")
            records.append({
                "id": str(item.get("id") or item.get("name") or f"job-{index:04d}"),
                "command": _normalize_command(root, command_value),
                "device_capable": bool(item.get("device_capable", True)),
                "phase": str(item.get("phase") or "training"),
                "family": str(item.get("family") or "generic"),
                "repeat_index": int(item.get("repeat_index", 0)),
            })
        return _dedupe_jobs(records)
    for relative in profile.get("preferred_training_entrypoints", []) or []:
        relative = str(relative)
        if (root / relative).is_file():
            records.append({"id": relative, "command": _command_for_path(root, relative), "device_capable": True, "phase": "training", "family": "entrypoint", "repeat_index": 0})
    for index, item in enumerate(profile.get("extra_jobs", []) or []):
        if isinstance(item, str):
            records.append({"id": item, "command": _normalize_command(root, item), "device_capable": True, "phase": "training", "family": "extra", "repeat_index": 0})
        elif isinstance(item, dict):
            command_value = item.get("command") or item.get("entrypoint")
            if not command_value:
                raise SystemExit(f"profile.extra_jobs[{index}] has no command/entrypoint")
            records.append({
                "id": str(item.get("id") or item.get("name") or f"extra-{index:04d}"),
                "command": _normalize_command(root, command_value),
                "device_capable": bool(item.get("device_capable", True)),
                "phase": str(item.get("phase") or "training"),
                "family": str(item.get("family") or "extra"),
                "repeat_index": int(item.get("repeat_index", 0)),
            })
        else:
            raise SystemExit(f"Unsupported extra job at index {index}")
    return _dedupe_jobs(records)


def _covered_paths(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> set[str]:
    covered = {str(item).replace("\\", "/") for item in (profile.get("ignore_entrypoints", []) or [])}
    covered.update(str(item).replace("\\", "/") for item in (profile.get("dynamic_registry_covers", []) or []))
    for job in jobs:
        for part in job["command"]:
            try:
                path = Path(part)
                if path.is_absolute():
                    covered.add(path.resolve().relative_to(root).as_posix())
                elif (root / path).exists():
                    covered.add(path.as_posix())
            except Exception:
                pass
    for item in profile.get("preferred_dataset_entrypoints", []) or []:
        if (root / str(item)).is_file():
            covered.add(str(item).replace("\\", "/"))
    return covered


def _candidate_training_entrypoints(root: Path) -> List[str]:
    candidates: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part.lower() in SKIP_PARTS for part in relative.parts):
            continue
        rel = relative.as_posix()
        if rel == "run_all_training.py":
            continue
        name_hit = bool(TRAINING_NAME_RE.search(path.stem))
        body_hit = False
        executable_hit = path.suffix.lower() != ".py"
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        if path.suffix.lower() == ".py":
            body_hit = bool(TRAINING_BODY_RE.search(text))
            executable_hit = "__name__" in text and "__main__" in text
            candidate = executable_hit and (name_hit or body_hit)
        else:
            body_hit = bool(re.search(r"\bpython(?:3)?\b|\btorchrun\b|\baccelerate\b", text, re.I))
            candidate = name_hit and body_hit
        if candidate:
            candidates.append(rel)
    return sorted(set(candidates))


def _coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = _candidate_training_entrypoints(root)
    covered = _covered_paths(root, profile, jobs)
    missing = [path for path in candidates if path not in covered]
    preferred = [str(x).replace("\\", "/") for x in profile.get("preferred_training_entrypoints", []) or []]
    stale_preferred = [path for path in preferred if not (root / path).is_file()]
    explicit_missing: List[str] = []
    for job in profile.get("jobs", []) or []:
        if isinstance(job, dict) and job.get("entrypoint") and not (root / str(job["entrypoint"])).is_file():
            explicit_missing.append(str(job["entrypoint"]))
    return {
        "schema": CONTROLLER_SCHEMA,
        "repository": profile.get("repository"),
        "opf_reference_repository": OPF_REFERENCE_REPOSITORY,
        "opf_reference_commit": OPF_REFERENCE_COMMIT,
        "compiled_job_count": len(jobs),
        "compiled_jobs": [{"id": job["id"], "command": job["command"], "device_capable": job["device_capable"], "phase": job["phase"], "family": job["family"]} for job in jobs],
        "training_candidates": candidates,
        "uncovered_training_candidates": missing,
        "stale_preferred_training_entrypoints": stale_preferred,
        "missing_explicit_entrypoints": explicit_missing,
        "coverage_ok": not missing and not explicit_missing,
    }


def _run_setup(root: Path, profile: Dict[str, Any]) -> None:
    if os.environ.get("TRAINING_CONTROL_SKIP_SETUP") == "1":
        return
    commands = [_normalize_command(root, value) for value in (profile.get("setup_commands", []) or [])]
    if not commands:
        for relative in profile.get("preferred_dataset_entrypoints", []) or []:
            if (root / str(relative)).is_file():
                commands.append(_command_for_path(root, str(relative)))
    seen: set[Tuple[str, ...]] = set()
    for command in commands:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        print("[training-control] setup:", shlex.join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)


def _custom_flags(argv: Sequence[str]) -> Tuple[Dict[str, bool], List[str]]:
    flags = {"audit": False, "list_jobs": False, "skip_setup": False, "allow_uncovered": False}
    forwarded: List[str] = []
    for arg in argv:
        if arg == "--audit-training-coverage": flags["audit"] = True
        elif arg == "--list-training-jobs": flags["list_jobs"] = True
        elif arg == "--skip-setup": flags["skip_setup"] = True
        elif arg == "--allow-uncovered-training": flags["allow_uncovered"] = True
        else: forwarded.append(arg)
    return flags, forwarded


def _ensure_opf_cli(argv: Sequence[str]) -> List[str]:
    answer = list(argv)
    if "--mode" not in answer: answer = ["--mode", "massive", *answer]
    if "--results-dir" not in answer: answer.extend(["--results-dir", "Results/training_control"])
    if "--post-launch-sample-delay-sec" not in answer: answer.extend(["--post-launch-sample-delay-sec", "60"])
    if "--scheduler" not in answer: answer.extend(["--scheduler", "gpu_first"])
    return answer


def _builder_factory(root: Path, profile: Dict[str, Any], records: Sequence[Dict[str, Any]], opf):
    repository = str(profile.get("repository") or root.name)
    repo_slug = _slug(repository.split("/")[-1])
    def build_suite_jobs(args, run_root, results_root):
        specs = []
        for record in records:
            label = _slug(record["id"])
            phase = _slug(record.get("phase") or "training")
            family = _slug(record.get("family") or repo_slug)
            repeat_index = int(record.get("repeat_index", 0))
            case_name = f"{repo_slug}:{label}"
            seed = opf.derive_suite_seed(int(args.seed), mode=str(args.mode), case_name=case_name, repeat_index=repeat_index, phase=phase, family=family, head_variant="single_head", depth=0, width=0)
            job_id = opf.build_job_id(phase, case_name, repeat_index, label)
            result_dir = Path(results_root) / "generic_jobs" / label
            specs.append(opf.JobSpec(job_id=job_id, phase=phase, family=family, case_name=case_name, repeat_index=repeat_index, depth=0, width=0, seed=int(seed), results_dir=str(result_dir), command=tuple(record["command"]), device_capable=bool(record.get("device_capable", True)), head_variant="single_head"))
        start = max(0, int(args.job_start_index))
        specs = specs[start:]
        if int(args.job_limit) > 0:
            specs = specs[:int(args.job_limit)]
        return specs
    return build_suite_jobs


def main(argv: Sequence[str] | None = None) -> int:
    root = _repo_root()
    profile = _load_profile()
    flags, forwarded = _custom_flags(list(sys.argv[1:] if argv is None else argv))
    if flags["skip_setup"]:
        os.environ["TRAINING_CONTROL_SKIP_SETUP"] = "1"
    records = _job_records(root, profile)
    report = _coverage_report(root, profile, records)
    report_path = root / ".training_control" / "coverage_report.json"
    _write_atomic(report_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    if flags["audit"] or flags["list_jobs"]:
        print(json.dumps(report, indent=2, sort_keys=True))
    if flags["audit"]:
        return 0 if report["coverage_ok"] else 2
    strict = bool(profile.get("strict_coverage", True)) and not flags["allow_uncovered"]
    if strict and not report["coverage_ok"]:
        raise SystemExit(f"Training coverage audit failed. See {report_path}. Use --allow-uncovered-training only for diagnosis.")
    if not records:
        print("[training-control] no trainable jobs declared/discovered; nothing to schedule")
        return 0
    _run_setup(root, profile)
    cache = _prepare_opf_runtime(root)
    opf = _import_opf_scheduler(cache)
    opf.build_suite_jobs = _builder_factory(root, profile, records, opf)
    sys.argv = [str(Path(__file__).resolve()), *_ensure_opf_cli(forwarded)]
    opf.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
