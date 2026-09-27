#!/usr/bin/env python3
"""Current cross-repository training controller.

This module deliberately does not fork the OPF_ADP scheduler. It imports the
existing universal adapter and repins its scheduler cache to an exact OPF_ADP
commit/blob set, then strengthens repository coverage, reachability, resumability,
and job-DAG auditing.

Scheduling semantics remain the literal pinned OPF_ADP implementation.  This
layer only discovers/audits repository-specific work and preserves metadata that
is not part of OPF's JobSpec.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shlex
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

import universal_training_controller as base

OPF_REFERENCE_COMMIT = "a34c31259bd5d5f58081e3766918f9df63017455"
OPF_RUNTIME_BLOBS = {
    "utils/opf_massive_suite_runner.py": "b97d47499c83bc6ed3a5753f7f3009b624c94868",
    "utils/runtime_tuning.py": "f1cbfc44e009701a5540a046f2cd6b9f41f16b74",
    "utils/ml_backends.py": "2fe2b24e530cab3d747c983c4457f4080703512f",
    "utils/logging_utils.py": "482ba94643aa921f49eebb835f29cf4930bb2498",
    "utils/opf_shared_defaults.py": "76ad434ecef1f708c835210d4bc86e0717999d99",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py": "dacb9a2c44d611c045fbb7512ba5327343f79a85",
}
AUDIT_SCHEMA = 5
SOURCE_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd", ".ipynb", ".r", ".jl"}
PYTHONISH_SUFFIXES = {".py", ".ipynb"}
EXEC_SCRIPT_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd", ".r", ".jl"}
SKIP = {
    ".git", ".training_control", "__pycache__", ".venv", "venv", "env",
    "node_modules", "results", "result", "outputs", "output",
    "checkpoints", "artifacts", "dist", "build",
}
TRAIN_PATTERNS = (
    re.compile(r"\btorch\.optim\b|\bloss\.backward\s*\(|\.backward\s*\(", re.I),
    re.compile(r"\b(?:model|estimator|pipeline|clf|regressor)\.fit\s*\(", re.I),
    re.compile(r"\bTrainer\s*\(|\btraining_step\s*\(|\btrain_one_epoch\b", re.I),
    re.compile(r"\b(?:PPO|DQN|A2C|SAC|TD3|REINFORCE)\s*\(|\.learn\s*\(", re.I),
    re.compile(r"\boptuna\b|\bGridSearchCV\b|\bRandomizedSearchCV\b", re.I),
)
MODEL_PATTERNS = (
    re.compile(r"class\s+\w+\s*\([^)]*(?:nn\.Module|torch\.nn\.Module|LightningModule)", re.I),
    re.compile(r"\b(?:Sequential|Functional|Model)\s*\(", re.I),
    re.compile(r"\b(?:RandomForest|XGB|LGBM|CatBoost|SVC|SVR|LogisticRegression|LinearRegression)\w*\s*\(", re.I),
)
CHECKPOINT_WRITE = re.compile(
    r"torch\.save\s*\(|save_checkpoint\s*\(|ModelCheckpoint|save_pretrained\s*\(|"
    r"checkpoint.*write|joblib\.dump\s*\(|pickle\.dump\s*\(", re.I,
)
CHECKPOINT_READ = re.compile(
    r"torch\.load\s*\(|load_state_dict\s*\(|load_checkpoint\s*\(|"
    r"resume_from_checkpoint|from_pretrained\s*\(|joblib\.load\s*\(|pickle\.load\s*\(", re.I,
)
RESUME_TOKEN = re.compile(r"\bresume\b|start_epoch|initial_epoch|checkpoint_last|last_checkpoint", re.I)
EARLY_STOP_TOKEN = re.compile(r"early[_ -]?stopp|patience|stopping_rounds|EarlyStopping", re.I)

MODEL_STATE_SAVE = re.compile(r"(?:model|module|network|net)\.state_dict\s*\(", re.I)
MODEL_STATE_LOAD = re.compile(r"(?:model|module|network|net)\.load_state_dict\s*\(", re.I)
OPTIMIZER_USE = re.compile(r"\b(?:torch\.optim|optimizer\s*=|optim\.)", re.I)
OPTIMIZER_STATE_SAVE = re.compile(r"optimizer\.state_dict\s*\(", re.I)
OPTIMIZER_STATE_LOAD = re.compile(r"optimizer\.load_state_dict\s*\(", re.I)
SCHEDULER_USE = re.compile(r"\b(?:lr_scheduler|scheduler\s*=|scheduler\.step\s*\()", re.I)
SCHEDULER_STATE_SAVE = re.compile(r"scheduler\.state_dict\s*\(", re.I)
SCHEDULER_STATE_LOAD = re.compile(r"scheduler\.load_state_dict\s*\(", re.I)
SCALER_USE = re.compile(r"\b(?:GradScaler|scaler\.(?:scale|step|update)\s*\()", re.I)
SCALER_STATE_SAVE = re.compile(r"scaler\.state_dict\s*\(", re.I)
SCALER_STATE_LOAD = re.compile(r"scaler\.load_state_dict\s*\(", re.I)
RNG_USE = re.compile(r"\b(?:torch\.rand|torch\.randn|np\.random|numpy\.random|random\.)", re.I)
RNG_SAVE = re.compile(
    r"get_rng_state|cuda\.get_rng_state|np\.random\.get_state|numpy\.random\.get_state|random\.getstate", re.I,
)
RNG_LOAD = re.compile(
    r"set_rng_state|cuda\.set_rng_state|np\.random\.set_state|numpy\.random\.set_state|random\.setstate", re.I,
)
PROGRESS_SAVE = re.compile(
    r"['\"](?:epoch|step|global_step|iteration|iter|batch_idx)['\"]\s*:", re.I,
)
PROGRESS_LOAD = re.compile(
    r"(?:start_epoch|initial_epoch|global_step|resume_step|start_step)\s*=|"
    r"\[['\"](?:epoch|step|global_step|iteration|iter|batch_idx)['\"]\]|"
    r"\.get\(\s*['\"](?:epoch|step|global_step|iteration|iter|batch_idx)['\"]", re.I,
)

LOCAL_SCRIPT_TOKEN_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:py|sh|ps1|bat|cmd|r|jl))", re.I,
)
PYTHON_M_RE = re.compile(r"(?:^|\s)(?:python(?:3)?|py)\s+-m\s+([A-Za-z_][\w.]*)", re.I)


def _configure_reference() -> None:
    base.OPF_REFERENCE_COMMIT = OPF_REFERENCE_COMMIT
    base.OPF_RAW_ROOT = (
        f"https://raw.githubusercontent.com/{base.OPF_REFERENCE_REPOSITORY}/"
        f"{OPF_REFERENCE_COMMIT}"
    )
    base.OPF_RUNTIME_BLOBS = dict(OPF_RUNTIME_BLOBS)
    base.OPF_RUNTIME_FILES = tuple(OPF_RUNTIME_BLOBS)


def _iter_sources(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel = path.relative_to(root)
        if any(part.lower() in SKIP for part in rel.parts):
            continue
        yield path


def _read_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".ipynb":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            cells = payload.get("cells", []) if isinstance(payload, dict) else []
            return "\n".join(
                "".join(cell.get("source", []))
                for cell in cells
                if isinstance(cell, dict) and cell.get("cell_type") == "code"
            )
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _is_executable_script(path: Path, text: str) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "__name__" in text and "__main__" in text
    if suffix == ".ipynb":
        return False
    return suffix in {".sh", ".ps1", ".bat", ".cmd", ".r", ".jl"}


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _training_inventory(root: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    executable_candidates: List[str] = []
    model_surfaces: List[str] = []
    training_logic_surfaces: List[str] = []
    source_count = 0
    for path in _iter_sources(root):
        source_count += 1
        rel = path.relative_to(root).as_posix()
        if rel in {
            "run_all_training.py", "tools/universal_training_controller.py",
            "tools/universal_training_controller_current.py",
            "tools/universal_training_controller_entry.py",
        }:
            continue
        text = _read_text(path)
        train_hit = _matches_any(text, TRAIN_PATTERNS)
        model_hit = _matches_any(text, MODEL_PATTERNS)
        executable = _is_executable_script(path, text)
        if train_hit:
            training_logic_surfaces.append(rel)
            if executable:
                executable_candidates.append(rel)
        if model_hit:
            model_surfaces.append(rel)
        if train_hit or model_hit:
            files.append({
                "path": rel,
                "training_logic": bool(train_hit),
                "model_surface": bool(model_hit),
                "executable": bool(executable),
                "checkpoint_write": bool(CHECKPOINT_WRITE.search(text)),
                "checkpoint_read": bool(CHECKPOINT_READ.search(text)),
                "resume_token": bool(RESUME_TOKEN.search(text)),
                "early_stopping": bool(EARLY_STOP_TOKEN.search(text)),
            })
    return {
        "source_files_scanned": source_count,
        "training_files": sorted(files, key=lambda x: x["path"]),
        "training_logic_surfaces": sorted(set(training_logic_surfaces)),
        "model_surfaces": sorted(set(model_surfaces)),
        "executable_training_candidates": sorted(set(executable_candidates)),
    }


def _covered_by_patterns(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        p = str(pattern).replace("\\", "/")
        if normalized == p or fnmatch.fnmatch(normalized, p):
            return True
    return False


def _module_names_for_path(rel: str) -> Set[str]:
    path = Path(rel)
    if path.suffix.lower() != ".py":
        return set()
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return set()
    names = {".".join(parts)}
    if parts[0] in {"src", "lib", "python"} and len(parts) > 1:
        names.add(".".join(parts[1:]))
    return {name for name in names if name}


def _python_module_index(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    collisions: Set[str] = set()
    for path in _iter_sources(root):
        if path.suffix.lower() != ".py":
            continue
        rel = path.relative_to(root).as_posix()
        for module in _module_names_for_path(rel):
            if module in index and index[module] != rel:
                collisions.add(module)
            else:
                index[module] = rel
    for module in collisions:
        index.pop(module, None)
    return index


def _resolve_module(index: Mapping[str, str], module: str) -> Set[str]:
    module = module.strip(".")
    if not module:
        return set()
    resolved: Set[str] = set()
    if module in index:
        resolved.add(index[module])
    parts = module.split(".")
    for n in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:n])
        if prefix in index:
            resolved.add(index[prefix])
    return resolved


def _relative_import_module(current_rel: str, module: str | None, level: int) -> str:
    current_path = Path(current_rel)
    package_parts = list(current_path.with_suffix("").parts[:-1])
    if current_path.name == "__init__.py":
        package_parts = list(current_path.parent.parts)
    if package_parts and package_parts[0] in {"src", "lib", "python"}:
        package_parts = package_parts[1:]
    if level > 0:
        trim = max(0, level - 1)
        if trim:
            package_parts = package_parts[:-trim] if trim <= len(package_parts) else []
    suffix = [p for p in (module or "").split(".") if p]
    return ".".join([*package_parts, *suffix])


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: List[str] = []
        cur: ast.AST | None = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _python_import_edges(root: Path, rel: str, index: Mapping[str, str]) -> Set[str]:
    text = _read_text(root / rel)
    if not text:
        return set()
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return set()
    edges: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.update(_resolve_module(index, alias.name))
        elif isinstance(node, ast.ImportFrom):
            base_module = (
                _relative_import_module(rel, node.module, int(node.level))
                if int(node.level or 0) > 0 else (node.module or "")
            )
            edges.update(_resolve_module(index, base_module))
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = ".".join(x for x in (base_module, alias.name) if x)
                edges.update(_resolve_module(index, child))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"importlib.import_module", "__import__"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    edges.update(_resolve_module(index, arg.value))
    edges.discard(rel)
    return edges


def _path_from_token(root: Path, owner_rel: str, token: str) -> str | None:
    token = token.strip().strip("'\"")
    candidates = [root / token, (root / owner_rel).parent / token]
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            if candidate.is_file():
                rel = candidate.relative_to(root.resolve()).as_posix()
                if Path(rel).suffix.lower() in EXEC_SCRIPT_SUFFIXES:
                    return rel
        except Exception:
            continue
    return None


def _literal_command_tokens(node: ast.AST) -> List[str]:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return []
    if isinstance(value, str):
        try:
            return shlex.split(value, posix=True)
        except Exception:
            return value.split()
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if isinstance(x, (str, int, float))]
    return []


def _python_execution_edges(root: Path, rel: str, index: Mapping[str, str]) -> Set[str]:
    text = _read_text(root / rel)
    if not text:
        return set()
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return set()
    edges: Set[str] = set()
    subprocess_names = {
        "subprocess.run", "subprocess.Popen", "subprocess.call",
        "subprocess.check_call", "subprocess.check_output", "os.system",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == "runpy.run_path" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                target = _path_from_token(root, rel, arg.value)
                if target:
                    edges.add(target)
        elif name in subprocess_names and node.args:
            tokens = _literal_command_tokens(node.args[0])
            joined = " ".join(tokens)
            for token in tokens:
                target = _path_from_token(root, rel, token)
                if target:
                    edges.add(target)
            for match in PYTHON_M_RE.finditer(joined):
                edges.update(_resolve_module(index, match.group(1)))
    return edges


def _text_execution_edges(root: Path, rel: str, index: Mapping[str, str]) -> Set[str]:
    path = root / rel
    if path.suffix.lower() == ".py":
        return _python_execution_edges(root, rel, index)
    text = _read_text(path)
    edges: Set[str] = set()
    for match in LOCAL_SCRIPT_TOKEN_RE.finditer(text):
        target = _path_from_token(root, rel, match.group("path"))
        if target and target != rel:
            edges.add(target)
    for match in PYTHON_M_RE.finditer(text):
        edges.update(_resolve_module(index, match.group(1)))
    return edges


def _command_repo_paths(root: Path, jobs: Sequence[Dict[str, Any]]) -> Set[str]:
    covered: Set[str] = set()
    index = _python_module_index(root)
    for job in jobs:
        command = [str(x) for x in job.get("command", [])]
        for part in command:
            try:
                candidate = Path(part)
                if candidate.is_absolute():
                    covered.add(candidate.resolve().relative_to(root).as_posix())
                elif (root / candidate).is_file():
                    covered.add((root / candidate).resolve().relative_to(root).as_posix())
            except Exception:
                continue
        for i, part in enumerate(command[:-1]):
            if part == "-m":
                covered.update(_resolve_module(index, command[i + 1]))
    return {
        rel for rel in covered
        if (root / rel).is_file() and Path(rel).suffix.lower() in SOURCE_SUFFIXES
    }


def _reachability(root: Path, jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    index = _python_module_index(root)
    roots = sorted(_command_repo_paths(root, jobs))
    import_graph: Dict[str, Set[str]] = {}
    execution_graph: Dict[str, Set[str]] = {}
    sources = list(_iter_sources(root))
    for path in sources:
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".py":
            import_graph[rel] = _python_import_edges(root, rel, index)
        if path.suffix.lower() in EXEC_SCRIPT_SUFFIXES:
            execution_graph[rel] = _text_execution_edges(root, rel, index)

    executed: Set[str] = set()
    queue = deque(roots)
    while queue:
        rel = queue.popleft()
        if rel in executed:
            continue
        executed.add(rel)
        for nxt in execution_graph.get(rel, set()):
            if nxt not in executed:
                queue.append(nxt)

    reachable: Set[str] = set(executed)
    queue = deque(executed)
    while queue:
        rel = queue.popleft()
        for nxt in import_graph.get(rel, set()) | execution_graph.get(rel, set()):
            if nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)

    per_job: Dict[str, Dict[str, List[str]]] = {}
    for job in jobs:
        job_id = str(job.get("id"))
        direct = sorted(_command_repo_paths(root, [job]))
        job_executed: Set[str] = set()
        q = deque(direct)
        while q:
            rel = q.popleft()
            if rel in job_executed:
                continue
            job_executed.add(rel)
            q.extend(execution_graph.get(rel, set()) - job_executed)
        job_reachable: Set[str] = set(job_executed)
        q = deque(job_executed)
        while q:
            rel = q.popleft()
            for nxt in import_graph.get(rel, set()) | execution_graph.get(rel, set()):
                if nxt not in job_reachable:
                    job_reachable.add(nxt)
                    q.append(nxt)
        per_job[job_id] = {
            "direct": direct,
            "executed": sorted(job_executed),
            "reachable": sorted(job_reachable),
        }
    return {
        "direct_job_sources": roots,
        "executed_sources": sorted(executed),
        "reachable_sources": sorted(reachable),
        "per_job": per_job,
        "python_import_edge_count": sum(len(v) for v in import_graph.values()),
        "local_execution_edge_count": sum(len(v) for v in execution_graph.values()),
    }


def _checkpoint_contract_for_paths(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    combined_parts: List[str] = []
    evidence: List[Dict[str, Any]] = []
    for rel in sorted(set(paths)):
        path = root / rel
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        text = _read_text(path)
        if not text:
            continue
        combined_parts.append(text)
        evidence.append({
            "path": rel,
            "checkpoint_write": bool(CHECKPOINT_WRITE.search(text)),
            "checkpoint_read": bool(CHECKPOINT_READ.search(text)),
            "resume_token": bool(RESUME_TOKEN.search(text)),
            "early_stopping": bool(EARLY_STOP_TOKEN.search(text)),
        })
    text = "\n".join(combined_parts)
    optimizer_used = bool(OPTIMIZER_USE.search(text))
    scheduler_used = bool(SCHEDULER_USE.search(text))
    scaler_used = bool(SCALER_USE.search(text))
    rng_used = bool(RNG_USE.search(text))
    model_save = bool(MODEL_STATE_SAVE.search(text))
    model_load = bool(MODEL_STATE_LOAD.search(text))
    fields = {
        "checkpoint_write": bool(CHECKPOINT_WRITE.search(text)),
        "checkpoint_read": bool(CHECKPOINT_READ.search(text)),
        "resume_token": bool(RESUME_TOKEN.search(text)),
        "model_state_save": model_save,
        "model_state_load": model_load,
        "optimizer_used": optimizer_used,
        "optimizer_state_save": bool(OPTIMIZER_STATE_SAVE.search(text)),
        "optimizer_state_load": bool(OPTIMIZER_STATE_LOAD.search(text)),
        "scheduler_used": scheduler_used,
        "scheduler_state_save": bool(SCHEDULER_STATE_SAVE.search(text)),
        "scheduler_state_load": bool(SCHEDULER_STATE_LOAD.search(text)),
        "scaler_used": scaler_used,
        "scaler_state_save": bool(SCALER_STATE_SAVE.search(text)),
        "scaler_state_load": bool(SCALER_STATE_LOAD.search(text)),
        "rng_used": rng_used,
        "rng_state_save": bool(RNG_SAVE.search(text)),
        "rng_state_load": bool(RNG_LOAD.search(text)),
        "progress_state_save": bool(PROGRESS_SAVE.search(text)),
        "progress_state_load": bool(PROGRESS_LOAD.search(text)),
        "early_stopping": bool(EARLY_STOP_TOKEN.search(text)),
    }
    native = fields["checkpoint_write"] and fields["checkpoint_read"] and fields["resume_token"]
    model_state_explicit = model_save or model_load
    exact_requirements = [
        fields["checkpoint_write"], fields["checkpoint_read"], fields["resume_token"],
        (not model_state_explicit or (model_save and model_load)),
        (not optimizer_used or (fields["optimizer_state_save"] and fields["optimizer_state_load"])),
        (not scheduler_used or (fields["scheduler_state_save"] and fields["scheduler_state_load"])),
        (not scaler_used or (fields["scaler_state_save"] and fields["scaler_state_load"])),
        (not rng_used or (fields["rng_state_save"] and fields["rng_state_load"])),
        fields["progress_state_save"] and fields["progress_state_load"],
    ]
    return {
        **fields,
        "native_resume_detected": bool(native),
        "exact_resume_detected": bool(native and all(exact_requirements)),
        "source_evidence": evidence,
    }


def _metadata_for_record(profile: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    job_id = str(record.get("id"))
    metadata: Dict[str, Any] = {}
    configured = profile.get("job_metadata", {})
    if isinstance(configured, dict) and isinstance(configured.get(job_id), dict):
        metadata.update(configured[job_id])
    for collection_name in ("jobs", "extra_jobs"):
        collection = profile.get(collection_name, []) or []
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("id") or item.get("name") or item.get("entrypoint") or "")
            if candidate_id == job_id:
                metadata.update({
                    k: v for k, v in item.items()
                    if k not in {"id", "name", "command", "entrypoint", "device_capable", "phase", "family", "repeat_index"}
                })
    return metadata


def _enhanced_job_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = _ORIGINAL_JOB_RECORDS(root, profile)
    for record in records:
        record.update(_metadata_for_record(profile, record))
    return records


def _job_resume_evidence(root: Path, job: Dict[str, Any], reachability: Mapping[str, Any]) -> Dict[str, Any]:
    job_id = str(job.get("id"))
    paths = reachability.get("per_job", {}).get(job_id, {}).get("reachable", [])
    contract = _checkpoint_contract_for_paths(root, paths)
    declared = str(job.get("resume_strategy") or "").strip().lower()
    declared_contract = job.get("checkpoint_contract")
    declared_exact = isinstance(declared_contract, dict) and declared_contract.get("exact_resume") is True
    if declared in {"exact_checkpoint", "framework_exact_checkpoint"}:
        declared_exact = True
    native = bool(contract["native_resume_detected"])
    if declared in {"native_checkpoint", "framework_checkpoint", "exact_checkpoint", "framework_exact_checkpoint"}:
        native = True
    exact = bool(contract["exact_resume_detected"] or declared_exact)
    early = bool(contract["early_stopping"] or job.get("early_stopping") is True)
    return {
        "job_id": job_id,
        "resume_strategy": declared or ("native_checkpoint_detected" if native else "unproven"),
        "native_resume_proven": native,
        "exact_resume_proven": exact,
        "early_stopping_present": early,
        "checkpoint_contract": contract,
        "declared_checkpoint_contract": declared_contract,
    }


def _job_dag(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [str(job.get("id")) for job in jobs]
    known = set(ids)
    dependencies: Dict[str, List[str]] = {}
    unknown: Dict[str, List[str]] = {}
    for job in jobs:
        job_id = str(job.get("id"))
        raw = job.get("depends_on", []) or []
        if isinstance(raw, str):
            raw = [raw]
        deps = [str(x) for x in raw]
        dependencies[job_id] = sorted(set(dep for dep in deps if dep in known and dep != job_id))
        bad = sorted(set(dep for dep in deps if dep not in known or dep == job_id))
        if bad:
            unknown[job_id] = bad
    indegree = {job_id: len(dependencies[job_id]) for job_id in ids}
    children: Dict[str, List[str]] = {job_id: [] for job_id in ids}
    for job_id, deps in dependencies.items():
        for dep in deps:
            children[dep].append(job_id)
    q = deque(sorted(job_id for job_id, degree in indegree.items() if degree == 0))
    order: List[str] = []
    while q:
        current = q.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                q.append(child)
    cyclic = sorted(job_id for job_id, degree in indegree.items() if degree > 0)
    edges = [{"from": dep, "to": job_id} for job_id in ids for dep in dependencies[job_id]]
    return {
        "nodes": ids, "edges": edges, "dependencies": dependencies,
        "unknown_dependencies": unknown, "cyclic_nodes": cyclic,
        "topological_order": order if not cyclic and not unknown else [],
        "valid": not cyclic and not unknown,
        "runtime_dependency_enforcement_required": bool(edges),
        "runtime_dependency_enforced": False,
    }


def _enhanced_coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    original = _ORIGINAL_COVERAGE_REPORT(root, profile, jobs)
    inventory = _training_inventory(root)
    reach = _reachability(root, jobs)
    directly_executed = set(reach["executed_sources"])
    reachable = set(reach["reachable_sources"])
    cover_patterns = list(profile.get("dynamic_registry_covers", []) or [])
    cover_patterns += list(profile.get("ignore_entrypoints", []) or [])
    setup_paths = [
        str(item).replace("\\", "/")
        for item in (profile.get("preferred_dataset_entrypoints", []) or [])
        if (root / str(item)).is_file()
    ]
    reachable.update(setup_paths)
    executable_uncovered = [
        path for path in inventory["executable_training_candidates"]
        if path not in directly_executed and not _covered_by_patterns(path, cover_patterns)
    ]
    model_unaccounted = [
        path for path in inventory["model_surfaces"]
        if path not in reachable and not _covered_by_patterns(path, cover_patterns)
    ]
    logic_unaccounted = [
        path for path in inventory["training_logic_surfaces"]
        if path not in reachable and not _covered_by_patterns(path, cover_patterns)
    ]
    resume = [_job_resume_evidence(root, job, reach) for job in jobs]
    unresolved_native = [r["job_id"] for r in resume if not r["native_resume_proven"]]
    unresolved_exact = [r["job_id"] for r in resume if not r["exact_resume_proven"]]
    missing_early = [r["job_id"] for r in resume if not r["early_stopping_present"]]
    dag = _job_dag(jobs)
    require_resume = bool(profile.get("require_native_resume", False)) or os.environ.get("TRAINING_CONTROL_REQUIRE_NATIVE_RESUME") == "1"
    require_exact = bool(profile.get("require_exact_resume", False)) or os.environ.get("TRAINING_CONTROL_REQUIRE_EXACT_RESUME") == "1"
    require_early = bool(profile.get("require_early_stopping", False)) or os.environ.get("TRAINING_CONTROL_REQUIRE_EARLY_STOPPING") == "1"
    require_model_accounting = bool(profile.get("require_model_surface_accounting", True))
    require_dag = bool(profile.get("require_dag_enforcement", False)) or os.environ.get("TRAINING_CONTROL_REQUIRE_DAG_ENFORCEMENT") == "1"
    strict_missing = list(executable_uncovered)
    if require_model_accounting:
        strict_missing.extend(model_unaccounted)
        strict_missing.extend(logic_unaccounted)
    dag_ok = bool(dag["valid"]) and (
        not require_dag or not dag["runtime_dependency_enforcement_required"] or dag["runtime_dependency_enforced"]
    )
    original.update({
        "schema": AUDIT_SCHEMA,
        "opf_reference_commit": OPF_REFERENCE_COMMIT,
        "opf_runtime_blobs": OPF_RUNTIME_BLOBS,
        "inventory": inventory,
        "reachability": reach,
        "uncovered_executable_training_candidates": sorted(set(executable_uncovered)),
        "unaccounted_model_surfaces": sorted(set(model_unaccounted)),
        "unaccounted_training_logic_surfaces": sorted(set(logic_unaccounted)),
        "resume_audit": resume,
        "unresolved_native_resume_jobs": unresolved_native,
        "unresolved_exact_resume_jobs": unresolved_exact,
        "jobs_without_early_stopping_evidence": missing_early,
        "job_dag": dag,
        "require_native_resume": require_resume,
        "require_exact_resume": require_exact,
        "require_early_stopping": require_early,
        "require_model_surface_accounting": require_model_accounting,
        "require_dag_enforcement": require_dag,
    })
    original["coverage_ok"] = (
        bool(original.get("coverage_ok", True)) and not strict_missing
        and (not require_resume or not unresolved_native)
        and (not require_exact or not unresolved_exact)
        and (not require_early or not missing_early) and dag_ok
    )
    return original


def _install_enhancements() -> None:
    _configure_reference()
    base._job_records = _enhanced_job_records
    base._coverage_report = _enhanced_coverage_report


_ORIGINAL_JOB_RECORDS = base._job_records
_ORIGINAL_COVERAGE_REPORT = base._coverage_report


def main(argv: Sequence[str] | None = None) -> int:
    _install_enhancements()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
