#!/usr/bin/env python3
"""Account-wide verification for the OPF-derived training-control rollout.

This is the top-level certification command for the Anurag9000 repository estate.
It verifies *remote wiring* for every live owner repository and can additionally
perform source-level audits in local checkouts.

Remote verification (no model/data execution):
  * enumerate the owner's live repositories from GitHub, not a hard-coded list;
  * exclude only the immutable OPF_ADP reference repository;
  * require a ``main`` ref and ``run_all_training.py`` on main;
  * require every launcher to pin the canonical v20 bootstrap blob; the immutable
    source commit may differ when it resolves to those exact byte-identical bytes;
  * flag non-main default branches and enumerate all extra branch refs;
  * emit an atomic machine-readable estate certificate.

Local source verification (``--clone-missing`` and/or ``--checkout-root``):
  * ensure every repository is checked out at its remote main HEAD;
  * run ``run_all_training.py --training-control-audit`` only (no datasets/models);
  * require literal OPF mechanism parity and a current training-surface snapshot;
  * require zero uncovered executable trainers, zero unaccounted model/training
    surfaces, a valid/enforced DAG, and all strict recovery/training contracts;
  * require zero unresolved exact-resume training jobs and zero training jobs
    lacking early-stopping evidence (unless the shared contract explicitly
    records a valid exemption);
  * collect every per-repository coverage report into one estate certificate.

A zero exit code means the exact source trees that were audited satisfy all
currently enabled controller contracts.  It is intentionally not a substitute
for the separate runtime/fault-injection test suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

OWNER = "Anurag9000"
REFERENCE_REPO = "OPF_ADP"
CANONICAL_HOST_REPO = "Anurag9000/RigorousRAG"
CANONICAL_BOOTSTRAP_COMMIT = "8080f8c8e55d802d4220bcc1c9b62a4f2e2ce052"
CANONICAL_BOOTSTRAP_BLOB = "a739ff9e31d9be7b5c9b0fe8d9bcfca6d75c846b"
SCHEMA = 1


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _headers(raw: bool = False) -> dict[str, str]:
    result = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "User-Agent": "opf-account-training-control-audit/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        result["Authorization"] = f"Bearer {token}"
    return result


def _request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers=_headers(False))
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_raw(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_headers(True))
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _gh_api_json(endpoint: str) -> Any:
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        return json.loads(subprocess.check_output([gh, "api", endpoint], text=True))
    except Exception:
        return None


def _json(endpoint: str) -> Any:
    via_gh = _gh_api_json(endpoint)
    if via_gh is not None:
        return via_gh
    return _request_json("https://api.github.com/" + endpoint.lstrip("/"))


def _raw(repo: str, path: str, ref: str) -> bytes:
    gh = shutil.which("gh")
    endpoint = f"repos/{repo}/contents/{urllib.parse.quote(path, safe='/')}?ref={urllib.parse.quote(ref, safe='')}"
    if gh:
        try:
            return subprocess.check_output(
                [gh, "api", "-H", "Accept: application/vnd.github.raw+json", endpoint],
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    return _request_raw("https://api.github.com/" + endpoint)


def _owner_repositories() -> list[dict[str, Any]]:
    repositories: list[dict[str, Any]] = []
    page = 1
    while True:
        rows = _json(f"user/repos?affiliation=owner&per_page=100&page={page}&sort=full_name")
        if not isinstance(rows, list):
            raise RuntimeError("GitHub repository enumeration did not return a list")
        repositories.extend(row for row in rows if isinstance(row, dict))
        if len(rows) < 100:
            break
        page += 1
    return sorted(repositories, key=lambda row: str(row.get("full_name") or ""))


def _branch_names(repo: str) -> list[str]:
    names: list[str] = []
    page = 1
    while True:
        rows = _json(f"repos/{repo}/branches?per_page=100&page={page}")
        if not isinstance(rows, list):
            break
        names.extend(str(row.get("name")) for row in rows if isinstance(row, dict) and row.get("name"))
        if len(rows) < 100:
            break
        page += 1
    return sorted(set(names))


def _main_sha(repo: str) -> str | None:
    try:
        ref = _json(f"repos/{repo}/git/ref/heads/main")
        return str(ref["object"]["sha"])
    except Exception:
        return None


def _launcher_remote_audit(repo_row: dict[str, Any]) -> dict[str, Any]:
    full_name = str(repo_row.get("full_name") or "")
    default_branch = str(repo_row.get("default_branch") or "")
    branches = _branch_names(full_name)
    main_sha = _main_sha(full_name)
    errors: list[str] = []
    launcher_blob = None
    launcher_text = ""
    if not main_sha:
        errors.append("missing main branch")
    else:
        try:
            launcher = _raw(full_name, "run_all_training.py", "main")
            launcher_blob = _git_blob_sha(launcher)
            launcher_text = launcher.decode("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"missing/unreadable run_all_training.py on main: {exc}")

    if launcher_text:
        if CANONICAL_BOOTSTRAP_BLOB not in launcher_text:
            errors.append("launcher does not pin canonical v20 bootstrap blob")
        if "universal_training_controller_entry.py" not in launcher_text:
            errors.append("launcher does not invoke canonical bootstrap entrypoint")
        if "require_literal_opf_mechanism_parity" not in launcher_text:
            errors.append("launcher does not explicitly require literal OPF mechanism parity")

    return {
        "repository": full_name,
        "default_branch": default_branch,
        "default_branch_is_main": default_branch == "main",
        "main_sha": main_sha,
        "branches": branches,
        "extra_branches": [name for name in branches if name != "main"],
        "launcher_blob": launcher_blob,
        "canonical_bootstrap_commit": CANONICAL_BOOTSTRAP_COMMIT,
        "canonical_bootstrap_blob": CANONICAL_BOOTSTRAP_BLOB,
        "errors": errors,
        "pass": not errors,
    }


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def _ensure_checkout(repo: str, main_sha: str, root: Path, clone_missing: bool) -> tuple[Path | None, list[str]]:
    name = repo.split("/", 1)[1]
    path = root / name
    errors: list[str] = []
    if not path.exists():
        if not clone_missing:
            return None, ["checkout missing"]
        gh = shutil.which("gh")
        if gh:
            completed = _run([gh, "repo", "clone", repo, str(path), "--", "--no-tags"], cwd=root)
        else:
            completed = _run(["git", "clone", "--no-tags", f"https://github.com/{repo}.git", str(path)], cwd=root)
        if completed.returncode != 0:
            return None, [f"clone failed: {completed.stderr.strip()}"]
    if not (path / ".git").exists():
        return None, ["checkout path is not a git repository"]
    fetched = _run(["git", "fetch", "origin", "main", "--no-tags"], cwd=path)
    if fetched.returncode != 0:
        errors.append(f"git fetch main failed: {fetched.stderr.strip()}")
        return None, errors
    head = _run(["git", "rev-parse", "origin/main"], cwd=path)
    observed = head.stdout.strip() if head.returncode == 0 else ""
    if observed != main_sha:
        errors.append(f"origin/main mismatch: {observed} != {main_sha}")
        return None, errors
    # The audit must inspect exactly remote main, not whatever local worktree was active.
    detached = _run(["git", "checkout", "--detach", main_sha], cwd=path)
    if detached.returncode != 0:
        errors.append(f"cannot detach at remote main: {detached.stderr.strip()}")
        return None, errors
    return path, errors


def _strict_local_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not bool(report.get("coverage_ok")):
        errors.append("coverage_ok=false")
    if not bool(report.get("strict_literal_opf_mechanism_parity_pass")):
        errors.append("literal OPF mechanism parity failed")
    certificate = report.get("opf_mechanism_certificate") or {}
    if not isinstance(certificate, dict) or not bool(certificate.get("pass")):
        errors.append("OPF mechanism certificate absent/failed")
    snapshot = report.get("training_surface_snapshot") or {}
    if not isinstance(snapshot, dict) or not snapshot.get("digest"):
        errors.append("training-surface snapshot absent")
    for key in (
        "uncovered_executable_training_candidates",
        "unaccounted_model_surfaces",
        "unaccounted_training_logic_surfaces",
        "unresolved_exact_resume_jobs",
        "training_jobs_without_exact_resume",
        "training_jobs_without_early_stopping",
        "malformed_training_early_stopping_exemptions",
    ):
        value = report.get(key, []) or []
        if value:
            errors.append(f"{key}={len(value) if isinstance(value, list) else value}")
    dag = report.get("job_dag") or {}
    if isinstance(dag, dict):
        if not bool(dag.get("valid", True)):
            errors.append("job DAG invalid")
        if dag.get("runtime_dependency_enforcement_required") and not dag.get("runtime_dependency_enforced"):
            errors.append("job DAG dependencies not runtime-enforced")
    return errors


def _local_audit(repo: str, main_sha: str, checkout_root: Path, clone_missing: bool, opf_root: Path | None) -> dict[str, Any]:
    checkout, errors = _ensure_checkout(repo, main_sha, checkout_root, clone_missing)
    result: dict[str, Any] = {"repository": repo, "checkout": None if checkout is None else str(checkout), "errors": list(errors)}
    if checkout is None:
        result["pass"] = False
        return result
    env = os.environ.copy()
    if opf_root is not None:
        env["OPF_REFERENCE_LOCAL_ROOT"] = str(opf_root.resolve())
    completed = _run([sys.executable, "run_all_training.py", "--training-control-audit"], cwd=checkout, env=env)
    report_path = checkout / ".training_control" / "coverage_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = {}
        errors.append(f"coverage report unavailable: {exc}")
    errors.extend(_strict_local_report(report) if isinstance(report, dict) else ["coverage report not an object"])
    if completed.returncode not in {0, 2}:
        errors.append(f"audit process returned unexpected {completed.returncode}")
    if completed.returncode == 2 and not errors:
        errors.append("audit reported strict failure without a parsed blocker")
    result.update(
        {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "coverage_report": report,
            "errors": errors,
            "pass": not errors and completed.returncode == 0,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--output", default=".training_control/account_wide_certificate.json")
    parser.add_argument("--checkout-root", type=Path)
    parser.add_argument("--clone-missing", action="store_true")
    parser.add_argument("--opf-root", type=Path)
    parser.add_argument("--remote-only", action="store_true")
    parser.add_argument("--require-main-default", action="store_true", help="also fail repositories whose GitHub default branch is not main")
    parser.add_argument("--require-main-only", action="store_true", help="also fail any repository that has any branch besides main")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.owner != OWNER:
        raise SystemExit(f"This estate verifier is intentionally scoped to {OWNER}")
    repositories = _owner_repositories()
    targets = [row for row in repositories if str(row.get("name")) != REFERENCE_REPO and not bool(row.get("archived"))]
    remote_rows = [_launcher_remote_audit(row) for row in targets]
    errors: list[str] = []
    for row in remote_rows:
        if not row["pass"]:
            errors.append(f"{row['repository']}: remote wiring failed")
        if args.require_main_default and not row["default_branch_is_main"]:
            errors.append(f"{row['repository']}: default branch is {row['default_branch']!r}, not main")
        if args.require_main_only and row["extra_branches"]:
            errors.append(f"{row['repository']}: extra branches {row['extra_branches']}")

    local_rows: list[dict[str, Any]] = []
    if not args.remote_only and args.checkout_root is not None:
        args.checkout_root.mkdir(parents=True, exist_ok=True)
        for row in remote_rows:
            if row.get("main_sha"):
                local_rows.append(
                    _local_audit(
                        str(row["repository"]),
                        str(row["main_sha"]),
                        args.checkout_root,
                        args.clone_missing,
                        args.opf_root,
                    )
                )
        errors.extend(f"{row['repository']}: local source audit failed" for row in local_rows if not row.get("pass"))

    certificate = {
        "schema": SCHEMA,
        "owner": OWNER,
        "reference_repository": f"{OWNER}/{REFERENCE_REPO}",
        "canonical_controller_host": CANONICAL_HOST_REPO,
        "canonical_bootstrap_commit": CANONICAL_BOOTSTRAP_COMMIT,
        "canonical_bootstrap_blob": CANONICAL_BOOTSTRAP_BLOB,
        "repository_count_including_reference": len(repositories),
        "target_repository_count": len(targets),
        "remote": remote_rows,
        "local": local_rows,
        "requirements": {
            "remote_wiring": True,
            "local_source_audit_requested": bool(not args.remote_only and args.checkout_root is not None),
            "require_main_default": bool(args.require_main_default),
            "require_main_only": bool(args.require_main_only),
        },
        "errors": errors,
        "pass": not errors,
    }
    _atomic_json(Path(args.output), certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())