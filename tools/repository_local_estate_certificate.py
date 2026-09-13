#!/usr/bin/env python3
"""Repository-local certificate for the account-wide OPF training-control estate.

This verifier intentionally requires no cross-repository GitHub API access.  The
current repository's branch topology is read through the authenticated ``origin``
remote installed by ``actions/checkout``.  The public immutable RigorousRAG v37
bootstrap is verified by Git blob identity.  The private OPF_ADP live-head check is
left to the account-wide v38 certificate, whose credential is explicitly required
to see private siblings.

Certificate invariants:
* a first-class repository-owned scientific authority is declared at the root;
* opaque historical preservation-adapter roots are forbidden;
* every non-RigorousRAG target pins the exact canonical v37 bootstrap bytes;
* the repository remote HEAD/default branch is ``main`` and no other live branch
  refs remain;
* the canonical public v37 bootstrap still has its expected Git blob identity;
* retained Python source (excluding generated/vendor/data trees) compiles;
* no model/data execution or benchmark success is claimed by this certificate.

Deeper model/dataset/task/objective/registry/DAG/resume/early-stopping closure stays
with each repository's own central controller.  Live private OPF drift is checked
by the account-wide v38 certificate, not guessed from a repository-scoped token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import urllib.request
from typing import Any

CANONICAL_BOOTSTRAP_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CANONICAL_BOOTSTRAP_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CANONICAL_OPF_COMMIT = "1d1dfbbf7521ac40ee60c1f78f84956bf5f70598"
CANONICAL_HOST = "Anurag9000/RigorousRAG"
SCHEMA_VERSION = 4

REQUIRED_ROOT_MARKERS = (
    "scientific_authority",
    "require_literal_opf_mechanism_parity",
    "require_all_retained_trainable_source_reachability",
    "strict_coverage",
)
FORBIDDEN_ROOT_MARKERS = (
    "TRAINING_LAUNCHER_BASE_REPOSITORY",
    "TRAINING_LAUNCHER_BASE_COMMIT",
    "TRAINING_LAUNCHER_BASE_BLOB",
    "repo_training_launcher_adapter.py",
)
EXCLUDED_PATH_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "third_party",
    "build",
    "dist",
    "artifacts",
    "runs",
    "outputs",
    "data",
    "datasets",
    "__pycache__",
}


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _public_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "opf-repository-local-estate-certificate/4"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    return completed.stdout


def _remote_topology(root: Path) -> tuple[str, list[str]]:
    # actions/checkout persists a current-repository credential in git config, so
    # these commands work for private repositories without any sibling access.
    symref = _git_output(root, "ls-remote", "--symref", "origin", "HEAD")
    default_branch = ""
    for line in symref.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "ref:" and fields[2] == "HEAD":
            prefix = "refs/heads/"
            if fields[1].startswith(prefix):
                default_branch = fields[1][len(prefix):]
            break
    heads = _git_output(root, "ls-remote", "--heads", "origin")
    branches: list[str] = []
    prefix = "refs/heads/"
    for line in heads.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[1].startswith(prefix):
            continue
        branches.append(fields[1][len(prefix):])
    return default_branch, sorted(set(branches))


def _compile_retained_python(root: Path) -> tuple[int, list[str]]:
    count = 0
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if EXCLUDED_PATH_PARTS.intersection(relative.parts):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            failures.append(f"{relative.as_posix()}: {type(exc).__name__}: {exc}")
        else:
            count += 1
    return count, failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".training_control/local_estate_certificate.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repository = str(args.repository).strip()
    if not repository or "/" not in repository:
        raise SystemExit("--repository or GITHUB_REPOSITORY must be owner/name")
    root = args.root.resolve()
    errors: list[str] = []

    launcher = root / "run_all_training.py"
    launcher_text = ""
    if not launcher.is_file():
        errors.append("missing run_all_training.py")
    else:
        launcher_text = launcher.read_text(encoding="utf-8", errors="replace")
        for marker in REQUIRED_ROOT_MARKERS:
            if marker not in launcher_text:
                errors.append(f"root launcher missing first-class authority marker: {marker}")
        for marker in FORBIDDEN_ROOT_MARKERS:
            if marker in launcher_text:
                errors.append(f"opaque preservation adapter marker remains: {marker}")
        if repository != CANONICAL_HOST:
            if CANONICAL_BOOTSTRAP_COMMIT not in launcher_text:
                errors.append("root launcher does not pin canonical v37 commit")
            if CANONICAL_BOOTSTRAP_BLOB not in launcher_text:
                errors.append("root launcher does not pin canonical v37 blob")

    try:
        default_branch, branches = _remote_topology(root)
    except Exception as exc:
        default_branch, branches = "", []
        errors.append(f"cannot inspect authenticated origin topology: {type(exc).__name__}: {exc}")
    extra_branches = sorted(set(branches) - {"main"})
    if default_branch != "main":
        errors.append(f"default branch is {default_branch!r}, not 'main'")
    if "main" not in branches:
        errors.append("main branch is missing")
    if extra_branches:
        errors.append("extra branch refs remain: " + ", ".join(extra_branches))

    bootstrap = _public_bytes(
        f"https://raw.githubusercontent.com/{CANONICAL_HOST}/{CANONICAL_BOOTSTRAP_COMMIT}/"
        "tools/universal_training_controller_entry.py"
    )
    observed_bootstrap_blob = _git_blob_sha(bootstrap)
    if observed_bootstrap_blob != CANONICAL_BOOTSTRAP_BLOB:
        errors.append(
            "canonical v37 bootstrap blob drifted: "
            f"{observed_bootstrap_blob} != {CANONICAL_BOOTSTRAP_BLOB}"
        )

    compiled, compile_failures = _compile_retained_python(root)
    errors.extend("python compile failure: " + value for value in compile_failures)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "default_branch": default_branch,
        "branches": branches,
        "extra_branches": extra_branches,
        "first_class_scientific_authority": bool(launcher_text)
        and all(marker in launcher_text for marker in REQUIRED_ROOT_MARKERS),
        "opaque_preservation_adapter_root": bool(launcher_text)
        and any(marker in launcher_text for marker in FORBIDDEN_ROOT_MARKERS),
        "canonical_v37_commit": CANONICAL_BOOTSTRAP_COMMIT,
        "canonical_v37_blob": CANONICAL_BOOTSTRAP_BLOB,
        "observed_v37_blob": observed_bootstrap_blob,
        "canonical_opf_commit": CANONICAL_OPF_COMMIT,
        "live_private_opf_head_checked_here": False,
        "live_private_opf_head_check_authority": "account-wide-v38-with-cross-repository-credential",
        "compiled_retained_python_files": compiled,
        "compile_failures": compile_failures,
        "errors": sorted(set(errors)),
        "pass": not errors,
        "source_topology_certificate_only": True,
        "model_training_executed": False,
        "dataset_downloaded": False,
        "benchmark_claim_emitted": False,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
