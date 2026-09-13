#!/usr/bin/env python3
"""Repository-local certificate for the account-wide OPF training-control estate.

This verifier is intentionally safe to run with a repository-scoped GitHub Actions
token.  It does not enumerate private sibling repositories.  Instead, each target
repository certifies its own source tree and GitHub topology against the immutable
shared controller/OPF contract.

Certificate invariants:
* a first-class repository-owned scientific authority is declared at the root;
* opaque historical preservation-adapter roots are forbidden;
* every non-RigorousRAG target pins the exact canonical v37 bootstrap bytes;
* the repository default branch is ``main`` and no other live branch refs remain;
* the canonical v37 bootstrap still has its expected Git blob identity;
* OPF_ADP/main is still the exact scheduler commit certified by v37;
* retained Python source (excluding generated/vendor/data trees) compiles;
* the emitted certificate never claims model/data execution or benchmark success.

The repository's own central controller remains responsible for deeper model,
dataset, task, objective, registry, DAG, resume and early-stopping closure.  This
program certifies the estate wiring/topology layer and makes those invariants
independently observable from every private repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import urllib.parse
import urllib.request
from typing import Any

CANONICAL_BOOTSTRAP_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CANONICAL_BOOTSTRAP_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CANONICAL_OPF_COMMIT = "1d1dfbbf7521ac40ee60c1f78f84956bf5f70598"
CANONICAL_HOST = "Anurag9000/RigorousRAG"
REFERENCE_REPO = "Anurag9000/OPF_ADP"
SCHEMA_VERSION = 2

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


def _headers(token: str, *, raw: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "User-Agent": "opf-repository-local-estate-certificate/2",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json(url: str, token: str) -> Any:
    request = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _bytes(url: str, token: str = "") -> bytes:
    request = urllib.request.Request(url, headers=_headers(token, raw=True))
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _branches(repository: str, token: str) -> list[str]:
    names: list[str] = []
    page = 1
    while True:
        rows = _json(
            f"https://api.github.com/repos/{repository}/branches?per_page=100&page={page}",
            token,
        )
        if not isinstance(rows, list):
            raise RuntimeError("GitHub branch enumeration did not return a list")
        names.extend(str(row.get("name")) for row in rows if isinstance(row, dict) and row.get("name"))
        if len(rows) < 100:
            break
        page += 1
    return sorted(set(names))


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
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

    metadata = _json(f"https://api.github.com/repos/{repository}", token)
    default_branch = str(metadata.get("default_branch") or "") if isinstance(metadata, dict) else ""
    branches = _branches(repository, token)
    extra_branches = sorted(set(branches) - {"main"})
    if default_branch != "main":
        errors.append(f"default branch is {default_branch!r}, not 'main'")
    if "main" not in branches:
        errors.append("main branch is missing")
    if extra_branches:
        errors.append("extra branch refs remain: " + ", ".join(extra_branches))

    opf_row = _json(f"https://api.github.com/repos/{REFERENCE_REPO}/commits/main", token)
    live_opf = str(opf_row.get("sha") or "") if isinstance(opf_row, dict) else ""
    if live_opf != CANONICAL_OPF_COMMIT:
        errors.append(f"OPF_ADP/main drifted: {live_opf!r} != {CANONICAL_OPF_COMMIT}")

    bootstrap = _bytes(
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
        "observed_opf_main": live_opf,
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
