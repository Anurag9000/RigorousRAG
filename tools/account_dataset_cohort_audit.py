#!/usr/bin/env python3
"""Fail-closed account-wide audit for the dataset-cohort rollout.

The OPF scheduler audit proves process/resource-control parity.  This companion audit
proves the orthogonal dataset-cohort migration state for every live owner repo:

* ``RigorousRAG`` owns the canonical runtime bytes;
* a trainable/migration-ready repository pins those exact bytes through a tiny
  repository-local loader;
* a repository with no local optimizer owns an explicit N/A certificate; or
* a genuinely empty repository is reported as empty and cannot silently acquire
  source without becoming uncovered on the next audit.

The immutable ``OPF_ADP`` reference repository is excluded.  No repository is
classified from its name.  The audit enumerates the owner's live GitHub repositories
and verifies actual files on ``main``.  A non-empty repository with neither a pinned
runtime nor an explicit N/A certificate is a hard failure.

This is source/configuration evidence only.  ``runtime_pinned`` means the shared
runtime dependency is available; it does not claim that a repository-specific
adapter has already been promoted as the active scientific authority.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

OWNER = "Anurag9000"
REFERENCE_REPO = "OPF_ADP"
CANONICAL_REPO = "RigorousRAG"
RUNTIME_COMMIT = "0c50adb23e5ba58b7c49b18401950f0bf7e5b736"
RUNTIME_BLOB = "1ffb1af0f70812d3418a6f0959ae88c7a145939e"
RUNTIME_PATH = "training/dataset_cohort_runtime.py"
LOADER_PATHS = (
    "training_control/dataset_cohort_runtime_entry.py",
    "railguard/training/dataset_cohort_runtime_entry.py",
)
NA_PATH = "training_control/dataset_cohort_not_applicable_v1.py"
NA_SCHEMA = "opf-dataset-cohort-not-applicable/v1"
SCHEMA = "opf-account-dataset-cohort-audit/v1"


def _headers(raw: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "User-Agent": "opf-account-dataset-cohort-audit/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _gh(endpoint: str, *, raw: bool = False) -> bytes:
    gh = shutil.which("gh")
    if gh:
        command = [gh, "api"]
        if raw:
            command += ["-H", "Accept: application/vnd.github.raw+json"]
        command.append(endpoint)
        try:
            return subprocess.check_output(command, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    request = urllib.request.Request(
        "https://api.github.com/" + endpoint.lstrip("/"),
        headers=_headers(raw),
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _json(endpoint: str) -> Any:
    return json.loads(_gh(endpoint, raw=False).decode("utf-8"))


def _raw(repo: str, path: str, ref: str = "main") -> bytes | None:
    endpoint = (
        f"repos/{repo}/contents/{urllib.parse.quote(path, safe='/')}"
        f"?ref={urllib.parse.quote(ref, safe='')}"
    )
    try:
        return _gh(endpoint, raw=True)
    except (urllib.error.HTTPError, subprocess.CalledProcessError) as exc:
        status = getattr(exc, "code", None)
        if status == 404 or isinstance(exc, subprocess.CalledProcessError):
            return None
        raise


def _repositories() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _json(f"user/repos?affiliation=owner&per_page=100&page={page}&sort=full_name")
        if not isinstance(payload, list):
            raise RuntimeError("GitHub repository enumeration did not return a list")
        rows.extend(value for value in payload if isinstance(value, dict))
        if len(payload) < 100:
            break
        page += 1
    return sorted(rows, key=lambda value: str(value.get("full_name") or ""))


def _loader_state(repo: str) -> tuple[str | None, list[str]]:
    errors: list[str] = []
    for path in LOADER_PATHS:
        payload = _raw(repo, path)
        if payload is None:
            continue
        text = payload.decode("utf-8", errors="strict")
        if RUNTIME_COMMIT not in text:
            errors.append(f"{path}: missing canonical runtime commit {RUNTIME_COMMIT}")
        if RUNTIME_BLOB not in text:
            errors.append(f"{path}: missing canonical runtime blob {RUNTIME_BLOB}")
        if RUNTIME_PATH not in text:
            errors.append(f"{path}: missing canonical runtime path {RUNTIME_PATH}")
        return path, errors
    return None, errors


def _na_state(repo: str) -> tuple[dict[str, Any] | None, list[str]]:
    payload = _raw(repo, NA_PATH)
    if payload is None:
        return None, []
    text = payload.decode("utf-8", errors="strict")
    errors: list[str] = []
    if NA_SCHEMA not in text:
        errors.append(f"{NA_PATH}: missing N/A schema {NA_SCHEMA}")
    if repo not in text:
        errors.append(f"{NA_PATH}: certificate does not bind repository {repo}")
    if "applicable" not in text or "False" not in text:
        errors.append(f"{NA_PATH}: certificate does not explicitly declare applicable=False")
    if "reason" not in text:
        errors.append(f"{NA_PATH}: certificate lacks a reason")
    return {"path": NA_PATH, "schema": NA_SCHEMA}, errors


def _canonical_state(repo: str) -> tuple[dict[str, Any] | None, list[str]]:
    if repo != f"{OWNER}/{CANONICAL_REPO}":
        return None, []
    payload = _raw(repo, RUNTIME_PATH)
    if payload is None:
        return None, [f"canonical runtime is missing: {RUNTIME_PATH}"]
    # Git blob SHA-1 is deliberately computed locally; this is the identity pinned
    # into all repository-local loaders.
    import hashlib

    actual = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
    errors = [] if actual == RUNTIME_BLOB else [
        f"canonical runtime blob drift: {actual} != {RUNTIME_BLOB}"
    ]
    return {
        "path": RUNTIME_PATH,
        "blob": actual,
        "commit_pin": RUNTIME_COMMIT,
    }, errors


def audit() -> dict[str, Any]:
    repositories: list[dict[str, Any]] = []
    global_errors: list[str] = []
    for row in _repositories():
        if bool(row.get("archived")):
            continue
        name = str(row.get("name") or "")
        full_name = str(row.get("full_name") or "")
        if not full_name or name == REFERENCE_REPO:
            continue
        size_kb = int(row.get("size") or 0)
        default_branch = str(row.get("default_branch") or "")
        errors: list[str] = []
        state = "uncovered"
        evidence: dict[str, Any] = {}

        canonical, canonical_errors = _canonical_state(full_name)
        errors.extend(canonical_errors)
        if canonical is not None:
            state = "canonical_runtime"
            evidence["canonical_runtime"] = canonical
        else:
            loader, loader_errors = _loader_state(full_name)
            errors.extend(loader_errors)
            na, na_errors = _na_state(full_name)
            errors.extend(na_errors)
            if loader and na:
                errors.append("repository declares both runtime-pinned and not-applicable states")
            elif loader:
                state = "runtime_pinned"
                evidence["loader"] = loader
            elif na:
                state = "not_applicable"
                evidence["certificate"] = na
            elif size_kb == 0:
                state = "empty"
            else:
                errors.append(
                    "non-empty repository has neither canonical runtime pin nor explicit dataset-cohort N/A certificate"
                )

        if default_branch != "main" and size_kb > 0:
            # The estate convention is main-only.  This audit does not mutate refs,
            # but records the branch drift alongside cohort coverage.
            errors.append(f"default branch is {default_branch!r}, expected 'main'")

        item = {
            "repository": full_name,
            "size_kb": size_kb,
            "default_branch": default_branch,
            "state": state,
            "evidence": evidence,
            "errors": errors,
            "pass": not errors,
        }
        repositories.append(item)
        global_errors.extend(f"{full_name}: {message}" for message in errors)

    counts: dict[str, int] = {}
    for row in repositories:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return {
        "schema": SCHEMA,
        "owner": OWNER,
        "reference_repository_excluded": f"{OWNER}/{REFERENCE_REPO}",
        "canonical_runtime": {
            "repository": f"{OWNER}/{CANONICAL_REPO}",
            "commit": RUNTIME_COMMIT,
            "path": RUNTIME_PATH,
            "blob": RUNTIME_BLOB,
        },
        "repository_count": len(repositories),
        "state_counts": dict(sorted(counts.items())),
        "repositories": repositories,
        "errors": global_errors,
        "pass": not global_errors,
        "source_configuration_only": True,
        "execution_claim_emitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_control/account_dataset_cohort_audit.json"),
    )
    args = parser.parse_args()
    payload = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"pass": payload["pass"], "states": payload["state_counts"], "errors": payload["errors"]}, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
