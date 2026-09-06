#!/usr/bin/env python3
"""Execute a repository's immutable prior launcher with new controller pins.

This is a bootstrap-preservation utility, not a scheduler.  It retrieves the
exact previous ``run_all_training.py`` by Git blob identity, executes it with
``__file__`` bound to the current repository root so all local imports/paths stay
unchanged, replaces only the universal-controller commit/blob/URL globals, and
calls its original ``main``.  Thus repository-specific catalogs, policies,
matrices and lifecycle metadata remain byte-identical to their last committed
launcher implementation.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

NEW_CONTROLLER_COMMIT = "471388dc1d9fa50eeae6c116a1508503b516c92d"
NEW_CONTROLLER_BLOB = "14d2fee4e0d505578e97a295dfb662ece82dbee7"
NEW_CONTROLLER_URL = (
    f"https://raw.githubusercontent.com/Anurag9000/RigorousRAG/{NEW_CONTROLLER_COMMIT}/"
    "tools/universal_training_controller_entry.py"
)


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _verified(data: bytes, expected: str, label: str) -> bytes:
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"{label} blob mismatch: {actual} != {expected}")
    return data


def _fetch_url(url: str, *, token: str = "") -> bytes:
    headers = {"User-Agent": "central-training-launcher-adapter/1"}
    if token:
        headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _load_previous(root: Path, repository: str, commit: str, expected: str) -> bytes:
    # Prefer the local clone's immutable Git object: no network/private-repo
    # assumptions and exact historical content.
    git = shutil.which("git")
    if git and (root / ".git").exists():
        try:
            data = subprocess.check_output(
                [git, "-C", str(root), "show", f"{commit}:run_all_training.py"],
                stderr=subprocess.DEVNULL,
            )
            return _verified(data, expected, "local previous launcher")
        except Exception:
            pass

    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        owner, name = repository.split("/", 1)
        api = (
            f"https://api.github.com/repos/{owner}/{name}/contents/"
            f"{urllib.parse.quote('run_all_training.py')}?ref={commit}"
        )
        try:
            return _verified(_fetch_url(api, token=token), expected, "API previous launcher")
        except Exception:
            pass

    gh = shutil.which("gh")
    if gh:
        try:
            data = subprocess.check_output(
                [
                    gh, "api", "-H", "Accept: application/vnd.github.raw+json",
                    f"repos/{repository}/contents/run_all_training.py?ref={commit}",
                ],
                stderr=subprocess.DEVNULL,
            )
            return _verified(data, expected, "gh previous launcher")
        except Exception:
            pass

    raw = f"https://raw.githubusercontent.com/{repository}/{commit}/run_all_training.py"
    return _verified(_fetch_url(raw), expected, "raw previous launcher")


def execute_previous_launcher(
    *,
    repository: str,
    base_commit: str,
    base_blob: str,
    root: Path | None = None,
) -> int:
    root = (root or Path.cwd()).resolve()
    data = _load_previous(root, repository, base_commit, base_blob)
    namespace: dict[str, Any] = {
        "__name__": "_immutable_previous_training_launcher",
        "__file__": str(root / "run_all_training.py"),
        "__package__": None,
    }
    code = compile(data.decode("utf-8"), str(root / "run_all_training.py"), "exec")
    exec(code, namespace, namespace)

    # Different historical launchers used different compact names.  Only these
    # controller locator values are changed; every repository profile/catalog
    # and every job definition remains exactly the previous implementation.
    for name in ("C", "COMMIT", "CONTROLLER_COMMIT"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_COMMIT
    for name in ("S", "SHA", "CONTROLLER_BLOB"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_BLOB
    for name in ("U", "URL", "SOURCE", "CONTROLLER_URL"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_URL

    target = namespace.get("main")
    if not callable(target):
        raise RuntimeError(f"Pinned prior launcher for {repository} has no callable main()")
    result = target()
    return int(result or 0)


if __name__ == "__main__":
    repository = os.environ["TRAINING_LAUNCHER_BASE_REPOSITORY"]
    base_commit = os.environ["TRAINING_LAUNCHER_BASE_COMMIT"]
    base_blob = os.environ["TRAINING_LAUNCHER_BASE_BLOB"]
    raise SystemExit(
        execute_previous_launcher(
            repository=repository,
            base_commit=base_commit,
            base_blob=base_blob,
            root=Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()),
        )
    )
