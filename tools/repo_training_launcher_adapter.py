#!/usr/bin/env python3
"""Execute an immutable repository launcher with the current central controller.

This bootstrap-preservation utility retrieves the exact previous
``run_all_training.py`` by Git blob identity, executes it with ``__file__`` bound
to the current repository root, replaces only audited controller locator globals,
and calls its original ``main``. Repository-specific policies, matrices, lifecycle
metadata and scientific catalogs therefore remain the exact implementation chosen
by the pinned historical launcher.

v37 is the rate-limit-safe immutable bootstrap for the complete v20-v34/v36
controller stack. It materializes the scientific/controller bundle from one pinned
RigorousRAG archive, verifies every admitted file by Git blob identity, and keeps
resource scheduling exclusively in the literal current OPF_ADP runtime. No
resource scheduling behavior is implemented or changed here.

An active repository may additionally set ``TRAINING_LAUNCHER_FINAL_CATALOG`` and
``TRAINING_LAUNCHER_FINAL_CATALOG_BLOB`` to advance an immutable launcher's
``FINAL_CATALOG`` pointer without reconstructing the rest of that launcher.
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

NEW_CONTROLLER_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
NEW_CONTROLLER_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
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
    headers = {"User-Agent": "central-training-launcher-adapter/18"}
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


def _apply_catalog_override(namespace: dict[str, Any]) -> None:
    catalog = (os.environ.get("TRAINING_LAUNCHER_FINAL_CATALOG") or "").strip()
    blob = (os.environ.get("TRAINING_LAUNCHER_FINAL_CATALOG_BLOB") or "").strip()
    if bool(catalog) != bool(blob):
        raise RuntimeError(
            "TRAINING_LAUNCHER_FINAL_CATALOG and TRAINING_LAUNCHER_FINAL_CATALOG_BLOB "
            "must be supplied together"
        )
    if not catalog:
        return
    if "FINAL_CATALOG" not in namespace or "FINAL_CATALOG_BLOB" not in namespace:
        raise RuntimeError(
            "catalog override requested but pinned historical launcher does not expose "
            "FINAL_CATALOG/FINAL_CATALOG_BLOB"
        )
    namespace["FINAL_CATALOG"] = catalog
    namespace["FINAL_CATALOG_BLOB"] = blob


def execute_previous_launcher(
    *, repository: str, base_commit: str, base_blob: str, root: Path | None = None,
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

    for name in ("C", "COMMIT", "CONTROLLER_COMMIT"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_COMMIT
    for name in ("S", "SHA", "CONTROLLER_BLOB"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_BLOB
    for name in ("U", "URL", "SOURCE", "CONTROLLER_URL"):
        if name in namespace:
            namespace[name] = NEW_CONTROLLER_URL

    _apply_catalog_override(namespace)
    target = namespace.get("main")
    if not callable(target):
        raise RuntimeError(f"Pinned prior launcher for {repository} has no callable main()")
    return int(target() or 0)


if __name__ == "__main__":
    raise SystemExit(
        execute_previous_launcher(
            repository=os.environ["TRAINING_LAUNCHER_BASE_REPOSITORY"],
            base_commit=os.environ["TRAINING_LAUNCHER_BASE_COMMIT"],
            base_blob=os.environ["TRAINING_LAUNCHER_BASE_BLOB"],
            root=Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()),
        )
    )
