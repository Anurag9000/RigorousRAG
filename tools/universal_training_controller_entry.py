#!/usr/bin/env python3
"""Immutable flat bootstrap for exhaustive repository workload orchestration v36.

v36 removes the brittle historical bootstrap/cache ancestry while preserving the
*exact same* cumulative scientific/controller code and the literal current
OPF_ADP runtime. The complete v20-v34 controller stack is identified by Git blob
identity and materialized directly into one cache. No scheduler mechanism is
reimplemented here.

The resulting controller still installs every accumulated closure contract:
training-surface census, semantic multi-language learner discovery, DAG/catalog
closure, exact interruption resume, lifecycle and configuration matrices,
semantic early stopping, registry/subcommand/entrypoint discovery, model/dataset/
task/loss/optimizer/scheduler/sampler/preprocessing/augmentation/ensemble/pipeline
scientific closure, compatible-combination materialization, architecture and
method ontologies, Bayesian/federated/meta/sweep/data/decoding/robustness/modality
accounting, role/paradigm/protocol accounting, retained-trainable-source
reachability, and the literal OPF mechanism audit.

Resource admission, maximal admissible concurrency, CPU/GPU placement,
RAM/VRAM/swap pressure hysteresis, pause/resume, child adoption, persisted
scheduler state, retries, CUDA-OOM fallback, post-launch observation, telemetry,
logging and shutdown remain solely the byte-pinned OPF_ADP implementation.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HOST_REPO = "Anurag9000/RigorousRAG"
BUNDLE_VERSION = "v36"
CONTROLLER_FILES = {'tools/training_surface_census.py': '4db28f36151bdace4cdde9d3e429dac49c306129',
 'tools/training_surface_semantic_scan.py': 'd6185607102353f8d18b993427065832f3fb374b',
 'tools/universal_training_controller.py': '4353000e092ac286158c23500d91e898136fbab3',
 'tools/universal_training_controller_current.py': '09fe933dd520c7b97cdb0e86f5c5fbdc597336e4',
 'tools/universal_training_controller_dag.py': '3621f1fb0aeb843f1fb051cba074eedef67ac81e',
 'tools/universal_training_controller_exact_resume.py': '6f6311cbb7cdfa46d14ad5eb0adc8749c5080226',
 'tools/universal_training_controller_console.py': '89d4b8fde514ba426993d7068d3e4e6177600670',
 'tools/universal_training_controller_console_defaults.py': 'a4aae98c861764e0ece3bdcfb87f72eb531f6381',
 'tools/universal_training_controller_subcommands.py': 'a5c5d5ce5bfa719ec8942ca9eeae7d85143cf719',
 'tools/universal_training_controller_entrypoint_markers.py': 'd629e35d7bc1735bad841cc200d0ae532e16401a',
 'tools/universal_training_controller_inventory_scope.py': '2b2795fb53bb4e5fb8bb28c546229151d60b292b',
 'tools/universal_training_controller_audit_infrastructure.py': '40244bd2fd645bd201b90f107d95c7eefff64ffe',
 'tools/universal_training_controller_semantic_inventory.py': 'd4097817b26ad64baaf5d36a99455dac1b2adfa9',
 'tools/universal_training_controller_restart_exact.py': '436a34c7f87d87122b11d65d6c2cdefb2084fe26',
 'tools/universal_training_controller_opf_grace.py': '73db03de6eeb6cdcca685e1ffbfde60f08969f1e',
 'tools/universal_training_controller_registry_scheduling.py': 'b19334a09b52ad67b2e2c28ed36bcc10b6613175',
 'tools/universal_training_controller_training_contracts.py': 'dec455d8fa2e2cc88113f4d382f072908cc9b1ac',
 'tools/universal_training_controller_profile_file.py': '43f7ef739ce92f94ea7e3c444d6b0a56c34f61e3',
 'tools/universal_training_controller_job_catalog_v2.py': '9e0643ae5075e0901ffe28f26024db7b28d37a34',
 'tools/universal_training_controller_large_catalog.py': '805fbe26d0b6e0251b11a808e629f96e1d210b16',
 'tools/universal_training_controller_lifecycle.py': 'db8f0e93b1739ecf084a3ef8fa4dee7d33cc8452',
 'tools/universal_training_controller_metrics.py': 'c1c556ae30cde1fe4c4e913f3054c4d902544657',
 'tools/universal_training_controller_opf_mechanism_audit.py': '93ddb8583e642a2f5195d16f202d7a528ed2f0e2',
 'tools/universal_training_controller_deferred.py': '72e33311f00d0d2353c671a4c1663b1e9d0daf6a',
 'tools/universal_training_controller_deferred_v2.py': 'f0203b273ad58461178871a728c4ba18f73ab116',
 'tools/universal_training_controller_deferred_v3.py': '865378f887c269602676b1c7ca0859d25fd756b2',
 'tools/universal_training_controller_deferred_v4.py': '6dc85929f749cc1d5202d3481509e6db9b6aeb67',
 'tools/universal_training_controller_opf_reference_v2.py': 'ed59b42d50307fcb7a1c3ed9c8ae951b3ef3bd37',
 'tools/universal_training_controller_v20.py': 'b31aa9c11f3aaf19ef3078acfab198fd7df74f3b',
 'tools/universal_training_controller_lifecycle_exhaustive.py': '8e05a0ab48264cc3a3ffd8834bdf6ba40a9cf460',
 'tools/universal_training_controller_config_matrix.py': 'cfdc58f5dea0e5351871bb0adca7b1f39444d756',
 'tools/universal_training_controller_early_stopping_wiring.py': 'fdd00d7b406a77933c827237fb45db9d083001dc',
 'tools/universal_training_controller_lifecycle_affinity.py': '31e896f6ab10cc902d594e0c21c324d0f4092e8c',
 'tools/universal_training_controller_workload_closure.py': '622e0a2e3d45df8988613d9275c9478b600b89dc',
 'tools/universal_training_controller_dag_slicing.py': 'b5ecd8fbb152c1da40107cd023af19d1e86ab3e4',
 'tools/universal_training_controller_metrics_v2.py': '26bac973df28fa37f21c6464031fec2e3ece9908',
 'tools/universal_training_controller_registry_member_closure.py': '7bd369e1d27de859ed1278c34f5e5d01613ef9b1',
 'tools/universal_training_controller_v23.py': '5bddb1957ddf9cbe83e8e948c77d04cdc3edc05e',
 'tools/universal_training_controller_source_contracts_v24.py': '20e098c7df834ccd5d57eb7daf2056243c580f27',
 'tools/universal_training_controller_v24.py': '6f5e5a0a1478b9f15243b2ca550f96a645a45da0',
 'tools/universal_training_controller_scientific_surface_v25.py': '03ad055faa3d66acff8313083e7b6d1c6a37b977',
 'tools/universal_training_controller_v25.py': '05e860e652fb1c534b152322771cd90ed263d71f',
 'tools/universal_training_controller_selector_closure_v26.py': '1e0d3d15bb577e658b646fb5a689076f7cb0e058',
 'tools/universal_training_controller_v26.py': 'e545b9a72ddd360cde45e843198483203a6d4ed7',
 'tools/universal_training_controller_combination_closure_v27.py': 'b5d23bf38596a112482be5ecfa3835af92d552d0',
 'tools/universal_training_controller_v27.py': 'b4b026302f8bde1e408e3fe62632c8cc0a007685',
 'tools/universal_training_controller_declared_combination_materializer_v28.py': '249d49953b7ea42359b14eb2177500920e363afe',
 'tools/universal_training_controller_v28.py': '3ccf609957cc4bb9f6242518b768ccd39717ac14',
 'tools/universal_training_controller_scientific_ontology_v29.py': 'ff27586e040f98b150b50e6a7b8ee0e97423f4c3',
 'tools/universal_training_controller_v29.py': '78edf24d2c2c24b508bf7aa263332be7d76981fa',
 'tools/universal_training_controller_declaration_closure_v30.py': '02873be0cca8bf21f739fcb3a4576e5e810df256',
 'tools/universal_training_controller_v30.py': 'a55f4119eaa141a656aa431a7d3fa4767a3d124c',
 'tools/universal_training_controller_scientific_ontology_v31.py': '6ae33fbaf4a55e534232bdc9eca3a5727ae4c370',
 'tools/universal_training_controller_v31.py': '7c3e61a8924a7f88049b54be3654c3e086606bf1',
 'tools/universal_training_controller_scientific_ontology_v32.py': 'a0d4f27d0ba7f94f836ef60e10983257d3028236',
 'tools/universal_training_controller_v32.py': '99b0afbeb284700c0f7cbdb89dd09c6c60668c23',
 'tools/universal_training_controller_scientific_ontology_v33.py': 'a10ca7e562107af67d1ca30168cfc042a5f0bc3c',
 'tools/universal_training_controller_v33.py': '768ae59decf400171ebda9385b7318adb871c449',
 'tools/universal_training_controller_retained_training_closure_v34.py': 'afeb885dcbef6d20fe3294600e6ec9a7e191b5ed',
 'tools/universal_training_controller_v34.py': '90ce5e0425d23c86e8a2b13c21c45b75c0e9ba40'}

OPF_REPO = "Anurag9000/OPF_ADP"
OPF_COMMIT = "1d1dfbbf7521ac40ee60c1f78f84956bf5f70598"
OPF_FILES = {'utils/opf_massive_suite_runner.py': 'b2ae3d04f9398df5c18c7c13f4c939bce46b930d',
 'utils/runtime_tuning.py': 'f1cbfc44e009701a5540a046f2cd6b9f41f16b74',
 'utils/ml_backends.py': 'c4cd5eaf783cd7ffbb92ab01ec743ef7cbd13d84',
 'utils/logging_utils.py': '482ba94643aa921f49eebb835f29cf4930bb2498',
 'utils/opf_shared_defaults.py': 'bd76baa134b07567015d0151d5f14ba81dc667df',
 'DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py': 'ff79e8c51f1fb21a11e4687989198ef0abb07491',
 'tests/test_massive_scheduler_operational_contract.py': 'dec947ef375a346fb7abf06d77cbef1534852746'}
LEGACY_OPF_COMMIT = "a34c31259bd5d5f58081e3766918f9df63017455"
LEGACY_OPF_FILES = {'utils/opf_massive_suite_runner.py': 'b97d47499c83bc6ed3a5753f7f3009b624c94868',
 'utils/runtime_tuning.py': 'f1cbfc44e009701a5540a046f2cd6b9f41f16b74',
 'utils/ml_backends.py': '2fe2b24e530cab3d747c983c4457f4080703512f',
 'utils/logging_utils.py': '482ba94643aa921f49eebb835f29cf4930bb2498',
 'utils/opf_shared_defaults.py': '76ad434ecef1f708c835210d4bc86e0717999d99',
 'DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py': 'dacb9a2c44d611c045fbb7512ba5327343f79a85'}

INIT_FILES = (
    "utils/__init__.py",
    "DNN/__init__.py",
    "DNN/VANILLA/__init__.py",
    "DNN/VANILLA/Dyn_DNN4OPF/__init__.py",
    "DNN/VANILLA/Dyn_DNN4OPF/utils/__init__.py",
    "tests/__init__.py",
)
ARG_ALIASES = {
    "--training-control-audit": "--audit-training-coverage",
    "--training-control-list-jobs": "--list-training-jobs",
}
DIAGNOSTIC_FLAGS = frozenset({
    "--training-control-audit",
    "--audit-training-coverage",
    "--list-training-jobs",
    "--training-control-list-jobs",
    "--help",
    "-h",
    "--version",
})
SELF_TEST_FLAG = "--training-control-bootstrap-self-test"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "opf-exhaustive-training-controller/39", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _github_headers(*, raw: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if raw:
        headers["Accept"] = "application/vnd.github.raw+json"
    return headers


def fetch_git_blob(repository: str, expected: str) -> bytes:
    """Fetch an immutable Git blob by content identity, independent of tree drift."""
    api = f"https://api.github.com/repos/{repository}/git/blobs/{expected}"
    payload = json.loads(fetch(api, _github_headers()).decode("utf-8"))
    encoding = str(payload.get("encoding", "")).lower()
    encoded = payload.get("content")
    if encoding != "base64" or not isinstance(encoded, str):
        raise RuntimeError(f"Unexpected Git blob response for {expected}: encoding={encoding!r}")
    data = base64.b64decode(encoded.replace("\n", ""))
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Controller Git blob mismatch: {actual} != {expected}")
    return data


def _verified_local(root: Path, relative: str, expected: str) -> bytes | None:
    try:
        data = (root / relative).read_bytes()
    except Exception:
        return None
    return data if git_blob_sha(data) == expected else None


def prepare_controller_cache(root: Path) -> Path:
    digest = hashlib.sha256(
        json.dumps(CONTROLLER_FILES, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache = root / ".training_control" / "controller_bundle" / f"{BUNDLE_VERSION}-{digest}"
    marker = cache / "BUNDLE.json"
    expected_marker = {
        "schema": "rigorousrag.training_control.bundle.v36",
        "repository": HOST_REPO,
        "files": CONTROLLER_FILES,
    }
    try:
        valid = (
            json.loads(marker.read_text(encoding="utf-8")) == expected_marker
            and all(
                (cache / Path(relative).name).is_file()
                and git_blob_sha((cache / Path(relative).name).read_bytes()) == expected
                for relative, expected in CONTROLLER_FILES.items()
            )
        )
    except Exception:
        valid = False
    if valid:
        return cache

    for relative, expected in CONTROLLER_FILES.items():
        destination = cache / Path(relative).name
        if destination.is_file() and git_blob_sha(destination.read_bytes()) == expected:
            continue
        data = _verified_local(root, relative, expected)
        if data is None:
            data = fetch_git_blob(HOST_REPO, expected)
        atomic_write(destination, data)

    atomic_write(marker, (json.dumps(expected_marker, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return cache


def _verified_opf_local(root: Path, relative: str, expected: str) -> bytes | None:
    roots: list[Path] = []
    explicit = os.environ.get("OPF_REFERENCE_LOCAL_ROOT", "").strip()
    if explicit:
        roots.append(Path(explicit).expanduser())
    roots.extend((
        root.parent / "OPF_ADP",
        Path.home() / "OPF_ADP",
        Path.home() / "projects" / "OPF_ADP",
        Path.home() / "Projects" / "OPF_ADP",
    ))
    for base in roots:
        try:
            data = (base / relative).read_bytes()
        except Exception:
            continue
        if git_blob_sha(data) == expected:
            return data
    return None


def fetch_opf(root: Path, relative: str, expected: str, commit: str) -> bytes:
    data = _verified_opf_local(root, relative, expected)
    if data is not None:
        return data

    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        api = (
            f"https://api.github.com/repos/{OPF_REPO}/contents/"
            f"{urllib.parse.quote(relative, safe='/')}?ref={commit}"
        )
        try:
            data = fetch(api, {
                "Accept": "application/vnd.github.raw+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            })
            if git_blob_sha(data) == expected:
                return data
        except Exception:
            pass

    gh = shutil.which("gh")
    if gh:
        try:
            data = subprocess.check_output(
                [
                    gh,
                    "api",
                    "-H",
                    "Accept: application/vnd.github.raw+json",
                    f"repos/{OPF_REPO}/contents/{relative}?ref={commit}",
                ],
                stderr=subprocess.DEVNULL,
            )
            if git_blob_sha(data) == expected:
                return data
        except Exception:
            pass

    try:
        raw = f"https://raw.githubusercontent.com/{OPF_REPO}/{commit}/{relative}"
        data = fetch(raw)
        if git_blob_sha(data) == expected:
            return data
    except Exception:
        pass
    raise RuntimeError(
        f"Cannot obtain verified private OPF reference {relative}@{commit}. "
        "Keep a sibling OPF_ADP checkout, set OPF_REFERENCE_LOCAL_ROOT, export a "
        "cross-repository GH_TOKEN/GITHUB_TOKEN, or authenticate the gh CLI."
    )


def prepare_reference_cache(root: Path, commit: str, files: dict[str, str]) -> Path:
    cache = root / ".training_control" / "opf_reference" / commit
    marker = cache / "REFERENCE.json"
    expected_marker = {"repository": OPF_REPO, "commit": commit, "files": files}
    try:
        valid = (
            json.loads(marker.read_text(encoding="utf-8")) == expected_marker
            and all(
                (cache / relative).is_file()
                and git_blob_sha((cache / relative).read_bytes()) == expected
                for relative, expected in files.items()
            )
        )
    except Exception:
        valid = False
    if valid:
        return cache

    for relative, expected in files.items():
        data = fetch_opf(root, relative, expected, commit)
        if git_blob_sha(data) != expected:
            raise RuntimeError(f"Pinned OPF blob mismatch for {relative}@{commit}")
        atomic_write(cache / relative, data)
    for relative in INIT_FILES:
        target = cache / relative
        if not target.exists():
            atomic_write(target, b"")
    atomic_write(marker, (json.dumps(expected_marker, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return cache


def diagnostic_only(argv: list[str]) -> bool:
    return bool(argv) and any(argument in DIAGNOSTIC_FLAGS for argument in argv)


def canonical_argv(argv: list[str]) -> list[str]:
    return [ARG_ALIASES.get(argument, argument) for argument in argv]


def _self_test(root: Path) -> int:
    cache = prepare_controller_cache(root)
    for relative, expected in CONTROLLER_FILES.items():
        path = cache / Path(relative).name
        data = path.read_bytes()
        if git_blob_sha(data) != expected:
            raise RuntimeError(f"bundle self-test mismatch for {relative}")
        compile(data, str(path), "exec")
    if os.environ.get("TRAINING_CONTROL_SELF_TEST_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
        prepare_reference_cache(root, OPF_COMMIT, OPF_FILES)
    print(json.dumps({
        "status": "ok",
        "bundle": BUNDLE_VERSION,
        "controller_files": len(CONTROLLER_FILES),
        "controller_manifest_sha256": hashlib.sha256(
            json.dumps(CONTROLLER_FILES, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "opf_commit": OPF_COMMIT,
        "opf_files": len(OPF_FILES),
    }, sort_keys=True))
    return 0


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    if SELF_TEST_FLAG in argv:
        return _self_test(root)

    cache = prepare_controller_cache(root)
    if not diagnostic_only(argv):
        prepare_reference_cache(root, OPF_COMMIT, OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            prepare_reference_cache(root, LEGACY_OPF_COMMIT, LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v34.py"), *canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
