import json
import os
import subprocess
import sys

import pytest


def _base_environment(tmp_path) -> dict[str, str]:
    return {
        "API_KEY_OWNERS_JSON": json.dumps({"alice-key": "alice"}),
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "JOB_DB_PATH": str(tmp_path / "jobs.sqlite3"),
        "DOCUMENT_DB_PATH": str(tmp_path / "documents.sqlite3"),
        "CHROMA_PATH": str(tmp_path / "vectors"),
        "CLASSIC_STORAGE_DIR": str(tmp_path / "classic"),
        "ORPHAN_CLEANUP_ON_STARTUP": "false",
        "ALLOWED_MODELS": "test-model",
        "DEFAULT_MODEL": "test-model",
        "OPENAI_API_KEY": "",
        "OPENAI_BASE_URL": "",
    }


def _run_server_import(script: str, environment: dict[str, str]):
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def test_malformed_numeric_environment_imports_with_bounded_defaults(tmp_path):
    environment = _base_environment(tmp_path)
    environment.update({
        "MAX_UPLOAD_BYTES": "not-an-integer",
        "MAX_REMOTE_REDIRECTS": "nan",
        "JOB_TTL_SECONDS": "-999",
        "REQUESTS_PER_MINUTE": "infinity",
        "QUERY_WORKERS": "0",
        "QUERY_MAX_PENDING": "-1",
        "QUERY_TIMEOUT_SECONDS": "nan",
        "INGEST_WORKERS": "999999",
        "INGEST_MAX_ATTEMPTS": "bad",
        "INGEST_MAX_PENDING": "bad",
        "INGEST_ADMISSION_RETRY_SECONDS": "inf",
        "MAX_REQUEST_BODY_BYTES": "bad",
        "MAX_DOCX_COMPRESSION_RATIO": "nan",
        "MAX_CONCURRENT_TOOL_WORKERS": "128",
        "MAX_PENDING_TOOL_TASKS": "1",
        "PORT": "999999",
    })
    result = _run_server_import(
        """
import server as module
assert module.DEFAULT_MAX_UPLOAD_BYTES == 50_000_000
assert module.JOB_TTL_SECONDS == 60
assert module.REQUESTS_PER_MINUTE == 60
assert module.QUERY_WORKERS == 1
assert module.QUERY_MAX_PENDING >= module.QUERY_WORKERS
assert module.QUERY_TIMEOUT_SECONDS == 120.0
assert module.INGEST_WORKERS == 16
assert module.INGEST_MAX_ATTEMPTS == 3
assert module.INGEST_MAX_PENDING >= module.INGEST_WORKERS
assert module.INGEST_ADMISSION_RETRY_SECONDS == 1.0
assert module.MAX_REQUEST_BODY_BYTES >= module.DEFAULT_MAX_UPLOAD_BYTES
assert int(module.os.environ['MAX_PENDING_TOOL_TASKS']) >= 128
assert int(module.os.environ['MAX_RESPONSE_TOKENS']) <= 16_000
assert int(module.os.environ['PORT']) == 65_535
assert module.app.version == '4.4.0'
module._cancel_scheduled_ingestions()
module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
""",
        environment,
    )

    assert result.returncode == 0, result.stderr


def test_request_body_limit_is_never_below_upload_limit(tmp_path):
    environment = _base_environment(tmp_path)
    environment.update({
        "MAX_UPLOAD_BYTES": "123456",
        "MAX_REQUEST_BODY_BYTES": "1",
    })
    result = _run_server_import(
        """
import server as module
assert module.DEFAULT_MAX_UPLOAD_BYTES == 123_456
assert module.MAX_REQUEST_BODY_BYTES == 123_456
assert module.os.environ['MAX_REQUEST_BODY_BYTES'] == '123456'
module._cancel_scheduled_ingestions()
module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
""",
        environment,
    )

    assert result.returncode == 0, result.stderr


def _directory_symlink_or_skip(link, outside):
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable in this environment.")


@pytest.mark.parametrize(
    ("setting", "suffix"),
    [
        ("UPLOAD_DIR", "uploads"),
        ("JOB_DB_PATH", "state/jobs.sqlite3"),
        ("DOCUMENT_DB_PATH", "state/documents.sqlite3"),
        ("CHROMA_PATH", "vectors"),
        ("CLASSIC_STORAGE_DIR", "classic"),
    ],
)
def test_symlinked_service_path_component_is_rejected_before_server_app_import(
    tmp_path,
    setting,
    suffix,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-parent"
    _directory_symlink_or_skip(link, outside)
    environment = _base_environment(tmp_path)
    environment[setting] = str(link / suffix)

    result = _run_server_import("import server", environment)

    assert result.returncode != 0
    assert setting in result.stderr
    assert "symbolic-link components" in result.stderr
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "setting",
    [
        "UPLOAD_DIR",
        "JOB_DB_PATH",
        "DOCUMENT_DB_PATH",
        "CHROMA_PATH",
        "CLASSIC_STORAGE_DIR",
    ],
)
@pytest.mark.parametrize("control", ["\t", "\n", "\r", "\x7f"])
def test_control_bearing_service_paths_fail_before_server_app_import(
    tmp_path,
    setting,
    control,
):
    environment = _base_environment(tmp_path)
    environment[setting] = str(tmp_path / f"unsafe{control}path")

    result = _run_server_import("import server", environment)

    assert result.returncode != 0
    assert setting in result.stderr
    assert "invalid or too long" in result.stderr
    assert not any("unsafe" in path.name for path in tmp_path.iterdir())

def test_reparse_service_path_component_is_rejected_before_server_app_import(
    tmp_path,
):
    target = tmp_path / "reparse-root"
    target.mkdir()
    environment = _base_environment(tmp_path)
    environment["UPLOAD_DIR"] = str(target)
    environment["REPARSE_TARGET"] = str(target)

    result = _run_server_import(
        r"""
import os
import pathlib
import types

target = os.path.abspath(os.environ["REPARSE_TARGET"])
original_lstat = pathlib.Path.lstat
def reparse_lstat(self):
    metadata = original_lstat(self)
    if os.path.abspath(self) == target:
        return types.SimpleNamespace(
            st_mode=metadata.st_mode,
            st_file_attributes=0x400,
        )
    return metadata
pathlib.Path.lstat = reparse_lstat
import server
""",
        environment,
    )

    assert result.returncode != 0
    assert "UPLOAD_DIR" in result.stderr
    assert "reparse points" in result.stderr
