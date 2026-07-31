from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXACT_HEAD_WORKFLOW = ROOT / ".github" / "workflows" / "release-locks.yml"
_CHECKOUT_PIN = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
_SETUP_PYTHON_PIN = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
_UPLOAD_ARTIFACT_PIN = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def test_active_runtime_settings_are_exposed_in_env_and_compose():
    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    required = {
        "SUMMARY_MODEL",
        "EMBEDDING_MODEL",
        "MAX_RESPONSE_TOKENS",
        "MAX_TOOL_ARGUMENT_CHARS",
        "MAX_TOOL_RESULT_CHARS",
        "MAX_EVIDENCE_SOURCES",
        "MAX_CONCURRENT_TOOL_WORKERS",
        "MAX_PENDING_TOOL_TASKS",
        "QUERY_WORKERS",
        "QUERY_MAX_PENDING",
        "QUERY_TIMEOUT_SECONDS",
        "INGEST_WORKERS",
        "INGEST_MAX_PENDING",
        "INGEST_MAX_ATTEMPTS",
        "MAX_CHUNKS_PER_DOCUMENT",
        "DOCUMENT_LIST_SCAN_BATCH",
        "MAX_DOCUMENT_LIST_SCAN_CHUNKS",
        "MAX_VECTOR_METADATA_ITEMS",
        "MAX_SECTIONS_PER_DOCUMENT",
        "MAX_RAG_QUERY_CHARS",
        "MAX_REMOTE_DOWNLOAD_BYTES",
        "MAX_REMOTE_REQUEST_BODY_BYTES",
        "REMOTE_REQUEST_TIMEOUT_SECONDS",
        "MAX_REMOTE_REDIRECTS",
        "MAX_UPLOAD_BYTES",
        "MAX_REQUEST_BODY_BYTES",
    }

    for name in sorted(required):
        assert f"{name}=" in env_text, f"{name} missing from .env.example"
        assert f"{name}: ${{{name}:-" in compose_text, (
            f"{name} missing from docker-compose.yml"
        )


def test_exact_head_workflow_fetches_history_before_whitespace_check():
    workflow = EXACT_HEAD_WORKFLOW.read_text(encoding="utf-8")

    assert "fetch-depth: 0" in workflow
    assert "git merge-base HEAD" in workflow
    assert "git diff --check" in workflow
    assert "git diff --check HEAD^" not in workflow


def test_exact_head_workflow_is_unconditional_and_complete():
    workflow = EXACT_HEAD_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Exact-head verification and release locks" in workflow
    assert "  pull_request:\n" in workflow
    assert "  merge_group:\n" in workflow
    assert "paths:" not in workflow.split("permissions:", 1)[0]
    assert "python-version: [\"3.10\", \"3.11\", \"3.12\"]" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "docker compose config --quiet" in workflow
    assert "docker build --tag rigorousrag:ci ." in workflow
    assert "os: [ubuntu-latest, windows-latest, macos-latest]" in workflow
    assert "python scripts/verify_release_lock.py" in workflow
    assert "--require-hashes" in workflow


def test_exact_head_workflow_pins_third_party_actions_by_commit():
    workflow = EXACT_HEAD_WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count(_CHECKOUT_PIN) == 5
    assert workflow.count(_SETUP_PYTHON_PIN) == 3
    assert workflow.count(_UPLOAD_ARTIFACT_PIN) == 1
    assert "actions/checkout@v" not in workflow
    assert "actions/setup-python@v" not in workflow
    assert "actions/upload-artifact@v" not in workflow
    assert workflow.count("persist-credentials: false") == 5


def test_obsolete_duplicate_workflows_are_absent():
    workflow_dir = ROOT / ".github" / "workflows"

    assert not (workflow_dir / "ci.yml").exists()
    assert not (workflow_dir / "exact-head-verification.yml").exists()
