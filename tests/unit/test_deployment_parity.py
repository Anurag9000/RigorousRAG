from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def test_ci_fetches_history_before_merge_base_whitespace_check():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "fetch-depth: 0" in workflow
    assert "git merge-base HEAD" in workflow
    assert "git diff --check" in workflow
    assert "git diff --check HEAD^" not in workflow
