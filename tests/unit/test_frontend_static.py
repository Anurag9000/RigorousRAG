from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


def _all_frontend_scripts() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(FRONTEND.glob("*.js"))
    )


def test_frontend_has_no_unsafe_html_assignment_or_persistent_local_storage():
    scripts = _all_frontend_scripts()
    assert ".innerHTML" not in scripts
    assert "localStorage" not in scripts
    assert "sessionStorage" in scripts
    assert '"X-API-Key"' in scripts
    assert "X-Owner-ID" not in scripts


def test_frontend_is_self_contained_and_tools_remain_mobile_accessible():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "cdn.jsdelivr" not in html
    assert "fonts.googleapis" not in html
    assert 'id="right-panel"' in html
    assert ".right-panel.open" in html
    assert 'id="open-tools"' in html
    preload = html.index('<script src="preload.js" defer></script>')
    app = html.index('<script src="app.js" defer></script>')
    lifecycle = html.index('<script src="lifecycle.js" defer></script>')
    assert preload < app < lifecycle


def test_frontend_uses_server_model_configuration():
    script = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert 'fetchApi("/config")' in script
    assert "allowed_models" in script
    assert "default_model" in script


def test_frontend_understands_durable_jobs_and_visual_source_capability():
    script = (FRONTEND / "lifecycle.js").read_text(encoding="utf-8")
    for state in ("queued", "processing", "finalizing", "success", "failed"):
        assert f"{state}:" in script
    assert "source_retained" in script
    assert "visual_source_available" in script
    assert "Text evidence only" in script
    assert "Visual PDF eligible; identity and limits verified on use" in script
    assert "Figure tool" in script
    assert "figureButton.disabled = !visualEligible" in script


def test_frontend_applies_request_deadlines_and_bounded_file_enumeration():
    lifecycle = (FRONTEND / "lifecycle.js").read_text(encoding="utf-8")
    preload = (FRONTEND / "preload.js").read_text(encoding="utf-8")

    assert "AbortController" in lifecycle
    assert "AbortController" in preload
    assert "DEFAULT_CLIENT_REQUEST_TIMEOUT_MS" in lifecycle
    assert "DEFAULT_PRELOAD_TIMEOUT_MS" in preload
    assert "MAX_CLIENT_REQUEST_TIMEOUT_MS" in lifecycle
    assert "MAX_CLIENT_UPLOAD_FILES = 100" in lifecycle
    assert "boundedUploadFiles" in lifecycle
    assert "for (const file of files)" in lifecycle
    assert "Array.from(files" not in lifecycle
    assert "Request timed out before the server responded." in lifecycle
    assert "documents.slice(0, 5000)" in lifecycle


def test_frontend_fields_match_server_character_ceilings():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    required = {
        'id="query-input" maxlength="20000"',
        'id="api-key-input" maxlength="4096"',
        'id="ve-claim" maxlength="10000"',
        'id="ve-figure" maxlength="200"',
        'id="ve-docid" maxlength="200"',
        'id="proto-text" maxlength="30000"',
        'id="proto-docid" maxlength="200"',
        'id="debate-claim" maxlength="10000"',
        'id="debate-evidence" maxlength="30000"',
        'id="bib-title" maxlength="1000"',
        'id="bib-authors" maxlength="3000"',
        'id="bib-journal" maxlength="1000"',
        'id="bib-doi" maxlength="500"',
    }
    for marker in required:
        assert marker in html
