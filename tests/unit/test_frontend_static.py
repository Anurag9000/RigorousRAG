from pathlib import Path


def _all_frontend_scripts() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("frontend").glob("*.js"))
    )


def test_frontend_has_no_unsafe_html_assignment_or_persistent_local_storage():
    scripts = _all_frontend_scripts()
    assert ".innerHTML" not in scripts
    assert "localStorage" not in scripts
    assert "sessionStorage" in scripts
    assert '"X-API-Key"' in scripts
    assert "X-Owner-ID" not in scripts


def test_frontend_is_self_contained_and_tools_remain_mobile_accessible():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "cdn.jsdelivr" not in html
    assert "fonts.googleapis" not in html
    assert 'id="right-panel"' in html
    assert ".right-panel.open" in html
    assert 'id="open-tools"' in html
    assert '<script src="app.js" defer></script>' in html
    assert '<script src="lifecycle.js" defer></script>' in html


def test_frontend_uses_server_model_configuration():
    script = Path("frontend/app.js").read_text(encoding="utf-8")
    assert 'fetchApi("/config")' in script
    assert "allowed_models" in script
    assert "default_model" in script


def test_frontend_understands_durable_jobs_and_visual_source_capability():
    script = Path("frontend/lifecycle.js").read_text(encoding="utf-8")
    for state in ("queued", "processing", "finalizing", "success", "failed"):
        assert f"{state}:" in script
    assert "source_retained" in script
    assert "visual_source_available" in script
    assert "Text evidence only" in script
    assert "Visual PDF eligible; identity and limits verified on use" in script
    assert "Figure tool" in script
    assert "figureButton.disabled = !visualEligible" in script
