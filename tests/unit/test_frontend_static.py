from pathlib import Path


def test_frontend_has_no_unsafe_html_assignment_or_persistent_local_storage():
    script = Path("frontend/app.js").read_text(encoding="utf-8")
    assert ".innerHTML" not in script
    assert "localStorage" not in script
    assert "sessionStorage" in script
    assert '"X-API-Key"' in script
    assert "X-Owner-ID" not in script


def test_frontend_is_self_contained_and_tools_remain_mobile_accessible():
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "cdn.jsdelivr" not in html
    assert "fonts.googleapis" not in html
    assert 'id="right-panel"' in html
    assert ".right-panel.open" in html
    assert 'id="open-tools"' in html


def test_frontend_uses_server_model_configuration():
    script = Path("frontend/app.js").read_text(encoding="utf-8")
    assert 'fetchApi("/config")' in script
    assert "allowed_models" in script
    assert "default_model" in script
