import itertools
import shutil
from pathlib import Path

import pytest

import setup_frontend


ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_frontend_verifies_without_mutation(tmp_path):
    repository = tmp_path / "repo"
    frontend = repository / "frontend"
    frontend.mkdir(parents=True)
    source = ROOT / "frontend"
    for name in setup_frontend._REQUIRED_ASSETS:
        shutil.copy2(source / name, frontend / name)
    before = {
        name: (frontend / name).read_bytes()
        for name in setup_frontend._REQUIRED_ASSETS
    }

    verified = setup_frontend.verify_frontend(repository)

    assert verified == list(setup_frontend._REQUIRED_ASSETS)
    after = {
        name: (frontend / name).read_bytes()
        for name in setup_frontend._REQUIRED_ASSETS
    }
    assert after == before


def test_verifier_rejects_missing_stale_and_symlinked_assets(tmp_path):
    repository = tmp_path / "repo"
    frontend = repository / "frontend"
    frontend.mkdir(parents=True)

    with pytest.raises(ValueError, match="index.html"):
        setup_frontend.verify_frontend(repository)

    source = ROOT / "frontend"
    for name in setup_frontend._REQUIRED_ASSETS:
        shutil.copy2(source / name, frontend / name)
    (frontend / "app.js").write_text(
        'const output = element.innerHTML; localStorage.setItem("key", "secret");',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required marker|forbidden token"):
        setup_frontend.verify_frontend(repository)

    (frontend / "app.js").unlink()
    try:
        (frontend / "app.js").symlink_to(source / "app.js")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    with pytest.raises(ValueError, match="symbolic link"):
        setup_frontend.verify_frontend(repository)


def test_verifier_rejects_control_paths_and_infinite_argument_streams(tmp_path):
    with pytest.raises(ValueError, match="invalid or too long"):
        setup_frontend.verify_frontend(tmp_path / "bad\nroot")

    assert setup_frontend.main("not-an-argv-list") == 2
    assert setup_frontend.main((str(index) for index in itertools.count())) == 2


def test_main_reports_verification_without_generation(capsys):
    assert setup_frontend.main([str(ROOT)]) == 0

    output = capsys.readouterr().out
    assert "Verified checked-in frontend assets" in output
    assert "No files were generated or modified" in output
