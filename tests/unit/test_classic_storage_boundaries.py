import os
import subprocess
import sys

import pytest

from storage import CrawlState, StorageManager


def test_standalone_storage_import_normalizes_malformed_byte_limit():
    environment = os.environ.copy()
    environment["CLASSIC_MAX_SNAPSHOT_FILE_BYTES"] = "nan"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, storage; "
                "s=storage.StorageManager(); "
                "assert s.max_snapshot_file_bytes == 250_000_000; "
                "assert os.environ['CLASSIC_MAX_SNAPSHOT_FILE_BYTES']=='250000000'"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_symlinked_legacy_member_is_not_followed(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text('{"schema_version":3,"pages":{}}', encoding="utf-8")
    manager = StorageManager(tmp_path / "data")
    try:
        manager.crawl_path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    assert manager.load_crawl_state() == CrawlState.empty()
    assert target.read_text(encoding="utf-8").startswith("{")
    assert manager.crawl_path.is_symlink()


def test_non_standard_json_is_quarantined(tmp_path):
    manager = StorageManager(tmp_path)
    manager.crawl_path.write_text(
        '{"schema_version":3,"pages":NaN}',
        encoding="utf-8",
    )

    assert manager.load_crawl_state() == CrawlState.empty()
    assert not manager.crawl_path.exists()
    assert list(tmp_path.glob("crawl_state.json.corrupt-*"))


def test_fifo_member_is_refused_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform.")
    manager = StorageManager(tmp_path)
    os.mkfifo(manager.crawl_path)

    assert manager.load_crawl_state() == CrawlState.empty()
    assert manager.crawl_path.exists()


def test_oversized_member_is_quarantined(tmp_path, monkeypatch):
    manager = StorageManager(tmp_path)
    monkeypatch.setattr(manager, "max_snapshot_file_bytes", 20)
    manager.crawl_path.write_text("x" * 21, encoding="utf-8")

    assert manager.load_crawl_state() == CrawlState.empty()
    assert not manager.crawl_path.exists()
    assert list(tmp_path.glob("crawl_state.json.corrupt-*"))
