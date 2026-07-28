import os
import subprocess
import sys


def _run_import(script: str, environment: dict[str, str]):
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_security_import_survives_malformed_network_budgets():
    result = _run_import(
        """
import tools.security as module
assert module.DEFAULT_MAX_DOWNLOAD_BYTES == 5_000_000
assert module.DEFAULT_MAX_UPLOAD_BYTES == 50_000_000
assert module.DEFAULT_REQUEST_TIMEOUT == 15.0
assert module.MAX_REDIRECTS == 4
""",
        {
            "MAX_REMOTE_DOWNLOAD_BYTES": "bad",
            "MAX_UPLOAD_BYTES": "bad",
            "REMOTE_REQUEST_TIMEOUT_SECONDS": "nan",
            "MAX_REMOTE_REDIRECTS": "infinity",
        },
    )

    assert result.returncode == 0, result.stderr


def test_ingestion_import_survives_malformed_parser_budgets():
    result = _run_import(
        """
import os
import tools.ingestion as module
assert module._OCR_MAX_PAGES == 50
assert module._OCR_DPI == 200
assert module._OCR_TIMEOUT_SECONDS == 30
assert module._MAX_PDF_PAGES == 2000
assert module._MAX_DOCX_MEMBERS == 10000
assert module._MAX_DOCX_COMPRESSION_RATIO == 1000.0
assert os.environ['OCR_MAX_PAGES'] == '50'
""",
        {
            "MAX_UPLOAD_BYTES": "bad",
            "OCR_MAX_PAGES": "bad",
            "OCR_DPI": "bad",
            "OCR_TIMEOUT_SECONDS": "bad",
            "OCR_MIN_TEXT_CHARS": "bad",
            "MAX_PDF_PAGES": "bad",
            "MAX_PDF_RENDER_PIXELS": "bad",
            "MAX_EXTRACTED_CHARS": "bad",
            "MAX_DOCX_MEMBERS": "bad",
            "MAX_DOCX_UNCOMPRESSED_BYTES": "bad",
            "MAX_DOCX_COMPRESSION_RATIO": "nan",
        },
    )

    assert result.returncode == 0, result.stderr


def test_bounded_configuration_helper_clamps_and_writes_back():
    result = _run_import(
        """
import os
from tools.config import bounded_float_env, bounded_int_env
assert bounded_int_env('VALUE_INT', 5, minimum=1, maximum=10, write_back=True) == 10
assert os.environ['VALUE_INT'] == '10'
assert bounded_float_env('VALUE_FLOAT', 2.5, minimum=0.1, maximum=5.0, write_back=True) == 2.5
assert os.environ['VALUE_FLOAT'] == '2.5'
""",
        {
            "VALUE_INT": "999999",
            "VALUE_FLOAT": "nan",
        },
    )

    assert result.returncode == 0, result.stderr
