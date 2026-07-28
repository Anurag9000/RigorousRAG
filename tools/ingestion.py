"""Failure-safe configuration boundary over document ingestion.

The full parser, OCR, redaction, and archive implementation remains in
``ingestion_legacy``. This module normalizes parser budgets before importing it so
standalone CLI use cannot crash on malformed deployment environment values.
"""

from __future__ import annotations

import os
import sys

from tools.config import bounded_float_env, bounded_int_env

_INTEGER_BUDGETS = {
    "MAX_UPLOAD_BYTES": (50_000_000, 1, 1_000_000_000),
    "OCR_MAX_PAGES": (50, 1, 500),
    "OCR_DPI": (200, 100, 400),
    "OCR_TIMEOUT_SECONDS": (30, 1, 300),
    "OCR_MIN_TEXT_CHARS": (40, 0, 2000),
    "MAX_PDF_PAGES": (2000, 1, 10_000),
    "MAX_PDF_RENDER_PIXELS": (40_000_000, 1_000_000, 250_000_000),
    "MAX_EXTRACTED_CHARS": (5_000_000, 100_000, 50_000_000),
    "MAX_DOCX_MEMBERS": (10_000, 10, 100_000),
    "MAX_DOCX_UNCOMPRESSED_BYTES": (200_000_000, 1, 2_000_000_000),
}
for _name, (_default, _minimum, _maximum) in _INTEGER_BUDGETS.items():
    bounded_int_env(
        _name,
        _default,
        minimum=_minimum,
        maximum=_maximum,
        write_back=True,
    )
bounded_float_env(
    "MAX_DOCX_COMPRESSION_RATIO",
    1000.0,
    minimum=10.0,
    maximum=100_000.0,
    write_back=True,
)

from tools import ingestion_legacy as _implementation

_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
