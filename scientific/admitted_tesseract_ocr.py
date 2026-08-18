"""Supply-chain-admitted Tesseract OCR adapter for authoritative extraction.

The low-level ``TesseractOCRProvider`` intentionally uses the host-configured pytesseract
runtime.  This authoritative variant instead binds both the exact executable bytes and the
exact ``tessdata`` model tree to admitted artifact proofs, re-verifies them immediately
before every OCR invocation, and invokes the binary directly without a shell.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.admitted_local_executable import AdmittedLocalExecutable, require_admitted_local_executable
from models.admitted_local_tree import AdmittedLocalArtifactTree

_MAX_IMAGE_PIXELS = 100_000_000
_MAX_TSV_BYTES = 64 * 1024 * 1024
_MAX_STDERR_BYTES = 2 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]") from exc
    if not math.isfinite(selected) or not minimum <= selected <= maximum:
        raise ValueError(f"{label} must be finite and in [{minimum}, {maximum}]")
    return selected


@dataclass(frozen=True)
class TesseractOCRContract:
    language: str = "eng"
    page_segmentation_mode: int = 3
    engine_mode: int = 3
    timeout_seconds: float = 30.0
    max_image_pixels: int = 40_000_000

    def __post_init__(self) -> None:
        language = _text(self.language, "language", 100)
        # Tesseract accepts language combinations like eng+deu.  Keep the source contract
        # path-like/control-free so it cannot be repurposed as an arbitrary CLI argument.
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-")
        if any(ch not in allowed for ch in language):
            raise ValueError("language contains unsupported characters")
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "page_segmentation_mode", _bounded_int(self.page_segmentation_mode, "page_segmentation_mode", 0, 13))
        object.__setattr__(self, "engine_mode", _bounded_int(self.engine_mode, "engine_mode", 0, 3))
        object.__setattr__(self, "timeout_seconds", _bounded_float(self.timeout_seconds, "timeout_seconds", 0.1, 600.0))
        object.__setattr__(self, "max_image_pixels", _bounded_int(self.max_image_pixels, "max_image_pixels", 1, _MAX_IMAGE_PIXELS))

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-admitted-tesseract-contract/v1",
                "language": self.language,
                "page_segmentation_mode": self.page_segmentation_mode,
                "engine_mode": self.engine_mode,
                "timeout_seconds": self.timeout_seconds,
                "max_image_pixels": self.max_image_pixels,
            }
        )


class AdmittedTesseractOCRProvider:
    """OCR provider whose engine and language-model tree are admitted exact artifacts."""

    def __init__(
        self,
        *,
        executable: AdmittedLocalExecutable,
        tessdata: AdmittedLocalArtifactTree,
        contract: TesseractOCRContract = TesseractOCRContract(),
    ) -> None:
        if not isinstance(executable, AdmittedLocalExecutable):
            raise ValueError("executable must be AdmittedLocalExecutable")
        if not isinstance(tessdata, AdmittedLocalArtifactTree):
            raise ValueError("tessdata must be AdmittedLocalArtifactTree")
        if not isinstance(contract, TesseractOCRContract):
            raise ValueError("contract must be TesseractOCRContract")
        # Fail early at construction and re-check again immediately before each invocation.
        executable.verify()
        tessdata.verify(required_artifact_type="model")
        self.executable = executable
        self.tessdata = tessdata
        self.contract = contract

    @property
    def artifact_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-tesseract-artifact-bundle/v1",
                "executable_binding_sha256": self.executable.binding_sha256,
                "tessdata_binding_sha256": self.tessdata.binding_sha256,
            }
        )

    @property
    def contract_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-admitted-tesseract-provider/v1",
                "artifact_sha256": self.artifact_sha256,
                "ocr_contract_sha256": self.contract.contract_sha256,
            }
        )

    def _encode_image(self, image: Any, path: Path) -> None:
        size = getattr(image, "size", None)
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("image must expose PIL-compatible .size")
        width, height = int(size[0]), int(size[1])
        if width < 1 or height < 1 or width * height > self.contract.max_image_pixels:
            raise ValueError("image dimensions exceed admitted OCR contract")
        if not hasattr(image, "save"):
            raise ValueError("image must expose a PIL-compatible save method")
        try:
            image.save(path, format="PNG")
        except Exception as exc:
            raise RuntimeError("could not encode image for OCR") from exc
        if not path.is_file() or path.stat().st_size < 1:
            raise RuntimeError("OCR image encoding produced no regular input file")

    @staticmethod
    def _parse_tsv(value: bytes) -> tuple[str, float | None]:
        if len(value) > _MAX_TSV_BYTES:
            raise RuntimeError("Tesseract TSV output exceeds configured bound")
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("Tesseract TSV output is not UTF-8") from exc
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        required = {"text", "conf"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError("Tesseract TSV output lacks required columns")
        words: list[str] = []
        confidences: list[float] = []
        for row in reader:
            raw = row.get("text")
            selected = " ".join(str(raw or "").split())
            if not selected:
                continue
            words.append(selected)
            if sum(len(word) for word in words) > 20_000_000:
                raise RuntimeError("Tesseract recognized text exceeds configured bound")
            try:
                confidence = float(row.get("conf", ""))
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(confidence) and confidence >= 0.0:
                confidences.append(min(max(confidence / 100.0, 0.0), 1.0))
        return " ".join(words), (sum(confidences) / len(confidences) if confidences else None)

    def recognize(self, image: Any) -> tuple[str, float | None]:
        executable = require_admitted_local_executable(self.executable)
        tessdata = self.tessdata.verify(required_artifact_type="model")
        with tempfile.TemporaryDirectory(prefix="rigorousrag-ocr-") as directory:
            root = Path(directory)
            image_path = root / "input.png"
            output_base = root / "ocr"
            stderr_path = root / "stderr.log"
            self._encode_image(image, image_path)
            command = [
                executable,
                str(image_path),
                str(output_base),
                "--tessdata-dir",
                tessdata,
                "-l",
                self.contract.language,
                "--psm",
                str(self.contract.page_segmentation_mode),
                "--oem",
                str(self.contract.engine_mode),
                "tsv",
            ]
            try:
                with stderr_path.open("wb") as stderr_handle:
                    completed = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_handle,
                        shell=False,
                        check=False,
                        timeout=self.contract.timeout_seconds,
                        cwd=str(root),
                    )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Tesseract OCR exceeded configured timeout") from exc
            except OSError as exc:
                raise RuntimeError("could not execute admitted Tesseract binary") from exc
            if stderr_path.exists() and stderr_path.stat().st_size > _MAX_STDERR_BYTES:
                raise RuntimeError("Tesseract stderr exceeds configured bound")
            if completed.returncode != 0:
                raise RuntimeError("admitted Tesseract execution failed")
            output_path = Path(f"{output_base}.tsv")
            if not output_path.is_file():
                raise RuntimeError("Tesseract did not produce expected TSV output")
            if output_path.stat().st_size > _MAX_TSV_BYTES:
                raise RuntimeError("Tesseract TSV output exceeds configured bound")
            try:
                output = output_path.read_bytes()
            except OSError as exc:
                raise RuntimeError("could not read Tesseract TSV output") from exc
            return self._parse_tsv(output)


__all__ = ["AdmittedTesseractOCRProvider", "TesseractOCRContract"]
