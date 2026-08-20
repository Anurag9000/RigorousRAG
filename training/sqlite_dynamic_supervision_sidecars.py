"""Streaming, restart-verifiable dynamic-RAG supervision sidecars.

Historical JSON-array sidecars remain readable through their original providers. Production-scale
sidecars use JSONL and are published once into a closed SQLite authority directory. The source
bytes, semantic contract, canonical row set, index bytes and receipt are all bound. Provider
lookups are exact `(episode_id, step_id)` SQL queries, so annotation/gain/value/counterfactual
supervision never requires a corpus-sized Python dictionary.

Supported JSONL rows (one object per non-blank line):
- information_need: {episode_id, step_id, spans:[{start,end}, ...]}
- realized_gain:    {episode_id, step_id, gain:number}
- logged_value:     {episode_id, step_id, value:number}
- counterfactual:   {episode_id, step_id, utilities:{action:number, ...}}

Numeric sidecars require an explicit semantic-contract SHA-256. Importing this module performs no
model execution, download, retrieval or training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_data import DynamicRagEpisodeStep, TextSpan
from training.dynamic_retrieval_policy import DynamicRetrievalAction

_KINDS = frozenset({"information_need", "realized_gain", "logged_value", "counterfactual"})
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_SPANS = 1_000_000
_MAX_ACTIONS = 64
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _kind(value: Any) -> str:
    selected = _identifier(value, "sidecar kind", 100).lower()
    if selected not in _KINDS:
        raise ValueError(f"unsupported dynamic supervision sidecar kind {selected!r}")
    return selected


def _atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _strict_json_bytes(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _pair(raw: Mapping[str, Any], required: set[str], label: str) -> tuple[str, str]:
    if set(raw) != required:
        raise ValueError(f"{label} fields are invalid")
    return _identifier(raw["episode_id"], f"{label}.episode_id"), _identifier(raw["step_id"], f"{label}.step_id")


def _normalize_row(kind: str, raw: Mapping[str, Any], line_number: int) -> tuple[str, str, Mapping[str, Any]]:
    label = f"dynamic supervision {kind} line {line_number}"
    if kind == "information_need":
        episode_id, step_id = _pair(raw, {"episode_id", "step_id", "spans"}, label)
        spans_raw = raw["spans"]
        if not isinstance(spans_raw, list) or len(spans_raw) > _MAX_SPANS:
            raise ValueError(f"{label}.spans must be a bounded array")
        spans = []
        for index, item in enumerate(spans_raw):
            if not isinstance(item, Mapping) or set(item) != {"start", "end"}:
                raise ValueError(f"{label}.spans[{index}] must be a closed start/end object")
            span = TextSpan(start=item["start"], end=item["end"])
            spans.append({"start": span.start, "end": span.end})
        payload = {"spans": spans}
    elif kind == "realized_gain":
        episode_id, step_id = _pair(raw, {"episode_id", "step_id", "gain"}, label)
        payload = {"gain": _finite(raw["gain"], f"{label}.gain")}
    elif kind == "logged_value":
        episode_id, step_id = _pair(raw, {"episode_id", "step_id", "value"}, label)
        payload = {"value": _finite(raw["value"], f"{label}.value")}
    else:
        episode_id, step_id = _pair(raw, {"episode_id", "step_id", "utilities"}, label)
        utilities_raw = raw["utilities"]
        if not isinstance(utilities_raw, Mapping) or not utilities_raw or len(utilities_raw) > _MAX_ACTIONS:
            raise ValueError(f"{label}.utilities must be a bounded non-empty object")
        utilities: dict[str, float] = {}
        for raw_action, raw_value in utilities_raw.items():
            action = DynamicRetrievalAction(raw_action)
            if action.value in utilities:
                raise ValueError(f"{label} duplicates action {action.value}")
            utilities[action.value] = _finite(raw_value, f"{label}.utilities[{action.value}]")
        payload = {"utilities": dict(sorted(utilities.items()))}
    return episode_id, step_id, payload


def _row_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for episode_id, step_id, payload_json in connection.execute(
        "SELECT episode_id,step_id,payload_json FROM entries ORDER BY episode_id COLLATE BINARY,step_id COLLATE BINARY"
    ):
        digest.update(_canonical({"episode_id": episode_id, "step_id": step_id, "payload": json.loads(payload_json)}))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class SqliteDynamicSidecarReceipt:
    kind: str
    source_path: str
    source_sha256: str
    semantic_contract_sha256: str | None
    record_count: int
    row_digest_sha256: str
    index_sha256: str
    provider_contract_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _kind(self.kind))
        source = safe_advanced_path(self.source_path, label="dynamic supervision sidecar source", must_exist=True, require_file=True)
        object.__setattr__(self, "source_path", str(source))
        for field in ("source_sha256", "row_digest_sha256", "index_sha256", "provider_contract_sha256", "receipt_sha256"):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        if self.semantic_contract_sha256 is not None:
            object.__setattr__(self, "semantic_contract_sha256", _sha(self.semantic_contract_sha256, "semantic_contract_sha256"))
        if self.kind != "information_need" and self.semantic_contract_sha256 is None:
            raise ValueError("numeric dynamic supervision sidecars require semantic_contract_sha256")
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("dynamic supervision sidecar record_count must be positive")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("dynamic supervision sidecar receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-sqlite-dynamic-supervision-sidecar-receipt/v1",
            "kind": self.kind,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "record_count": self.record_count,
            "row_digest_sha256": self.row_digest_sha256,
            "index_sha256": self.index_sha256,
            "provider_contract_sha256": self.provider_contract_sha256,
        }


def _provider_contract(*, kind: str, source_sha256: str, semantic_contract_sha256: str | None, record_count: int, row_digest_sha256: str) -> str:
    semantics = {
        "information_need": "character_spans_over_exact_record_context_with_explicit_empty_negative_exact_pair_keying",
        "realized_gain": "measured_post_action_utility_delta_keyed_by_exact_episode_step_pair",
        "logged_value": "logged_state_value_keyed_by_exact_episode_step_pair",
        "counterfactual": "per_legal_action_utility_keyed_by_exact_episode_step_pair",
    }[kind]
    return _digest({
        "schema": "rigorousrag-sqlite-dynamic-supervision-provider/v1",
        "kind": kind,
        "source_sha256": source_sha256,
        "semantic_contract_sha256": semantic_contract_sha256,
        "record_count": record_count,
        "row_digest_sha256": row_digest_sha256,
        "semantics": semantics,
    })


def publish_sqlite_dynamic_supervision_sidecar(
    source_path: str | Path,
    *,
    expected_sha256: str,
    kind: str,
    output_dir: str | Path,
    semantic_contract_sha256: str | None = None,
) -> SqliteDynamicSidecarReceipt:
    selected_kind = _kind(kind)
    source = safe_advanced_path(source_path, label="dynamic supervision JSONL source", must_exist=True, require_file=True)
    expected_source_sha = _sha(expected_sha256, "expected sidecar sha256")
    actual_source_sha = _stream_sha(source)
    if actual_source_sha != expected_source_sha:
        raise ValueError("dynamic supervision sidecar source digest mismatch")
    semantic_sha = None if semantic_contract_sha256 is None else _sha(semantic_contract_sha256, "semantic_contract_sha256")
    if selected_kind != "information_need" and semantic_sha is None:
        raise ValueError("numeric dynamic supervision sidecars require semantic_contract_sha256")

    root = safe_advanced_path(output_dir, label="dynamic supervision sidecar authority output", must_exist=False)
    if root.exists():
        raise ValueError("dynamic supervision sidecar authority output must not already exist")
    parent = safe_advanced_path(root.parent, label="dynamic supervision sidecar authority parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or selected_kind}-stage-", dir=parent))
    database = stage / "authority.sqlite"
    connection = sqlite3.connect(str(database), timeout=30.0)
    published = False
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("CREATE TABLE entries(episode_id TEXT NOT NULL,step_id TEXT NOT NULL,payload_json TEXT NOT NULL,PRIMARY KEY(episode_id,step_id)) WITHOUT ROWID")
        count = 0
        with source.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if len(line) > _MAX_LINE_BYTES:
                    raise ValueError(f"dynamic supervision line {line_number} exceeds byte safety bound")
                if count >= _MAX_RECORDS:
                    raise ValueError("dynamic supervision sidecar exceeds record safety bound")
                episode_id, step_id, payload = _normalize_row(selected_kind, _strict_json_bytes(line, f"dynamic supervision line {line_number}"), line_number)
                try:
                    connection.execute("INSERT INTO entries(episode_id,step_id,payload_json) VALUES(?,?,?)", (episode_id, step_id, _canonical(payload).decode("utf-8")))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate dynamic supervision identity {(episode_id, step_id)!r}") from exc
                count += 1
                if count % 10_000 == 0:
                    connection.commit()
        if count <= 0:
            raise ValueError("dynamic supervision sidecar may not be empty")
        connection.commit()
        row_sha = _row_digest(connection)
        connection.close()
        index_sha = _stream_sha(database)
        contract_sha = _provider_contract(kind=selected_kind, source_sha256=actual_source_sha, semantic_contract_sha256=semantic_sha, record_count=count, row_digest_sha256=row_sha)
        unsigned = {
            "schema": "rigorousrag-sqlite-dynamic-supervision-sidecar-receipt/v1",
            "kind": selected_kind,
            "source_path": str(source),
            "source_sha256": actual_source_sha,
            "semantic_contract_sha256": semantic_sha,
            "record_count": count,
            "row_digest_sha256": row_sha,
            "index_sha256": index_sha,
            "provider_contract_sha256": contract_sha,
        }
        receipt = SqliteDynamicSidecarReceipt(
            kind=selected_kind,
            source_path=str(source),
            source_sha256=actual_source_sha,
            semantic_contract_sha256=semantic_sha,
            record_count=count,
            row_digest_sha256=row_sha,
            index_sha256=index_sha,
            provider_contract_sha256=contract_sha,
            receipt_sha256=_digest(unsigned),
        )
        _atomic(stage / "sidecar_receipt.json", _canonical({**receipt.unsigned(), "receipt_sha256": receipt.receipt_sha256}) + b"\n")
        if {item.name for item in stage.iterdir()} != {"authority.sqlite", "sidecar_receipt.json"}:
            raise RuntimeError("dynamic supervision sidecar authority directory is not closed")
        os.replace(stage, root)
        published = True
        verified = verify_sqlite_dynamic_supervision_sidecar(root / "sidecar_receipt.json")
        if verified.receipt_sha256 != receipt.receipt_sha256:
            shutil.rmtree(root, ignore_errors=True)
            published = False
            raise RuntimeError("dynamic supervision sidecar identity changed after publication")
        return verified
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_sqlite_dynamic_supervision_sidecar(path: str | Path) -> SqliteDynamicSidecarReceipt:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("dynamic supervision sidecar receipt may not be a symlink")
    receipt_path = safe_advanced_path(raw_path, label="dynamic supervision sidecar receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / "sidecar_receipt.json":
        raise ValueError("dynamic supervision sidecar receipt must use canonical filename")
    if {item.name for item in root.iterdir()} != {"authority.sqlite", "sidecar_receipt.json"}:
        raise ValueError("dynamic supervision sidecar authority directory is not closed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("dynamic supervision sidecar authority contains non-regular child")
    if receipt_path.stat().st_size <= 0 or receipt_path.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("dynamic supervision sidecar receipt exceeds byte safety bound")
    raw = _strict_json_bytes(receipt_path.read_bytes(), "dynamic supervision sidecar receipt")
    expected = {"schema", "kind", "source_path", "source_sha256", "semantic_contract_sha256", "record_count", "row_digest_sha256", "index_sha256", "provider_contract_sha256", "receipt_sha256"}
    if set(raw) != expected or raw.get("schema") != "rigorousrag-sqlite-dynamic-supervision-sidecar-receipt/v1":
        raise ValueError("unsupported dynamic supervision sidecar receipt schema")
    receipt = SqliteDynamicSidecarReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    if _stream_sha(Path(receipt.source_path)) != receipt.source_sha256:
        raise ValueError("dynamic supervision source bytes changed after indexing")
    database = root / "authority.sqlite"
    if _stream_sha(database) != receipt.index_sha256:
        raise ValueError("dynamic supervision SQLite bytes differ from receipt")
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30.0) as connection:
        row = connection.execute("SELECT COUNT(*) FROM entries").fetchone()
        count = int(row[0])
        row_sha = _row_digest(connection)
    if count != receipt.record_count or row_sha != receipt.row_digest_sha256:
        raise ValueError("dynamic supervision SQLite row authority differs from receipt")
    expected_contract = _provider_contract(kind=receipt.kind, source_sha256=receipt.source_sha256, semantic_contract_sha256=receipt.semantic_contract_sha256, record_count=receipt.record_count, row_digest_sha256=receipt.row_digest_sha256)
    if expected_contract != receipt.provider_contract_sha256:
        raise ValueError("dynamic supervision provider contract differs from receipt")
    return receipt


class _SqliteDynamicProvider:
    expected_kind: str

    def __init__(self, receipt_path: str | Path) -> None:
        receipt = verify_sqlite_dynamic_supervision_sidecar(receipt_path)
        if receipt.kind != self.expected_kind:
            raise ValueError(f"sidecar kind {receipt.kind!r} cannot back {type(self).__name__}")
        selected = safe_advanced_path(receipt_path, label="dynamic supervision sidecar receipt", must_exist=True, require_file=True)
        self.receipt = receipt
        self.database_path = str(selected.parent / "authority.sqlite")

    @property
    def contract_sha256(self) -> str:
        return self.receipt.provider_contract_sha256

    def _payload(self, episode_id: str, step_id: str) -> Mapping[str, Any]:
        episode = _identifier(episode_id, "episode_id")
        step = _identifier(step_id, "step_id")
        with sqlite3.connect(f"file:{self.database_path}?mode=ro", uri=True, timeout=30.0) as connection:
            row = connection.execute("SELECT payload_json FROM entries WHERE episode_id=? AND step_id=?", (episode, step)).fetchone()
        if row is None:
            raise ValueError(f"dynamic supervision sidecar lacks step {(episode, step)!r}")
        value = json.loads(str(row[0]), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        if not isinstance(value, Mapping):
            raise ValueError("dynamic supervision indexed payload is malformed")
        return value


class SqliteInformationNeedAnnotationProvider(_SqliteDynamicProvider):
    expected_kind = "information_need"

    def spans(self, step: LegalDynamicRagEpisodeStep) -> Sequence[TextSpan]:
        payload = self._payload(step.episode_id, step.step_id)
        spans_raw = payload.get("spans")
        if not isinstance(spans_raw, list):
            raise ValueError("indexed information-need spans are malformed")
        spans = tuple(TextSpan(start=item["start"], end=item["end"]) for item in spans_raw)
        if any(span.end > len(step.context) for span in spans):
            raise ValueError("information-need span lies outside exact step context")
        return spans


class SqliteRealizedRetrievalGainProvider(_SqliteDynamicProvider):
    expected_kind = "realized_gain"

    def gains(self, steps: Sequence[LegalDynamicRagEpisodeStep]) -> Sequence[float]:
        return tuple(_finite(self._payload(step.episode_id, step.step_id)["gain"], "realized retrieval gain") for step in steps)


class SqliteLoggedValueProvider(_SqliteDynamicProvider):
    expected_kind = "logged_value"

    def values(self, steps: Sequence[DynamicRagEpisodeStep]) -> Sequence[float]:
        return tuple(_finite(self._payload(step.episode_id, step.step_id)["value"], "logged state value") for step in steps)


class SqliteCounterfactualActionProvider(_SqliteDynamicProvider):
    expected_kind = "counterfactual"

    def action_utilities(self, step: DynamicRagEpisodeStep) -> Mapping[DynamicRetrievalAction, float]:
        payload = self._payload(step.episode_id, step.step_id)
        raw = payload.get("utilities")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("indexed counterfactual utilities are malformed")
        return {DynamicRetrievalAction(action): _finite(value, f"counterfactual utility {action}") for action, value in raw.items()}


__all__ = [
    "SqliteCounterfactualActionProvider",
    "SqliteDynamicSidecarReceipt",
    "SqliteInformationNeedAnnotationProvider",
    "SqliteLoggedValueProvider",
    "SqliteRealizedRetrievalGainProvider",
    "publish_sqlite_dynamic_supervision_sidecar",
    "verify_sqlite_dynamic_supervision_sidecar",
]
