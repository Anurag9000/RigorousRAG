"""Versioned registry for embedding, reranker, generator, and task adapters."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple


_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][A-Za-z0-9.-]+)?$")


def _version_key(value: str) -> Tuple[int, int, int, str]:
    match = _VERSION.match(value)
    if not match:
        raise ValueError("version must follow semantic version form MAJOR.MINOR.PATCH.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), value


@dataclass(frozen=True)
class AdapterVersion:
    name: str
    version: str
    kind: str
    artifact_uri: str
    checksum_sha256: str
    base_model: str = ""
    tags: Tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class AdapterRegistry:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._versions: Dict[str, Dict[str, AdapterVersion]] = {}
        self._active: Dict[str, str] = {}
        if self.path is not None and self.path.exists():
            self._load()

    @staticmethod
    def checksum_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def register(self, record: AdapterVersion, *, replace: bool = False) -> None:
        _version_key(record.version)
        if not record.name or not record.kind or not record.artifact_uri:
            raise ValueError("name, kind, and artifact_uri are required.")
        checksum = record.checksum_sha256.lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError("checksum_sha256 must contain exactly 64 hexadecimal characters.")
        bucket = self._versions.setdefault(record.name, {})
        if record.version in bucket and not replace:
            raise ValueError(f"{record.name}@{record.version} is already registered.")
        bucket[record.version] = record
        self._persist()

    def versions(self, name: str) -> Tuple[AdapterVersion, ...]:
        values = list(self._versions.get(name, {}).values())
        values.sort(key=lambda record: _version_key(record.version), reverse=True)
        return tuple(values)

    def latest(self, name: str) -> Optional[AdapterVersion]:
        values = self.versions(name)
        return values[0] if values else None

    def promote(self, name: str, version: str) -> AdapterVersion:
        try:
            record = self._versions[name][version]
        except KeyError as exc:
            raise KeyError(f"unknown adapter version {name}@{version}") from exc
        self._active[name] = version
        self._persist()
        return record

    def active(self, name: str) -> Optional[AdapterVersion]:
        version = self._active.get(name)
        return self._versions.get(name, {}).get(version) if version else None

    def rollback(self, name: str) -> AdapterVersion:
        current = self._active.get(name)
        versions = list(self.versions(name))
        if not versions:
            raise KeyError(name)
        if current is None:
            return self.promote(name, versions[0].version)
        ordered = [record.version for record in versions]
        try:
            index = ordered.index(current)
        except ValueError as exc:
            raise RuntimeError("active adapter version is not present in the registry.") from exc
        if index + 1 >= len(ordered):
            raise ValueError("no previous adapter version is available for rollback.")
        return self.promote(name, ordered[index + 1])

    def compatible(
        self,
        *,
        kind: Optional[str] = None,
        tags: Iterable[str] = (),
    ) -> Tuple[AdapterVersion, ...]:
        required_tags = set(tags)
        output = []
        for bucket in self._versions.values():
            for record in bucket.values():
                if kind is not None and record.kind != kind:
                    continue
                if not required_tags.issubset(record.tags):
                    continue
                output.append(record)
        output.sort(key=lambda record: (record.name, _version_key(record.version)))
        return tuple(output)

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self._active,
            "versions": [
                asdict(record)
                for name in sorted(self._versions)
                for record in self.versions(name)
            ],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._versions.clear()
        for raw in payload.get("versions", []):
            raw["tags"] = tuple(raw.get("tags") or ())
            record = AdapterVersion(**raw)
            self._versions.setdefault(record.name, {})[record.version] = record
        self._active = {
            str(name): str(version) for name, version in (payload.get("active") or {}).items()
        }
        for name, version in self._active.items():
            if version not in self._versions.get(name, {}):
                raise ValueError(f"active adapter {name}@{version} is missing from registry.")
