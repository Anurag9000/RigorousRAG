"""Dataset-card, checksum, split, and license governance for RAG experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    path: str
    sha256: str
    examples: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.path.strip():
            raise ValueError("split name and path are required.")
        checksum = self.sha256.lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError("split sha256 must be exactly 64 hexadecimal characters.")
        if self.examples is not None and (
            isinstance(self.examples, bool) or int(self.examples) != self.examples or self.examples < 0
        ):
            raise ValueError("examples must be a non-negative integer when supplied.")


@dataclass(frozen=True)
class DatasetCard:
    dataset_id: str
    version: str
    license_id: str
    homepage: str = ""
    citation: str = ""
    languages: Tuple[str, ...] = ()
    modalities: Tuple[str, ...] = ("text",)
    tasks: Tuple[str, ...] = ()
    splits: Tuple[DatasetSplit, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip() or not self.version.strip() or not self.license_id.strip():
            raise ValueError("dataset_id, version, and license_id are required.")
        names = [split.name for split in self.splits]
        if len(set(names)) != len(names):
            raise ValueError("dataset split names must be unique.")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DatasetRegistry:
    """Persistent registry that refuses silent dataset-card mutation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cards: dict[tuple[str, str], DatasetCard] = {}
        if self.path.exists():
            self._load()

    def register(self, card: DatasetCard) -> None:
        key = (card.dataset_id, card.version)
        existing = self._cards.get(key)
        if existing is not None and existing != card:
            raise ValueError("a different card is already registered for this dataset/version.")
        self._cards[key] = card
        self._persist()

    def get(self, dataset_id: str, version: str) -> DatasetCard:
        return self._cards[(dataset_id, version)]

    def versions(self, dataset_id: str) -> Tuple[DatasetCard, ...]:
        values = [card for (name, _), card in self._cards.items() if name == dataset_id]
        values.sort(key=lambda card: card.version)
        return tuple(values)

    def verify_local_split(
        self,
        dataset_id: str,
        version: str,
        split_name: str,
        *,
        base_dir: str | Path = ".",
        max_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> bool:
        if isinstance(max_bytes, bool) or int(max_bytes) != max_bytes or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        card = self.get(dataset_id, version)
        matches = [split for split in card.splits if split.name == split_name]
        if len(matches) != 1:
            raise KeyError(f"unknown split {split_name!r}.")
        split = matches[0]
        root = Path(base_dir).resolve()
        candidate = (root / split.path).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError("dataset split path escapes the governed base directory.")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("dataset split exceeds max_bytes.")
                digest.update(block)
        return digest.hexdigest() == split.sha256.lower()

    def require_license(self, dataset_id: str, version: str, allowed: Tuple[str, ...]) -> None:
        card = self.get(dataset_id, version)
        if card.license_id not in set(allowed):
            raise PermissionError(
                f"dataset license {card.license_id!r} is not in the experiment allowlist."
            )

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(card) for card in sorted(self._cards.values(), key=lambda item: (item.dataset_id, item.version))]
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> None:
        raw_cards = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw_cards, list):
            raise ValueError("dataset registry must contain a JSON list.")
        for raw in raw_cards:
            splits = tuple(DatasetSplit(**item) for item in raw.get("splits", ()))
            card = DatasetCard(
                dataset_id=raw["dataset_id"],
                version=raw["version"],
                license_id=raw["license_id"],
                homepage=raw.get("homepage", ""),
                citation=raw.get("citation", ""),
                languages=tuple(raw.get("languages", ())),
                modalities=tuple(raw.get("modalities", ("text",))),
                tasks=tuple(raw.get("tasks", ())),
                splits=splits,
                metadata=dict(raw.get("metadata", {})),
            )
            key = (card.dataset_id, card.version)
            if key in self._cards:
                raise ValueError("duplicate dataset/version entry in registry.")
            self._cards[key] = card
