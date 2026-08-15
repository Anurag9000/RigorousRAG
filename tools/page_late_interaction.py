"""Page-native late-interaction retrieval primitives (ColPali/ColQwen-style contract).

No model is downloaded here.  A backend supplies bounded token/patch vectors for query
and rendered pages; this module validates immutable artifacts and performs deterministic
MaxSim scoring plus document->page->region selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.multimodal_evidence import EvidenceRegion
from tools.security import normalize_owner_id

_MAX_DIM = 8192
_MAX_QUERY_TOKENS = 2048
_MAX_PAGE_PATCHES = 16384
_MAX_PAGES = 100_000
_MAX_REGIONS = 100_000


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    result = value.strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _vector(values: Sequence[Any], label: str, *, expected_dim: int | None = None) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric vector")
    if not 1 <= len(values) <= _MAX_DIM:
        raise ValueError(f"{label} has an invalid dimension")
    if expected_dim is not None and len(values) != expected_dim:
        raise ValueError(f"{label} dimension mismatch")
    result: list[float] = []
    norm_sq = 0.0
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{label} contains a non-numeric value")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} contains a non-numeric value") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{label} contains a non-finite value")
        result.append(parsed)
        norm_sq += parsed * parsed
    if norm_sq <= 0.0:
        raise ValueError(f"{label} may not be the zero vector")
    norm = math.sqrt(norm_sq)
    return tuple(value / norm for value in result)


def _matrix(rows: Sequence[Sequence[Any]], label: str, maximum_rows: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(rows, (str, bytes, bytearray)) or not 1 <= len(rows) <= maximum_rows:
        raise ValueError(f"{label} row count is invalid")
    first = _vector(rows[0], f"{label}[0]")
    dim = len(first)
    output = [first]
    for index, row in enumerate(rows[1:], start=1):
        output.append(_vector(row, f"{label}[{index}]", expected_dim=dim))
    return tuple(output)


@dataclass(frozen=True)
class QueryTokenEmbeddings:
    query_sha256: str
    model_id: str
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_sha256", _digest(self.query_sha256, "query_sha256"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", 300))
        object.__setattr__(self, "vectors", _matrix(self.vectors, "query vectors", _MAX_QUERY_TOKENS))

    @property
    def dimension(self) -> int:
        return len(self.vectors[0])


@dataclass(frozen=True)
class PageEmbeddingArtifact:
    owner_id: str
    doc_id: str
    source_sha256: str
    page_number: int
    rendered_page_sha256: str
    model_id: str
    patch_vectors: tuple[tuple[float, ...], ...]
    patch_region_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _text(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source_sha256"))
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or not 1 <= self.page_number <= _MAX_PAGES:
            raise ValueError("page_number is invalid")
        object.__setattr__(self, "rendered_page_sha256", _digest(self.rendered_page_sha256, "rendered_page_sha256"))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", 300))
        object.__setattr__(self, "patch_vectors", _matrix(self.patch_vectors, "patch vectors", _MAX_PAGE_PATCHES))
        if self.patch_region_ids:
            if len(self.patch_region_ids) != len(self.patch_vectors):
                raise ValueError("patch_region_ids must align one-to-one with patch vectors")
            object.__setattr__(self, "patch_region_ids", tuple(_digest(item, "patch_region_id") for item in self.patch_region_ids))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {str(k)[:100]: str(v)[:500] for k, v in self.metadata.items()})

    @property
    def dimension(self) -> int:
        return len(self.patch_vectors[0])

    @property
    def artifact_sha256(self) -> str:
        payload = {
            "contract": "rigorousrag-page-late-interaction-v1",
            "owner_id": self.owner_id,
            "doc_id": self.doc_id,
            "source_sha256": self.source_sha256,
            "page_number": self.page_number,
            "rendered_page_sha256": self.rendered_page_sha256,
            "model_id": self.model_id,
            "patch_vectors": self.patch_vectors,
            "patch_region_ids": self.patch_region_ids,
            "metadata": dict(self.metadata),
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


class PageEmbeddingBackend(Protocol):
    @property
    def model_id(self) -> str: ...
    def embed_query(self, query: str) -> Sequence[Sequence[float]]: ...
    def embed_page(self, rendered_page: bytes, *, page_number: int) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class LateInteractionScore:
    page_artifact_sha256: str
    score: float
    token_maxima: tuple[float, ...]
    winning_patch_indices: tuple[int, ...]
    winning_region_ids: tuple[str, ...]


def maxsim_score(query: QueryTokenEmbeddings, page: PageEmbeddingArtifact) -> LateInteractionScore:
    if query.model_id != page.model_id:
        raise ValueError("query and page embeddings must use the same model_id")
    if query.dimension != page.dimension:
        raise ValueError("query and page embedding dimensions differ")
    maxima: list[float] = []
    winners: list[int] = []
    regions: list[str] = []
    for query_vector in query.vectors:
        best_score = -math.inf
        best_index = 0
        for index, patch_vector in enumerate(page.patch_vectors):
            score = sum(q * p for q, p in zip(query_vector, patch_vector))
            if score > best_score:
                best_score = score
                best_index = index
        maxima.append(best_score)
        winners.append(best_index)
        if page.patch_region_ids:
            regions.append(page.patch_region_ids[best_index])
    total = sum(maxima)
    normalized = total / len(maxima) if maxima else 0.0
    return LateInteractionScore(page.artifact_sha256, normalized, tuple(maxima), tuple(winners), tuple(regions))


@dataclass(frozen=True)
class PageSearchHit:
    doc_id: str
    page_number: int
    page_artifact_sha256: str
    score: float
    winning_region_ids: tuple[str, ...]


def rank_pages(
    query: QueryTokenEmbeddings,
    pages: Sequence[PageEmbeddingArtifact],
    *,
    top_k: int = 20,
    max_pages_per_document: int = 4,
) -> tuple[PageSearchHit, ...]:
    if len(pages) > _MAX_PAGES:
        raise ValueError("too many pages")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
        raise ValueError("top_k is invalid")
    if isinstance(max_pages_per_document, bool) or not isinstance(max_pages_per_document, int) or not 1 <= max_pages_per_document <= 100:
        raise ValueError("max_pages_per_document is invalid")
    scored: list[tuple[PageEmbeddingArtifact, LateInteractionScore]] = []
    for page in pages:
        scored.append((page, maxsim_score(query, page)))
    scored.sort(key=lambda item: (-item[1].score, item[0].doc_id, item[0].page_number))
    counts: dict[str, int] = {}
    output: list[PageSearchHit] = []
    for page, result in scored:
        if counts.get(page.doc_id, 0) >= max_pages_per_document:
            continue
        counts[page.doc_id] = counts.get(page.doc_id, 0) + 1
        output.append(PageSearchHit(page.doc_id, page.page_number, page.artifact_sha256, result.score, tuple(dict.fromkeys(result.winning_region_ids))))
        if len(output) >= top_k:
            break
    return tuple(output)


def select_regions(
    hits: Sequence[PageSearchHit],
    regions: Sequence[EvidenceRegion],
    *,
    max_regions: int = 50,
    allowed_kinds: Sequence[str] = ("text", "table", "figure", "caption", "chart", "equation"),
) -> tuple[EvidenceRegion, ...]:
    if len(regions) > _MAX_REGIONS:
        raise ValueError("too many regions")
    if isinstance(max_regions, bool) or not isinstance(max_regions, int) or not 1 <= max_regions <= 1000:
        raise ValueError("max_regions is invalid")
    allowed = frozenset(str(item).strip().lower() for item in allowed_kinds)
    hit_keys = {(hit.doc_id, hit.page_number): hit for hit in hits}
    winning = {region_id for hit in hits for region_id in hit.winning_region_ids}
    ranked: list[tuple[int, float, EvidenceRegion]] = []
    for region in regions:
        hit = hit_keys.get((region.doc_id, region.page_number))
        if hit is None or region.kind not in allowed:
            continue
        priority = 0 if region.region_id in winning else 1
        ranked.append((priority, -hit.score, region))
    ranked.sort(key=lambda item: (item[0], item[1], item[2].doc_id, item[2].page_number, item[2].bbox.y0, item[2].bbox.x0))
    selected: list[EvidenceRegion] = []
    seen: set[str] = set()
    for _, _, region in ranked:
        if region.region_id in seen:
            continue
        seen.add(region.region_id)
        selected.append(region)
        if len(selected) >= max_regions:
            break
    return tuple(selected)


def query_embeddings(query: str, backend: PageEmbeddingBackend) -> QueryTokenEmbeddings:
    normalized = _text(query, "query", 20_000)
    vectors = backend.embed_query(normalized)
    return QueryTokenEmbeddings(hashlib.sha256(normalized.encode("utf-8")).hexdigest(), _text(backend.model_id, "model_id", 300), tuple(tuple(row) for row in vectors))


__all__ = [
    "LateInteractionScore",
    "PageEmbeddingArtifact",
    "PageEmbeddingBackend",
    "PageSearchHit",
    "QueryTokenEmbeddings",
    "maxsim_score",
    "query_embeddings",
    "rank_pages",
    "select_regions",
]
