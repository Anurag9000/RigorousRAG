"""Field-weighted owner-scoped BM25-style sparse search."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from tools.sparse_types import SparseMatch, SparseSearchHit
from tools.sparse_utils import (
    DEFAULT_FIELD_WEIGHTS,
    _MAX_MATCHES_PER_HIT,
    _MAX_QUERY_CHARS,
    _MAX_QUERY_TERMS,
    _MAX_RESULTS,
    _exact_int,
    _field_type,
    _finite,
    _identifier,
    _normalize_owner_id,
    _strict_json,
    tokenize,
)


class SparseSearchMixin:
    @staticmethod
    def _field_weights(custom: Mapping[str, float] | None) -> dict[str, float]:
        result = dict(DEFAULT_FIELD_WEIGHTS)
        if custom is None:
            return result
        if not isinstance(custom, Mapping) or len(custom) > 100:
            raise ValueError("field_weights must be a bounded mapping.")
        for raw_field, raw_weight in custom.items():
            field_name = _field_type(raw_field)
            result[field_name] = _finite(
                raw_weight,
                f"weight for {field_name}",
                minimum=0.0,
                maximum=100.0,
            )
        return result

    def search(
        self,
        query: str,
        *,
        owner_id: str,
        limit: int = 20,
        doc_id: str | None = None,
        field_types: Sequence[str] | None = None,
        field_weights: Mapping[str, float] | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[SparseSearchHit]:
        if not isinstance(query, str):
            raise ValueError("query must be a string.")
        bounded_query = query.strip()
        if len(bounded_query) > _MAX_QUERY_CHARS or any(
            ord(character) < 32 and character not in "\t\r\n"
            for character in bounded_query
        ):
            raise ValueError("query is invalid or too long.")
        terms = tuple(dict.fromkeys(tokenize(bounded_query)[:_MAX_QUERY_TERMS]))
        if not terms:
            return []
        owner = _normalize_owner_id(owner_id)
        result_limit = _exact_int(limit, "limit", minimum=1, maximum=_MAX_RESULTS)
        document_id = _identifier(doc_id, "doc_id") if doc_id is not None else None
        allowed_types = None
        if field_types is not None:
            if isinstance(field_types, (str, bytes, bytearray)) or len(field_types) > 100:
                raise ValueError("field_types must be a bounded sequence.")
            allowed_types = tuple(dict.fromkeys(_field_type(value) for value in field_types))
            if not allowed_types:
                return []
        weights = self._field_weights(field_weights)
        k1_value = _finite(k1, "k1", minimum=0.000001, maximum=10.0)
        b_value = _finite(b, "b", minimum=0.0, maximum=1.0)
        placeholders = ",".join("?" for _ in terms)
        clauses = ["p.owner_id=?", f"p.term IN ({placeholders})"]
        parameters: list[Any] = [owner, *terms]
        if document_id is not None:
            clauses.append("p.doc_id=?")
            parameters.append(document_id)
        if allowed_types is not None:
            clauses.append("f.field_type IN (" + ",".join("?" for _ in allowed_types) + ")")
            parameters.extend(allowed_types)
        where = " AND ".join(clauses)
        with self._lock, self._connect() as connection:
            corpus_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM sparse_documents WHERE owner_id=?",
                    (owner,),
                ).fetchone()[0]
            )
            if corpus_count <= 0:
                return []
            avg_rows = connection.execute(
                """SELECT field_type, AVG(token_count)
                   FROM sparse_fields WHERE owner_id=? GROUP BY field_type""",
                (owner,),
            ).fetchall()
            average_lengths = {str(row[0]): max(float(row[1] or 0.0), 1.0) for row in avg_rows}
            df_rows = connection.execute(
                f"""SELECT term, COUNT(DISTINCT doc_id)
                    FROM sparse_postings
                    WHERE owner_id=? AND term IN ({placeholders})
                    GROUP BY term""",
                (owner, *terms),
            ).fetchall()
            document_frequency = {str(row[0]): int(row[1]) for row in df_rows}
            rows = connection.execute(
                f"""SELECT p.doc_id, p.field_id, p.term, p.frequency, p.positions_json,
                           f.field_type, f.position, f.token_count, f.page_number,
                           f.section, f.metadata_json, d.generation,
                           d.profile_fingerprint, d.metadata_json
                    FROM sparse_postings p
                    JOIN sparse_fields f ON
                        f.owner_id=p.owner_id AND f.doc_id=p.doc_id AND f.field_id=p.field_id
                    JOIN sparse_documents d ON
                        d.owner_id=p.owner_id AND d.doc_id=p.doc_id
                    WHERE {where}""",
                parameters,
            ).fetchall()

        scores: defaultdict[str, float] = defaultdict(float)
        details: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        documents: dict[str, tuple[int, str, dict[str, Any]]] = {}
        invalid_docs: set[str] = set()
        for row in rows:
            current_doc_id = str(row[0])
            if current_doc_id in invalid_docs:
                continue
            try:
                document_metadata = _strict_json(str(row[13]), "sparse result document metadata")
                field_metadata = _strict_json(str(row[10]), "sparse result field metadata")
                positions_raw = json.loads(str(row[4]))
                frequency_value = int(row[3])
                field_token_count = int(row[7])
                if (
                    frequency_value <= 0
                    or not isinstance(positions_raw, list)
                    or len(positions_raw) != frequency_value
                    or positions_raw != sorted(set(positions_raw))
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or not 0 <= value < field_token_count
                        for value in positions_raw
                    )
                ):
                    raise ValueError("invalid positions")
            except Exception:
                invalid_docs.add(current_doc_id)
                scores.pop(current_doc_id, None)
                details.pop(current_doc_id, None)
                documents.pop(current_doc_id, None)
                continue
            documents[current_doc_id] = (
                int(row[11]),
                self._profile_fingerprint(str(row[12])),
                document_metadata,
            )
            field_name = str(row[5])
            weight = weights.get(field_name, 1.0 if field_name.startswith("custom:") else 0.0)
            if weight <= 0.0:
                continue
            term = str(row[2])
            frequency = int(row[3])
            length = max(int(row[7]), 1)
            average_length = average_lengths.get(field_name, 1.0)
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (corpus_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1_value * (
                1.0 - b_value + b_value * length / average_length
            )
            scores[current_doc_id] += (
                weight * idf * frequency * (k1_value + 1.0) / denominator
            )
            field_id = str(row[1])
            match = details[current_doc_id].setdefault(
                field_id,
                {
                    "field_type": field_name,
                    "field_position": int(row[6]),
                    "page_number": int(row[8]) if row[8] is not None else None,
                    "section": str(row[9]) if row[9] is not None else None,
                    "term_frequencies": {},
                    "positions": {},
                    "metadata": field_metadata,
                },
            )
            match["term_frequencies"][term] = frequency
            match["positions"][term] = tuple(int(value) for value in positions_raw)

        for current_doc_id in invalid_docs:
            scores.pop(current_doc_id, None)
        maximum = max(scores.values(), default=0.0)
        ordered = sorted(
            scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:result_limit]
        hits: list[SparseSearchHit] = []
        for current_doc_id, raw_score in ordered:
            generation, fingerprint, metadata = documents[current_doc_id]
            match_values = sorted(
                details[current_doc_id].items(),
                key=lambda item: (item[1]["field_position"], item[0]),
            )[:_MAX_MATCHES_PER_HIT]
            matches = tuple(
                SparseMatch(field_id=field_id, **value) for field_id, value in match_values
            )
            hits.append(
                SparseSearchHit(
                    doc_id=current_doc_id,
                    score=raw_score / maximum if maximum > 0.0 else 0.0,
                    generation=generation,
                    profile_fingerprint=fingerprint,
                    metadata=metadata,
                    matches=matches,
                )
            )
        return hits

