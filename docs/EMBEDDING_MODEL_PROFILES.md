# Embedding model profiles

RigorousRAG records embedding-model behavior as a declarative profile instead of treating
a model repository name as a complete indexing contract. A profile captures dimensions,
sequence limits, query/document instructions, normalization, language/domain, supported
representation modes, schema version, and whether an adapter-aware encoder is required.

## Built-in profiles

| Profile | Model | Dimensions | Max tokens | Query/document contract | Modes |
|---|---|---:|---:|---|---|
| `minilm-l6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | no prefix; normalized dense embedding | dense |
| `e5-base-v2` | `intfloat/e5-base-v2` | 768 | 512 | `query: ` / `passage: ` prefixes | dense |
| `bge-base-en-v1.5` | `BAAI/bge-base-en-v1.5` | 768 | 512 | retrieval instruction on queries; no passage prefix | dense |
| `gte-base` | `thenlper/gte-base` | 768 | 512 | no required prefix | dense |
| `instructor-base` | `hkunlp/instructor-base` | 768 | 512 | task/domain instructions; adapter-aware encoder | dense |
| `specter2` | `allenai/specter2` | 768 | 512 | scientific title/abstract or short query; adapter-aware | dense |
| `bge-m3` | `BAAI/bge-m3` | 1024 | 8192 | multilingual long-document profile | dense, sparse, multi-vector |

Primary model cards:

- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/intfloat/e5-base-v2
- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://huggingface.co/thenlper/gte-base
- https://huggingface.co/hkunlp/instructor-base
- https://huggingface.co/allenai/specter2_base
- https://huggingface.co/BAAI/bge-m3

The registry records model-card facts and encoding instructions; it does not assert that
one profile is universally best. Selection must be benchmarked on the target datasets,
languages, document lengths, query types, and resource envelope.

## Schema fingerprints

Every profile has a SHA-256 fingerprint over its complete canonical definition. A vector
or sparse generation must record this fingerprint. Changing dimensions, model,
normalization, instructions, modes, or schema version therefore creates a different index
schema and requires migration/reindexing rather than silently mixing embeddings.

Unknown operator-supplied model names remain supported through a compatibility profile,
but their dimensions and sequence limits are recorded as unknown. Such profiles cannot be
used for an automatic dimension-changing migration until the operator supplies a complete
profile.

## Operator profiles

`EMBEDDING_PROFILES_JSON` may add or replace bounded profiles. Duplicate keys,
non-standard JSON constants, unknown fields, padded identifiers, controls, unsupported
modes, fractional dimensions, and oversized configuration fail closed.

Example:

```json
{
  "lab-biomedical": {
    "model_name": "lab/biomedical-retriever",
    "dimensions": 768,
    "max_sequence_tokens": 512,
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "normalize_embeddings": true,
    "language": "English",
    "domain": "biomedical",
    "modes": ["dense"],
    "license": "internal",
    "source_url": "",
    "notes": "",
    "schema_version": 1,
    "requires_adapter": false
  }
}
```

Profile overrides are configuration, not automatically trusted executable code. Model
download, revision pinning, safe tensor formats, license review, cache policy, and
deployment resource limits remain operator responsibilities.
