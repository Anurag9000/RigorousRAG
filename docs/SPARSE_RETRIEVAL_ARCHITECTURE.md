# Persistent sparse retrieval architecture

`tools.sparse_index.SparseIndex` is the owner-scoped fielded lexical store used by the
second capability wave. It is intentionally dependency-free and uses SQLite transactions
rather than assuming that every deployment has FTS5, Elasticsearch, OpenSearch, Vespa, or
a learned sparse serving stack.

## Data model

Each sparse document is keyed by `(owner_id, doc_id)` and records:

- generation number;
- embedding/profile fingerprint used by the paired index generation;
- bounded public-safe metadata;
- field count and total token count;
- schema version and update time.

Fields are deterministic records with:

- field ID and ordered position;
- field type (`title`, `abstract`, `heading`, `body`, `caption`, `table`,
  `reference`, or an operator extension);
- text and token count;
- optional page and section provenance;
- bounded metadata.

A postings table stores one frequency per `(owner, document, field, term)`. Search computes
BM25F-style scores at query time with transparent field weights. It never crosses owner
scope and supports an exact document filter.

## Transaction and recovery contract

- A replacement is one `BEGIN IMMEDIATE` SQLite transaction.
- Existing document, fields, and postings are removed through foreign-key cascades only
  inside that transaction.
- Any insertion failure rolls the entire replacement back.
- `snapshot_document` captures the complete reconstructable sparse generation.
- `restore_document` exactly restores the previous generation or removes a newly-created
  generation when the prior snapshot was absent.
- `delete_document` is owner/document scoped.
- Generation numbers increase on replacement and are restored from snapshots during
  compensation.

The vector+sparse coordinator is a separate implementation step. Until that coordinator is
wired, the store is not represented as part of the authoritative ingestion commit.

## Filesystem and corruption boundary

The database and every existing path ancestor reject POSIX symbolic links and Windows
reparse points. The parent and database device/inode identities are captured and checked
before every operation. Parent replacement, database replacement, disappearance, or
redirection fails closed.

Stored metadata uses strict finite JSON. Corrupt document metadata prevents snapshot
publication; corrupt result metadata causes that hit to be skipped. Query text, owners,
document/field identifiers, field counts, characters, tokens, postings, results, and custom
field weights are bounded.

## Current scale envelope

This implementation is appropriate for the existing single-host architecture and for
transparent retrieval experiments. It is not a distributed inverted index. Large-scale
deployments may replace it with OpenSearch, Vespa, Tantivy, Lucene, or a learned sparse
service, but must preserve:

- owner isolation;
- stable document/field provenance;
- generation fingerprints;
- replacement snapshots and compensation;
- bounded queries and result materialization;
- transparent scoring/features;
- exact deletion and rebuild verification.
