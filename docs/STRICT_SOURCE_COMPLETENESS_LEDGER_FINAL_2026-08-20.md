# RigorousRAG — final strict source-completeness ledger (2026-08-20)

This ledger supersedes earlier source-status conclusions where they conflict with it. It is a
**source implementation** statement, not an empirical-results statement.

The completion bar used here is:

> Once exact local datasets/annotations and admitted local model/tokenizer artifacts exist, no new
> repository source should have to be written to govern/materialize the data, construct the
> implemented objectives/models, run staged training, resume exactly, validate, export, qualify,
> promote, or bind the trained artifact into the existing runtime authority.

Excluded from the claim are dataset/model acquisition, dependency installation, actual
materialization/model execution, training, inference, tests, benchmarks, empirical calibration,
hyperparameter tuning from results, and real credential/infrastructure operations.

## Final conclusion

Under that source-only definition, the Grounded-RAG and Dynamic-RAG continuation work is complete.
No additional major architecture, loss family, trainer/checkpoint algorithm, evaluator family,
export/promotion path, or data-authority algorithm was identified as missing in the final static
sweeps.

## Production package path

New production runs enter through the strict/current package surface:

- `rigorousrag-grounded-import`
- `rigorousrag-dynamic-publish`
- `rigorousrag-dynamic-sidecar`
- `rigorousrag-dynamic-feature-sidecar`
- `rigorousrag-dynamic-recordings`
- `rigorousrag-canonical-materialize`
- `rigorousrag-canonical-bundle`
- `rigorousrag-canonical-recipe`
- `rigorousrag-advanced-training`
- benchmark/corpus/qrels/evaluator/qualification commands
- `rigorousrag-advanced-release`
- `rigorousrag-runtime-bundle`

Historical/compatibility modules remain in source for reproducibility; they are not the production
entry path for new runs.

## Grounded source path

The final Grounded path is:

```text
exact local source bytes
 -> governed grounded import
 -> authoritative canonical Grounded v2 materialization
 -> disk-backed teacher/reference/document-utility cache authorities
 -> restart-verifiable canonical bundle
 -> canonical recipe v2 + resolved plan
 -> authoritative staged trainer
 -> exact checkpoint/resume
 -> inference artifact/export
 -> evaluator-bound qualification
 -> release-v5 / runtime bundle
```

Grounded canonical data now has:

- safe content-derived physical split filenames;
- production split ceiling of 100;
- global example-ID uniqueness through SQLite;
- evidence-set identity through SQLite;
- **global evidence-ID payload consistency across splits**: the same evidence ID may recur only
  when its text/source payload is identical;
- staged whole-directory publication;
- strict restart verification;
- disk-backed supervision-cache membership/content authority;
- exact producer/tokenizer/dataset/source/config cache identities;
- cache completeness proofs against the example universe.

The tensor/collation path includes causal and seq2seq generation, prompt/answer token alignment,
evidence masks, claim token positions, multi-evidence supporting/contradicting stances, citation
targets, support/contradiction targets, unsupported-token masks, abstention/reflection targets,
preference tensors, teacher logits, reference log probabilities and retriever-utility tensors.

Grounded objectives remain fully implemented: token NLL, citation pointer, support,
contradiction, abstention, reflection, unsupported-token unlikelihood, grounded preference/DPO,
teacher KL, retriever-coupling KL and weighted joint objectives.

## Dynamic feature authority

The source no longer stops at an abstract `DynamicFeatureProvider` protocol. Production reference
source exists for:

- next-token entropy;
- top-1 probability margin;
- evidence sufficiency;
- semantic support;
- contradiction risk;
- citation coverage;
- context novelty;
- unresolved-entity ratio;
- temporal uncertainty;
- retrieval-count fraction;
- token-budget fraction;
- elapsed-iteration fraction.

Semantic quantities are explicit contract-bound inputs rather than weak heuristic guesses.
Snapshot-keyed feature observations can be sealed from exact local JSONL into a restart-verifiable
SQLite authority. A local causal-LM uncertainty provider computes entropy/margin from actual
next-token logits.

`RuntimeBoundDynamicFeatureProvider` additionally binds structural budget fractions to the exact
active `DynamicRagRuntimePolicy`, rejecting feature vectors computed against a different retrieval,
token, or iteration budget.

## Runtime-to-training episode recording

A source gap identified during this continuation was the missing authoritative bridge from the
repo-owned `run_dynamic_rag` loop to `LegalDynamicRagEpisodeStep` records. That bridge now exists.

`run_authoritative_recorded_dynamic_rag_episode(...)` wraps the existing runtime rather than
forking it. Transparent feature/policy proxies record the exact snapshots/features/scores consumed
by the server loop.

Every raw recorded step contains:

- episode ID and deterministic step ID;
- exact training context;
- exact feature vector;
- exact legal-action set;
- exact server-selected action;
- deterministic behavior-action probability `1.0` for this argmax/tie-break runtime;
- request SHA;
- snapshot SHA;
- runtime-policy SHA;
- runtime-bound feature-provider SHA;
- policy artifact and policy contract SHA;
- behavior-policy SHA;
- context-provider SHA;
- exact finite action-score map plus its digest.

Strict restart verification reconstructs the same deterministic server action from the retained
score map and legal set. Thus a row is not accepted merely because its chosen action happens to be
legal.

Raw runtime logs are required to have **no** realized-gain target, advantage, value target,
hidden-cache key, or information-need span. Those are later supervision-materialization outputs.
Terminal utility may occur only on the final row under one exact terminal-utility-provider
identity.

## Cohort-safe context and recorded-runtime cohorts

Request identity is sealed per episode by `request_sha256`; the context-provider contract now
identifies the deterministic context construction policy rather than one request instance. This
allows a legitimate multi-query cohort to share one provider identity while still verifying the
request for every runtime snapshot.

Recorded episodes can be aggregated through the existing governed dynamic dataset publisher and
then sealed in a cohort envelope. Production cohort admission requires one coherent:

- runtime policy;
- runtime-bound feature provider;
- policy artifact;
- policy contract;
- deterministic behavior-policy contract;
- context construction policy;
- optional terminal-utility provider;
- aggregate runtime provider contract.

The cohort authority stores a sorted episode-source list and incrementally recomputes the exact
`rigorousrag-dynamic-trajectory-source-set/v1` identity used by the existing dynamic dataset
publisher. Strict cohort restart verification admits every source episode through the strict raw
runtime verifier.

If dataset publication succeeds but cohort sealing is interrupted,
`rigorousrag-dynamic-recordings seal-existing` verifies the already-published dataset and episode
authorities and emits only the missing cohort envelope. Runtime replay/republication is not
required.

## Single-receipt Dynamic canonical materialization

For repo-owned runtime recordings, the canonical-materialization schema

`rigorousrag-authoritative-recorded-dynamic-canonical-materialization-config/v1`

accepts one strict recorded-cohort receipt instead of manually copied source-shard hashes and
runtime-lineage hashes.

The bridge derives:

- exact governed source split paths;
- exact split SHA/count identities;
- source dataset manifest identity;
- source-set identity;
- runtime provider-stack identity;
- runtime-bound feature-provider identity;
- deterministic behavior-policy identity;
- source revision/reward configuration carried by the cohort lineage.

Operators still supply the actual target-supervision artifacts and local generator/tokenizer
bindings required to perform the later materialization.

## Dynamic supervision and policy training

Production dynamic supervision sidecars exist for information-need spans, realized retrieval
gains, logged values and legal-action counterfactual utilities. The production path uses
worker-local immutable SQLite connections rather than opening a database for every row, verifies
the sealed index per worker, and closes providers deterministically without masking a primary
materialization failure.

The materializer computes the source needed by the implemented curriculum: action targets,
realized/counterfactual gains, costs, rewards, returns, GAE advantages, value targets,
information-need labels and hidden-state cache keys/tensors.

The local dynamic hidden-state provider supports causal and encoder-decoder generators and handles
left/right padding correctly.

The authoritative dynamic collator preserves the logged behavior probability. The strict policy
step masks illegal actions, computes current masked-policy probability for the logged action, and
forms the capped importance ratio against the **logged** behavior probability. Therefore the
recorded value `1.0` has its precise deterministic-runtime meaning rather than being substituted by
training code.

Implemented dynamic objectives remain: imitation/action CE, information-need BCE, value Huber,
legal-action-masked off-policy policy gradient, action-cost loss, entropy regularization and joint
weighted objectives.

## Corpus-scale authority

Production paths avoid corpus-sized Python object sets/maps for advertised large corpora:

- global grounded identities: SQLite;
- dynamic episode/step uniqueness: SQLite;
- supervision-cache membership: SQLite;
- dynamic target sidecars: SQLite;
- dynamic feature observations: SQLite;
- benchmark/corpus/qrels identity authorities: disk-backed/SQLite;
- authoritative training random-access index: SQLite.

`ManifestBoundAuthoritativeJsonlDataset` performs a streaming validation pass and stores
ordinal→offset/length/row-SHA metadata on disk. Workers open the index immutable/read-only and each
selected row is byte-range SHA-checked and reparsed before use. The authoritative runner therefore
does not keep the training corpus or an O(N) Python row index in memory.

## Canonical bundle / recipe / resolved plan

The stable trainer config remains `rigorousrag-advanced-training-config/v1`.

The installed canonical recipe command is the v2 operator envelope. It can optionally set exact
per-stage:

- `max_steps`;
- `checkpoint_every_steps`;
- `learning_rate`.

Overrides are atomically written and re-parsed through the canonical config authority. Every v2
recipe also emits a resolved-plan artifact containing exact:

- stage order/kinds;
- max steps/checkpoint cadence/learning rate;
- every objective weight;
- objective and stage SHAs;
- architecture;
- dynamic budget/retrieval-stack identity where applicable;
- execution config;
- collator config;
- trainability policy;
- train/validation split identities;
- model/tokenizer bindings;
- supervision-cache contracts;
- checkpoint root/resume digest;
- final plan and resolved-plan digests.

Recipe admission also requires the source revision to match the canonical training-data authority;
Dynamic additionally binds the runtime-lineage revision and hidden-cache revision.

## Checkpoint, evaluation, export, promotion and runtime

These were already present and remain the shared source authority rather than being duplicated:

- deterministic staged PyTorch engine;
- optimizer/scheduler/AMP/DDP/accumulation/clipping;
- stage cursor and sampler/collator checkpointing;
- Python/Torch CPU/CUDA/NumPy RNG checkpoint state;
- model/optimizer/scheduler/scaler state;
- source/config/dataset/model architecture identities;
- content-addressed best/latest semantics;
- Grounded/Dynamic validation evaluators;
- inference-only artifact export;
- model card/training lineage;
- artifact admission/attestation;
- evaluator-bound promotion evidence;
- threshold direction semantics;
- exact arithmetic-mean production evaluator semantics;
- release-v5 qualification;
- runtime-stack promotion/rollback;
- strict runtime-bundle verification.

## Advanced public benchmark proposals

BRIGHT, RAGTruth, ASQA/ELI5 attribution, KILT, FreshQA, QASC/StrategyQA and related entries remain
**proposal-only** until exact local bytes/version/license evidence exist. This is deliberate.

After download, the governed benchmark importer already provides a declarative nested field-path
mapping profile with answer normalization, metadata extraction, generated IDs, exact source SHA,
version/license/card governance and promotability checks. Consequently no bespoke Python source is
required merely to adapt those exact local versions; hard-coding guessed/mutable upstream schemas
would be weaker than the existing governed profile.

## Final static closure

Final searches in the audited repository surfaces found no live `NotImplementedError`, TODO, FIXME
or placeholder marker. The compatibility runtime-recording primitive is not referenced by the
production recording path; production uses the atomic strict recorder.

No compile/import/test/model run is claimed by this ledger. The remaining work is empirical:
provide real local artifacts, execute the already-implemented paths, evaluate results and tune or
promote based on measured evidence.

## Final production flows

Grounded:

```text
local source bytes
 -> rigorousrag-grounded-import
 -> rigorousrag-canonical-materialize
 -> rigorousrag-canonical-bundle
 -> rigorousrag-canonical-recipe (v2 + resolved plan)
 -> rigorousrag-advanced-training
 -> checkpoint/export/evaluation
 -> rigorousrag-advanced-release
 -> rigorousrag-runtime-bundle
```

Dynamic from repo-owned runtime logs:

```text
run_authoritative_recorded_dynamic_rag_episode
 -> strict episode authorities
 -> rigorousrag-dynamic-recordings publish
    OR seal-existing for recovery
 -> strict cohort authority
 -> rigorousrag-dynamic-sidecar / dynamic-feature-sidecar as needed
 -> rigorousrag-canonical-materialize (recorded-cohort schema)
 -> rigorousrag-canonical-bundle
 -> rigorousrag-canonical-recipe (v2 + resolved plan)
 -> rigorousrag-advanced-training
 -> checkpoint/export/evaluation
 -> rigorousrag-advanced-release
 -> rigorousrag-runtime-bundle
```
