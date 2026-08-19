# RigorousRAG strict source implementation audit — continuation, 2026-08-19

## Status and scope

This ledger supersedes the implementation-status conclusions of older RigorousRAG source-audit documents for the source-only scope described below. Historical audit documents remain useful as chronological records but are not authoritative for current completeness.

Baseline for this continuation:

- repository: `Anurag9000/RigorousRAG`
- branch policy: direct commits to `main`; no feature branch or pull request is required for this work
- continuation baseline: `188cd218ec60466aa9bbea0a8e3f7f3d8e40bb34` (`feat: bind generator family and training config into artifacts`)
- implementation target: source required to define, materialize, train, resume, evaluate, export, qualify, attest, load and serve the newly added grounded-generation and dynamic-retrieval-policy families without requiring operators to write bespoke glue code

Explicitly excluded from a source-completeness claim:

- dataset downloading or license acceptance
- acquisition of real model/tokenizer weights
- generation of real annotation sidecars
- actual training, checkpoint production or checkpoint resume execution
- actual inference/model generation
- benchmark execution
- unit/integration/security/system tests
- repeated-seed experiments, ablations or statistical significance runs
- latency, VRAM, throughput and cost measurements
- production threshold calibration
- real signing/KMS/HSM operations
- cloud/vendor-specific deployment wiring
- disaster-recovery or multiregion exercises

Accordingly, this document makes no numerical quality, accuracy, robustness, latency or resource-use claim.

## Mission reconstructed for this continuation

The continuation was not a request for another isolated loss function. The source target was an end-to-end, content-bound implementation chain for the advanced RAG families already introduced in the prior audit:

1. governed training-record schemas;
2. deterministic local JSONL parsing and collation;
3. grounded generator supervision for language modeling, citations, support, contradiction, abstention, reflection, unsupported-content unlikelihood, preference optimization, teacher distillation and generator-supervised retriever coupling;
4. dynamic retrieve/continue/verify/abstain/stop policy training with information-need selection, value learning, off-policy policy gradients and action costs;
5. exact supervision materialization and immutable caches;
6. staged curricula and stage-local trainability policies;
7. checkpoint/resume identity binding;
8. causal and encoder-decoder generator support;
9. evaluation evidence and qualification;
10. inference-only artifact export;
11. signed supply-chain admission;
12. local-only runtime loading;
13. generation provider and dynamic-policy provider adapters;
14. learned information-need query construction;
15. dynamic runtime trajectory recording and target materialization;
16. one authoritative configuration/operator path;
17. path, manifest, receipt and cache fail-closed authority;
18. explicit preservation of multi-evidence, contested-evidence and legal-action semantics.

## Implemented sweep: data-to-tensor training foundation

The advanced training path now includes manifest-bound local JSONL data contracts and deterministic tensor construction.

Key source:

- `training/advanced_rag_data.py`
- `training/advanced_rag_authoritative_data.py`
- `training/advanced_rag_final_collation.py`
- `training/advanced_rag_multi_evidence.py`
- `training/seq2seq_grounded.py`

Implemented behavior includes:

- content-SHA-pinned local JSONL datasets;
- bounded records/text/evidence/claim/span counts;
- causal language-model prompt/answer target construction with explicit next-token label shifting;
- seq2seq prompt/decoder-target separation;
- claim character-span to token alignment;
- evidence tokenization and padded evidence-slot masks;
- preference chosen/rejected collation;
- teacher/reference/retriever supervision attachment;
- dynamic feature/action/value/hidden-state batching;
- resumable deterministic sampler/collator state through the generic trainer;
- fast-tokenizer/right-padding/pad-token contract before authoritative training.

## Implemented sweep: richer grounded evidence semantics

A claim no longer collapses evidence to one citation or one binary stance.

`StancedGroundedClaimAnnotation` preserves disjoint:

- `supporting_evidence_ids`
- `contradicting_evidence_ids`

One claim can therefore be simultaneously supported by some evidence and contradicted by other evidence. Legacy `evidence_ids` remains the union for compatibility with lower-level primitives.

The authoritative citation collation emits a full `[B,C,E]` citation target mask. The citation objective uses multi-positive cross entropy against all annotated supporting items rather than silently selecting the first evidence id.

Claim-level support and contradiction heads remain separate, so contested evidence can supervise both dimensions.

Claimless abstention examples are valid: claim-level objectives contribute differentiable zero while abstention/reflection/token objectives remain active.

## Implemented sweep: grounded objectives and model composition

Key source:

- `training/grounded_generation.py`
- `training/advanced_rag_models.py`
- `training/advanced_rag_steps.py`
- `training/advanced_rag_final_objectives.py`
- `training/grounded_supervision_pipeline.py`

Implemented grounded objectives:

- masked token NLL;
- multi-positive citation attribution;
- support BCE;
- contradiction BCE;
- answer-level abstention BCE;
- closed reflection-action classification;
- target-token unsupported-content unlikelihood;
- DPO-style grounded preference loss;
- target-masked teacher-token KL distillation;
- LM-supervised retriever distribution KL.

The final objective path is empty-supervision safe for valid cases such as an all-abstention batch: token, citation and claim losses contribute differentiable zero when no corresponding supervision is present rather than producing NaN or raising for a semantically valid empty mask.

`GroundedGeneratorTrainingModule` composes the admitted base language model, auxiliary grounding heads and optional retriever. `Seq2SeqGroundedGeneratorTrainingModule` provides a real encoder-decoder path rather than merely claiming seq2seq support while using causal collation.

## Implemented sweep: cached grounded supervision

Source includes concrete local cache/materialization support for:

- teacher token logits;
- reference-policy chosen/rejected log probabilities;
- per-document generator utility for retriever coupling;
- generator hidden states used by dynamic information-need selection.

`CachedDocumentUtilityRetrieverBatchBuilder` now binds:

- exact tokenizer SHA;
- exact utility-cache content contract;
- retriever-coupling configuration SHA.

The run-binding reconstruction uses the identical v2 digest formula, preventing a runner/verifier schema mismatch.

## Implemented sweep: exact supervision-cache authority

`training/advanced_rag_strict_cache.py` is the authoritative cache implementation.

It provides:

- root path authority and symlink rejection;
- strict JSON entry manifests with closed fields;
- file byte limits;
- streaming tensor SHA-256 verification;
- tensor-name verification;
- orphan/missing/unexpected entry detection;
- deterministic complete-cache `content_sha256`;
- `contract_sha256` over nominal cache identity plus exact content;
- sealing after the content contract is consumed;
- rejection of writes after sealing;
- per-read verification that the referenced entry still equals its sealed key/SHA/names/byte record.

Teacher, reference, retriever-utility and dynamic hidden-state cache contracts are included in authoritative training input identity and reconstructed during checkpoint verification. Changing cache tensor bytes under the same producer/tokenizer/dataset/config identity therefore changes the run/checkpoint identity.

## Implemented sweep: cache coverage preflight

`training/advanced_rag_cache_preflight.py` proves record-to-cache coverage before model/optimizer construction.

For active grounded objectives it checks the actual referenced:

- teacher keys;
- reference-policy fallback keys;
- document-utility keys.

For active dynamic need-selection learning it checks every hidden-state cache key.

This converts late batch/collator missing-key failures into deterministic preflight failures after the cache has already been content-bound and sealed.

## Implemented sweep: canonical curricula

`training/advanced_rag_curricula.py` encodes repository-owned methodology rather than leaving stage sequencing to prose.

Grounded curriculum covers the ordered families:

1. supervised language modeling;
2. citation attribution;
3. semantic support/contradiction grounding;
4. reflection and abstention;
5. optional generator-retriever coupling;
6. optional preference/distillation;
7. joint grounded optimization.

Dynamic curriculum covers:

1. action imitation;
2. information-need token selection;
3. state-value learning;
4. off-policy policy-gradient refinement;
5. cost-aware action control;
6. joint policy learning.

Stage definitions carry learning rates, optimizer-step limits and checkpoint cadence. Stage-local parameter trainability policies explicitly define frozen/unfrozen parameter groups while the generic engine preserves optimizer state consistently across stages.

## Implemented sweep: dynamic policy architecture and legality

Key source:

- `training/dynamic_retrieval_policy.py`
- `training/advanced_rag_models.py`
- `training/advanced_rag_action_legality.py`
- `training/advanced_rag_authoritative_data.py`

Dynamic features include uncertainty, evidence sufficiency/support/contradiction/citation coverage, context novelty, unresolved entities, temporal uncertainty and hard budget fractions.

`LegalDynamicRagEpisodeStep` preserves the exact `valid_actions` set for a logged state. The authoritative collator emits `[B,A]` action legality masks. Invalid actions are masked before:

- imitation loss;
- selected-action log probability;
- off-policy importance weighting;
- entropy computation;
- expected retrieval/verification/abstention action costs.

The logged action must itself be legal and present in the configured architecture.

At runtime, a trained architecture may omit an optional action such as `VERIFY`; the local serving adapter still returns the runtime's complete closed action map, assigning a finite unavailable sentinel to an action absent from the learned head so the server-owned closed-vocabulary contract remains stable.

## Implemented sweep: corrected off-policy learning

The strict dynamic step computes the importance ratio as:

`current_probability(logged_action) / logged_behavior_probability`

with clipping and detached ratio weighting. It no longer treats a logged-policy correction as an all-one placeholder.

Behavior probabilities, advantages and legal action identities are required when the off-policy curriculum is active.

## Implemented sweep: explicit state-value semantics

Immediate realized retrieval gain and state-value/return targets are distinct fields.

`LegalDynamicRagEpisodeStep` contains:

- `realized_retrieval_gain` — measured post-action utility delta used by reward shaping;
- `value_target` — GAE return used to train the controller's value head;
- `advantage` — GAE advantage used by policy-gradient learning.

The authoritative value stage requires `value_target`; it does not silently train the value head on the legacy immediate retrieval-gain field.

## Implemented sweep: runtime trajectory recording

`orchestration/dynamic_trajectory_recording.py` provides request-scoped wrappers around the existing dynamic feature and policy providers.

The bounded runtime itself continues to retain only trace hashes. The recorder observes the same already-released snapshot/features/scores passed to those providers and produces training-ready legal decision records externally.

Important authority details:

- runtime `request_sha256` is treated as an opaque caller-supplied request/envelope identity;
- released request text has a separate content SHA;
- text mutation is rejected;
- deterministic server argmax has behavior probability `1.0`;
- the recorder binds the exact `verification_enabled` runtime policy and recomputes legal actions with that value;
- recorder metadata binds snapshot, recorder, request and released-text identities.

## Implemented sweep: dynamic hidden-state and need-span preparation

`training/dynamic_trajectory_preparation.py` closes the manual-edit seam between recorded decisions and need-selection training.

It:

- executes a content-bound hidden-state provider supplied by the operator;
- writes exact hidden states into the authoritative safetensor cache;
- attaches deterministic `hidden_state_cache_key` values;
- accepts a strict information-need annotation provider;
- supports explicit empty-span negative labels;
- attaches annotation/provider/cache provenance;
- emits a self-verifying preparation receipt.

`SidecarInformationNeedAnnotationProvider` consumes a strict, content-hashed sidecar keyed by `episode_id:step_id`.

## Implemented sweep: generator hidden-state adapter

`GeneratorHiddenStateAdapter` handles both generator families:

- causal: final causal hidden states;
- seq2seq: encoder `last_hidden_state` over visible context.

The state summary selects the actual last visible mask position, not `sum(mask)-1`, so the reusable adapter remains correct under either left or right padding even though authoritative training requires right padding.

## Implemented sweep: learned information-need query construction

`orchestration/learned_information_need_query.py` provides a local learned query provider for the bounded dynamic runtime.

It binds:

- admitted dynamic-policy artifact;
- exact training-input identity;
- upstream generator artifact;
- tokenizer artifact;
- generator family;
- hidden-state adapter contract;
- opaque runtime request identity;
- released request-text SHA;
- query-selection configuration.

It scores context tokens through the trained information-need selector, applies bounded threshold/top-k fallback selection, preserves token order, decodes contiguous selected regions and emits bounded query text. The runtime still owns query release, retrieval and evidence admission.

## Implemented sweep: governed realized retrieval gain

Runtime recording occurs before the downstream retrieval outcome, so a recorded decision cannot truthfully know its realized gain at decision time.

`training/dynamic_reward_supervision.py` therefore provides a governed post-hoc measurement path:

- strict sidecar schema;
- content SHA;
- metric-contract SHA;
- one measured gain per episode/step;
- self-verifying application receipt;
- provider provenance added to updated records.

If the configured reward gives positive weight to realized retrieval gain, the end-to-end trajectory pipeline rejects unproven recorder placeholder zeros unless a governed gain provider is supplied or the records already carry valid gain-provider provenance.

## Implemented sweep: GAE and counterfactual target materialization

`training/dynamic_trajectory_materialization.py` groups logged steps by episode, obtains content-bound baseline values, computes reward shaping and GAE, and writes governed JSONL.

For authoritative records it materializes:

- explicit `advantage`;
- explicit `value_target` GAE return;
- legal action set;
- behavior probability;
- optional counterfactual metadata;
- source identity metadata.

Counterfactual utilities are filtered to legal actions. The provider must score the actual logged action. Improvement is measured against the actual logged legal action after action costs, not an assumed `CONTINUE` baseline; this remains well-defined when `CONTINUE` is illegal.

The materialization receipt recomputes its own digest.

## Implemented sweep: one governed trajectory pipeline

`training/dynamic_trajectory_pipeline.py` composes:

1. runtime-recorded legal decisions;
2. hidden-state and information-need preparation;
3. optional measured retrieval-gain binding;
4. GAE/value/counterfactual materialization;
5. final JSONL output;
6. one top-level lineage receipt.

This removes manual record editing from the intended methodology.

## Implemented sweep: curriculum-aware preflight

`training/advanced_rag_curriculum_preflight.py` rejects missing supervision before optimization.

Dynamic checks include:

- legal logged action;
- architecture action membership;
- hidden-state key for need-selection stages;
- explicit annotation provenance for an empty need-span negative;
- explicit `value_target` for value stages;
- advantage and behavior probability for off-policy stages.

Grounded checks include:

- claims on non-abstaining rows when claim objectives are active;
- preference pairs when preference stages are active;
- teacher cache keys for teacher distillation;
- valid stanced evidence identities.

The authoritative runners invoke this preflight directly, so direct Python use receives the same semantic checks as CLI use.

## Implemented sweep: turnkey trainers and validation

Key source:

- `training/advanced_rag_runner.py`
- `training/advanced_rag_authoritative_runner.py`
- `training/advanced_rag_evaluators.py`
- `training/torch_engine.py`

The final authoritative runners bind:

- exact train/validation bytes;
- immutable plan SHA;
- tokenizer identity;
- execution config;
- causal/seq2seq path;
- trainability policy;
- exact supervision-cache contents;
- exact retriever-supervision contract.

They construct deterministic datasets/samplers/collators/steps, perform curriculum and cache coverage preflight, then reuse the existing generic trainer for device placement, AMP/DDP, AdamW, schedulers, accumulation, clipping, evaluation, early stopping and checkpoint/resume.

Validation is run by the generic engine under `model.eval()` and `torch.no_grad()`.

The earlier cross-stage early-stopping comparison issue was corrected by stage-isolated validation score namespaces so incomparable curriculum stages do not inherit one another's best metric baseline.

## Implemented sweep: checkpoint authority

`training/advanced_checkpoint_authority.py` restricts checkpoint roots through the same filesystem authority used by configs and artifacts.

Checkpoint identity includes the generic trainer configuration, and the advanced run binding reconstructs the exact authoritative training input identity before export or evaluation.

`training/advanced_rag_run_binding.py` verifies a checkpoint against:

- plan SHA;
- exact training input SHA;
- generic training config SHA;
- bound run id;
- source commit;
- dataset manifest;
- model architecture;
- generator family;
- tokenizer identity;
- grounded retriever adapter identity where applicable.

Because exact cache content contracts are reconstructed, modifying a supervision tensor invalidates later checkpoint verification even if nominal cache producer metadata is unchanged.

## Implemented sweep: local artifact loading

`training/local_artifact_loading.py` provides fail-closed local artifact loading:

- complete local directory-tree digest verification;
- root/intermediate/descendant symlink rejection;
- path-escape checks;
- size/file-count bounds;
- `local_files_only=True`;
- `trust_remote_code=False`;
- safetensors-only model loading;
- exact model/tokenizer plan binding.

No network fallback is part of the authoritative path.

## Implemented sweep: artifact export and manifest integrity

`training/advanced_rag_artifacts.py` exports inference-only safetensor artifacts from verified content-addressed checkpoints.

Artifacts bind:

- checkpoint digest;
- plan SHA;
- exact training-input SHA;
- exact generic training-config SHA;
- source commit;
- dataset manifest;
- architecture SHA;
- base model/generator SHA;
- tokenizer SHA where applicable;
- retriever stack/config where applicable;
- dynamic budget SHA where applicable;
- generator family;
- included inference weight prefixes;
- weights SHA/byte size;
- optional evaluation receipt.

`training/advanced_rag_manifest_integrity.py` recomputes the artifact SHA from an in-memory manifest payload. Promotion/attestation cannot trust an arbitrary caller-constructed manifest digest.

## Implemented sweep: evaluation evidence

`evaluation/advanced_rag_receipts.py` turns already-executed governed benchmark runs into content-addressed evidence.

A run binds:

- benchmark id/manifest;
- evaluator contract;
- seed/repeat index;
- sample count;
- metrics;
- result artifact SHA;
- optional slice-metrics SHA.

The aggregate receipt binds the verified checkpoint/training identities and deterministic mean/median aggregation. Receipt IO uses strict JSON, byte bounds, path authority and self-verification.

No benchmark is executed by this source merely by constructing a receipt.

## Implemented sweep: promotion evidence

`training/advanced_rag_promotion_evidence.py` wraps empirical qualification in self-verifying evidence.

It verifies the artifact manifest first and binds:

- artifact SHA;
- qualification-policy SHA;
- evaluation receipt SHA;
- metrics SHA;
- promoted/blocked status;
- failure reason codes;
- nested primitive promotion receipt SHA.

The nested primitive receipt digest is independently recomputed before acceptance.

## Implemented sweep: signed supply-chain admission

`training/advanced_rag_attestation.py` bridges empirical promotion into the repository's existing artifact-attestation authority.

Admission binds and checks:

- exact promoted artifact;
- self-verifying promotion evidence;
- signed attestation statement;
- trusted verifier result;
- source revision;
- dependency lock;
- builder/key/verifier trust;
- required predicates;
- freshness;
- final content-addressed artifact directory.

`AdvancedArtifactAdmissionReceipt` recomputes its own digest on construction.

## Implemented sweep: runtime artifact reconstruction

`training/advanced_rag_runtime_loading.py` verifies the artifact manifest and local files before reconstructing:

- grounded causal or seq2seq inference module;
- optional retriever adapter with the exact trained positive-label index;
- dynamic controller and need selector.

Weights are loaded locally from safetensors with strict key/prefix checks. Runtime reconstruction is bound to the exported architecture/config identities rather than an operator guess.

## Implemented sweep: serving adapters

`orchestration/advanced_rag_model_providers.py` exposes narrow serving protocols only.

Grounded generation adapter:

- accepts released messages;
- renders a local prompt/chat template;
- enforces input/output budgets;
- calls only the admitted local generator;
- does not invent citation ids or tool authority.

Dynamic policy adapter:

- converts server-owned features to the trained feature order;
- produces raw learned logits for trained actions;
- fills optional absent actions with an explicit finite unavailable sentinel;
- never grants retrieval/tool authority.

The bounded orchestration still owns action masking, query release, retrieval, evidence admission, verification and stopping.

## Implemented sweep: grounded advisory auxiliary-head scoring

`orchestration/grounded_artifact_scoring.py` exposes post-export learned auxiliary-head scores for already-tokenized evidence/claim inputs:

- citation distribution;
- support probability;
- contradiction probability;
- abstention probability;
- reflection-action distribution.

The result is content-addressed and explicitly advisory. It does not replace the repository's server-owned output schema, citation allowlist, DLP, evidence authority or publication fence.

## Implemented sweep: citation-refined publication fence

The prior continuation already added server-owned post-generation citation refinement and `orchestration/refined_generation_publication.py`.

The publication envelope binds:

- exact authoritative generation result;
- exact answer SHA;
- exact server-owned evidence universe;
- original grounded citations;
- refinement receipt;
- runtime stack/fence state.

Publication is allowed only after authoritative checks and refinement resolve without review/abstention.

## Implemented sweep: bounded dynamic RAG runtime

The prior continuation added `orchestration/dynamic_rag_runtime.py`.

Server-owned runtime semantics include:

- closed actions: continue/retrieve/verify/abstain/stop;
- deterministic score selection under legal action masks;
- hard generation/retrieval/verification/evidence/character budgets;
- query-release boundary;
- evidence-admission boundary;
- verification boundary;
- trace digests;
- terminal stop/abstain behavior.

Learned policies score actions but do not obtain direct tool authority.

## Implemented sweep: one authoritative operator surface

`training/advanced_rag_operator.py` is the final local-only operator layer. `training/advanced_rag_cli.py` is now only a compatibility wrapper that delegates to it.

The same rich authoritative parser is used by validation and training; stanced grounded records and legal/value-target dynamic records are not accepted by one path and rejected by another.

Operator commands cover:

- validate;
- train/resume;
- verify checkpoint;
- build evaluation receipt;
- export artifact;
- verify artifact;
- load artifact;
- qualify artifact;
- verify promotion evidence;
- hash local artifact tree.

The operator uses strict config parsing, filesystem authority, local-only artifact loading, authoritative cache construction and verified checkpoint binding.

## Important integration defects found and corrected during this continuation

The continuation deliberately used adversarial integration sweeps rather than counting modules as complete. Concrete issues discovered and corrected include:

1. padded evidence slots could become invalid/fake citation candidates;
2. logged off-policy probability correction was initially represented as an all-one placeholder;
3. teacher/hidden-state caches initially assumed uniform sequence length;
4. teacher distillation initially included padding/non-target positions;
5. unsupported-content unlikelihood initially mismatched a `[B,T]` position mask with a `[B,T,V]` vocabulary-mask primitive;
6. seq2seq support was initially overstated relative to causal-style collation and was replaced with a true encoder-decoder path;
7. per-stage early stopping could compare incomparable metrics across curriculum stages;
8. runtime artifact export initially lacked complete generator-family/training identity;
9. dynamic policy serving initially omitted scores for optional actions absent from the trained architecture;
10. in-memory artifact manifests and admission receipts were not initially fully self-verifying;
11. cache manifests were initially less strict than the final path;
12. claim supervision initially collapsed multiple supporting evidence ids to one pointer;
13. claim records could not represent supporting and contradicting evidence separately;
14. dynamic training did not initially preserve or mask the legal action set from the logged runtime state;
15. trajectory materialization initially dropped legal action information;
16. the runtime result retained only hashes, leaving no governed source path for policy trajectories;
17. an initial trajectory-recorder implementation built an empty feature map before replacement; this was caught and corrected immediately;
18. dynamic value learning initially targeted a legacy immediate retrieval-gain field rather than an explicit GAE return;
19. hidden-state/need-span keys had no end-to-end preparation step;
20. realized retrieval gain had no governed post-hoc measurement path;
21. request SHA was initially assumed to equal raw request-text SHA, but the runtime contract treats it as opaque;
22. recorder legality initially assumed verification was always enabled;
23. counterfactual gain initially inherited an assumed CONTINUE baseline instead of the actual logged legal action;
24. CLI validation initially used the older record parser while training used the richer parser;
25. retriever-supervision builder and checkpoint verifier initially hashed different versions of its identity contract;
26. claimless abstention examples could be rejected by global claim preflight/all-empty masked BCE;
27. training input initially bound nominal cache identities rather than exact cache contents;
28. exact cache content could initially mutate after identity computation;
29. referenced cache-key absence could be deferred until collation rather than preflight.

These are all source-level corrections; none is presented as an empirical performance result.

## Static source verification performed

A fresh archive of current `main` was inspected after the implementation sweeps.

The following non-executing source checks were performed:

- Python bytecode compilation via `compileall`;
- AST parsing of every Python source file;
- repository-local import-target existence checks for the major internal namespaces;
- source scan for `NotImplementedError`, `TODO` and `FIXME` markers.

These checks are useful for syntax/source-shape confidence only. They are not unit tests, integration tests, model execution, training, inference or benchmark evidence.

## Current source-completeness conclusion

Within the user-defined source-only scope, this continuation did not leave a known material missing architecture/algorithm/data-collation/loss/training/checkpoint/evaluation/export/promotion/runtime-composition family after the final adversarial audit.

That conclusion means:

- required source paths are represented;
- final authoritative paths are identified and wired;
- lower-level legacy/research primitives may remain for compatibility but are not the configuration-driven authority;
- stale TODO-like conclusions from old audit documents should not be treated as current gaps;
- a new task should be added only when a concrete non-duplicate source seam is proven.

It does **not** mean the repository has been empirically validated without execution.

## What remains outside the source-only implementation scope

The remaining work is empirical, operational or deployment-specific:

1. acquire real benchmark/training datasets and record exact licenses/versions/checksums/splits;
2. materialize real grounded claims, stanced evidence, preferences and information-need annotation sidecars;
3. materialize real teacher/reference/document-utility/hidden-state caches;
4. materialize measured post-action retrieval-gain sidecars where that reward term is used;
5. install the chosen runtime/training dependencies in a target environment;
6. materialize admitted local model/tokenizer artifacts with exact tree SHAs;
7. actually train grounded generators and dynamic policies;
8. actually create and resume checkpoints;
9. actually run inference through causal and seq2seq paths;
10. execute unit/integration/security/system tests;
11. run BRIGHT/RAGTruth/attribution/multihop/temporal/poisoning and repository-owned benchmark suites;
12. run repeated seeds, ablations and statistical analyses;
13. calibrate support/contradiction/abstention/citation/reflection/action thresholds;
14. measure latency, memory, throughput and cost;
15. empirically validate dynamic retrieval gain, action costs and reward coefficients;
16. promote only artifacts that satisfy real evaluation policies;
17. perform real signature, KMS/HSM, SBOM, vulnerability and dependency-lock admission operations;
18. wire deployment-specific retrievers, feature analyzers, query-release services, evidence-admission services and verifiers;
19. run rollback, disaster-recovery and multiregion drills;
20. validate production SLOs and observability under load.

Those steps can reveal implementation defects and may lead to future source changes. Until they are executed, no runtime-correctness or quality claim should be inferred from source completeness alone.
