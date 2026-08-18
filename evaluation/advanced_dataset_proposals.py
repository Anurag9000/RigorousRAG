"""Planning-only benchmark proposals for newer RigorousRAG research surfaces.

The canonical :mod:`evaluation.dataset_governance` module deliberately distinguishes a
named benchmark *proposal* from a promotable immutable dataset manifest.  This file extends
that planning catalog for reasoning-intensive retrieval, grounding/hallucination analysis,
long-form attribution, temporal knowledge and additional multi-hop reasoning.

Nothing in this module downloads a dataset, asserts an exact version/checksum, or makes a
license decision.  Before any proposal is used for a governed experiment it must be turned
into a ``DatasetManifest`` with exact bytes, split identities, reviewed licensing and
leakage evidence.
"""

from __future__ import annotations

from evaluation.dataset_governance import DatasetModality, DatasetProposal, DatasetTask


ADVANCED_RAG_BENCHMARK_PROPOSALS: tuple[DatasetProposal, ...] = (
    DatasetProposal(
        "BRIGHT",
        (DatasetTask.RETRIEVAL, DatasetTask.RERANKING),
        (DatasetModality.TEXT,),
        "Reasoning-intensive retrieval target for measuring whether query reasoning, decomposition, learned planning and reranking improve retrieval beyond surface semantic matching.",
        "reasoning-intensive-retrieval",
    ),
    DatasetProposal(
        "RAGTruth",
        (DatasetTask.FACT_VERIFICATION, DatasetTask.CITATION, DatasetTask.QUESTION_ANSWERING),
        (DatasetModality.TEXT,),
        "Grounded-generation and hallucination-analysis target for claim-level support, contradiction, calibration, abstention and hallucination-detector training/evaluation.",
        "grounded-generation",
    ),
    DatasetProposal(
        "ASQA / ELI5 attribution suites",
        (DatasetTask.QUESTION_ANSWERING, DatasetTask.CITATION),
        (DatasetModality.TEXT,),
        "Long-form answer attribution target for citation correctness, completeness, excess-citation control, answer support and generate-then-refine experiments.",
        "long-form-attribution",
    ),
    DatasetProposal(
        "KILT task family",
        (DatasetTask.RETRIEVAL, DatasetTask.QUESTION_ANSWERING, DatasetTask.FACT_VERIFICATION, DatasetTask.CITATION),
        (DatasetModality.TEXT,),
        "Knowledge-intensive task family for end-to-end retrieval plus provenance-aware answer evaluation; pin each constituent task and corpus snapshot separately.",
        "knowledge-intensive-provenance",
    ),
    DatasetProposal(
        "FreshQA temporal QA",
        (DatasetTask.QUESTION_ANSWERING, DatasetTask.RETRIEVAL, DatasetTask.FACT_VERIFICATION),
        (DatasetModality.TEXT,),
        "Temporal/freshness target for dated evidence, stale-source abstention, temporal normalization and retrieval-stack freshness policies.",
        "temporal-freshness",
    ),
    DatasetProposal(
        "QASC / StrategyQA reasoning sets",
        (DatasetTask.MULTI_HOP, DatasetTask.QUESTION_ANSWERING, DatasetTask.RETRIEVAL),
        (DatasetModality.TEXT,),
        "Additional reasoning and multi-fact composition target for decomposition, dynamic retrieval and support-path analysis; task semantics must be audited before promotion.",
        "reasoning-multihop",
    ),
    DatasetProposal(
        "Repository-owned matched clean/poisoned RAG corpus",
        (DatasetTask.ADVERSARIAL_SECURITY, DatasetTask.RETRIEVAL, DatasetTask.CITATION, DatasetTask.FACT_VERIFICATION),
        (DatasetModality.TEXT, DatasetModality.PDF, DatasetModality.MULTIMODAL),
        "Digest-paired clean/attacked cases for retrieval compromise, citation compromise, answer attack success, duplicate/source concentration and clean-utility-retention reporting.",
        "rag-poisoning-robustness",
    ),
    DatasetProposal(
        "Repository-owned dynamic-RAG episode corpus",
        (DatasetTask.RETRIEVAL, DatasetTask.QUESTION_ANSWERING),
        (DatasetModality.TEXT,),
        "Logged generation-state/action/retrieval-outcome episodes for retrieve-vs-continue imitation, information-need span supervision, counterfactual value learning and off-policy evaluation.",
        "dynamic-retrieval-policy",
    ),
    DatasetProposal(
        "Repository-owned grounded preference corpus",
        (DatasetTask.QUESTION_ANSWERING, DatasetTask.CITATION, DatasetTask.FACT_VERIFICATION),
        (DatasetModality.TEXT,),
        "Chosen/rejected grounded response pairs bound to immutable evidence sets for citation attribution, abstention, unsupported-content unlikelihood and grounded preference optimization.",
        "grounded-generator-training",
    ),
)


__all__ = ["ADVANCED_RAG_BENCHMARK_PROPOSALS"]
