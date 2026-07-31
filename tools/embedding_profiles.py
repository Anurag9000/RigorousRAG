"""Built-in embedding contracts. Values remain data, never executable model code."""

from tools.embedding_models import EmbeddingProfile


def _profile(alias: str, **values) -> EmbeddingProfile:
    return EmbeddingProfile(alias=alias, **values)


BUILTIN_PROFILES: dict[str, EmbeddingProfile] = {
    "minilm-l6-v2": _profile(
        "minilm-l6-v2",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        max_sequence_tokens=256,
        normalize_embeddings=True,
        language="English",
        domain="general",
        modes=("dense",),
        license="Apache-2.0",
        source_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        notes="Compatibility profile for the repository's historical default.",
    ),
    "e5-base-v2": _profile(
        "e5-base-v2",
        model_name="intfloat/e5-base-v2",
        dimensions=768,
        max_sequence_tokens=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
        normalize_embeddings=True,
        language="English",
        domain="general",
        modes=("dense",),
        license="MIT",
        source_url="https://huggingface.co/intfloat/e5-base-v2",
        notes="E5 query and passage prefixes are part of the encoding contract.",
    ),
    "bge-base-en-v1.5": _profile(
        "bge-base-en-v1.5",
        model_name="BAAI/bge-base-en-v1.5",
        dimensions=768,
        max_sequence_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
        normalize_embeddings=True,
        language="English",
        domain="general",
        modes=("dense",),
        license="MIT",
        source_url="https://huggingface.co/BAAI/bge-base-en-v1.5",
        notes="The query instruction is configurable and remains fingerprinted.",
    ),
    "gte-base": _profile(
        "gte-base",
        model_name="thenlper/gte-base",
        dimensions=768,
        max_sequence_tokens=512,
        normalize_embeddings=True,
        language="English",
        domain="general",
        modes=("dense",),
        license="MIT",
        source_url="https://huggingface.co/thenlper/gte-base",
    ),
    "instructor-base": _profile(
        "instructor-base",
        model_name="hkunlp/instructor-base",
        dimensions=768,
        max_sequence_tokens=512,
        normalize_embeddings=True,
        language="English",
        domain="instruction-conditioned",
        modes=("dense",),
        license="Apache-2.0",
        source_url="https://huggingface.co/hkunlp/instructor-base",
        notes="Task instructions are supplied by an adapter-aware encoder.",
        requires_adapter=True,
    ),
    "specter2": _profile(
        "specter2",
        model_name="allenai/specter2_base",
        dimensions=768,
        max_sequence_tokens=512,
        normalize_embeddings=True,
        language="English",
        domain="scientific literature",
        modes=("dense",),
        license="Apache-2.0",
        source_url="https://huggingface.co/allenai/specter2_base",
        notes="Scientific adapters and title/abstract construction are contract data.",
        requires_adapter=True,
    ),
    "bge-m3": _profile(
        "bge-m3",
        model_name="BAAI/bge-m3",
        dimensions=1024,
        max_sequence_tokens=8192,
        normalize_embeddings=True,
        language="multilingual",
        domain="general long-document",
        modes=("dense", "sparse", "multi-vector"),
        license="MIT",
        source_url="https://huggingface.co/BAAI/bge-m3",
        notes="Multi-function representations require a model-specific adapter.",
        requires_adapter=True,
    ),
}

MODEL_ALIASES: dict[str, str] = {
    profile.model_name: alias for alias, profile in BUILTIN_PROFILES.items()
}
MODEL_ALIASES.update(
    {
        "all-MiniLM-L6-v2": "minilm-l6-v2",
        "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6-v2",
        "allenai/specter2": "specter2",
    }
)

__all__ = ["BUILTIN_PROFILES", "MODEL_ALIASES"]
