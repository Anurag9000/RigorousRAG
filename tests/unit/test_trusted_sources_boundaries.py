import itertools

import pytest

import trusted_sources
from trusted_sources import SourceCategory, derive_domain_suffixes


def test_complete_catalogue_exports_unique_immutable_https_seeds():
    assert isinstance(trusted_sources.CATEGORIES, tuple)
    assert isinstance(trusted_sources.ALL_TRUSTED_SEEDS, tuple)
    assert isinstance(trusted_sources.ALL_TRUSTED_DOMAINS, frozenset)
    assert len(trusted_sources.ALL_TRUSTED_SEEDS) == len(
        set(trusted_sources.ALL_TRUSTED_SEEDS)
    )
    assert all(seed.startswith("https://") for seed in trusted_sources.ALL_TRUSTED_SEEDS)
    assert all(seed.endswith("/") or "/" in seed[8:] for seed in trusted_sources.ALL_TRUSTED_SEEDS)
    assert "arxiv.org" in trusted_sources.ALL_TRUSTED_DOMAINS
    assert "www.nature.com" in trusted_sources.ALL_TRUSTED_DOMAINS
    assert "nature.com" in trusted_sources.ALL_TRUSTED_DOMAINS


def test_category_map_does_not_expose_mutable_seed_lists():
    categories = trusted_sources.category_map()
    first_name = trusted_sources.CATEGORIES[0].name

    assert isinstance(categories[first_name], tuple)
    with pytest.raises(TypeError):
        categories[first_name] = ()
    with pytest.raises(AttributeError):
        categories[first_name].append("https://attacker.test/")


def test_source_category_validates_name_description_and_seed_tuple():
    with pytest.raises(ValueError, match="names"):
        SourceCategory(name="", description="description", seeds=())
    with pytest.raises(ValueError, match="descriptions"):
        SourceCategory(name="Name", description="", seeds=())
    with pytest.raises(ValueError, match="descriptions"):
        SourceCategory(name="Name", description="x" * 2001, seeds=())
    with pytest.raises(ValueError, match="immutable tuple"):
        SourceCategory(
            name="Name",
            description="description",
            seeds=["https://example.test"],
        )
    with pytest.raises(ValueError, match="unique"):
        SourceCategory(
            name="Name",
            description="description",
            seeds=("https://example.test", "https://example.test/"),
        )


def test_seed_validation_rejects_credentials_controls_non_https_and_bad_hosts():
    invalid = (
        "http://example.test",
        "ftp://example.test",
        "https://alice:password@example.test",
        "https://example.test/path\r\nInjected: yes",
        "https:///missing-host",
        "https://not a hostname.test/",
        "https://example..test/",
        "https://-bad.example.test/",
        "https://127.0.0.1/",
        "https://[2606:4700:4700::1111]/",
        "https://example.test:8443/",
        "https://example.test/path#fragment",
        object(),
    )
    for seed in invalid:
        with pytest.raises(ValueError):
            SourceCategory(
                name="Name",
                description="description",
                seeds=(seed,),
            )


def test_default_https_port_is_normalized_away():
    category = SourceCategory(
        name="Name",
        description="description",
        seeds=("https://Sub.Example.test:443/path",),
    )

    assert category.seeds == ("https://sub.example.test/path",)


def test_domain_derivation_uses_hostnames_not_netloc_credentials_or_ports():
    domains = derive_domain_suffixes(
        [
            "https://www.example.test/path",
            "https://sub.example.test:443/path",
        ]
    )

    assert domains == frozenset(
        {"www.example.test", "example.test", "sub.example.test"}
    )
    assert all(":" not in domain for domain in domains)
    assert all("@" not in domain for domain in domains)


def test_domain_derivation_rejects_text_and_infinite_iterables(monkeypatch):
    with pytest.raises(ValueError, match="iterable"):
        derive_domain_suffixes("https://example.test")

    monkeypatch.setattr(trusted_sources, "_MAX_TOTAL_SEEDS", 3)
    with pytest.raises(ValueError, match="At most 3"):
        derive_domain_suffixes(
            f"https://{index}.example.test" for index in itertools.count()
        )
