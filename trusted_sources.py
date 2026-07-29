"""Curated and validated academic, governmental, and educational crawl seeds."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Tuple
from urllib.parse import urlparse, urlunparse

_MAX_CATEGORIES = 100
_MAX_SEEDS_PER_CATEGORY = 1000
_MAX_TOTAL_SEEDS = 10_000
_MAX_URL_CHARS = 4096


def _validated_seed(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Trusted source seeds must be strings.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > _MAX_URL_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("Trusted source seeds must contain valid bounded URLs.")
    try:
        parsed = urlparse(rendered)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Trusted source seeds must contain valid URLs.") from exc
    if parsed.scheme.lower() != "https":
        raise ValueError("Trusted source seeds must use HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Trusted source seeds may not contain credentials.")
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname or any(character.isspace() for character in hostname):
        raise ValueError("Trusted source seeds must contain valid hostnames.")
    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("Trusted source seeds must contain valid hostnames.") from exc
    rendered_host = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    netloc = rendered_host if port is None else f"{rendered_host}:{port}"
    return urlunparse(("https", netloc, parsed.path or "/", "", parsed.query, ""))


@dataclass(frozen=True)
class SourceCategory:
    name: str
    description: str
    seeds: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 200:
            raise ValueError("Source-category names must contain 1-200 characters.")
        if not isinstance(self.description, str) or len(self.description) > 2000:
            raise ValueError(
                "Source-category descriptions must contain at most 2,000 characters."
            )
        if not isinstance(self.seeds, tuple):
            raise ValueError("Source-category seeds must be an immutable tuple.")
        if len(self.seeds) > _MAX_SEEDS_PER_CATEGORY:
            raise ValueError(
                f"Source categories may contain at most {_MAX_SEEDS_PER_CATEGORY} seeds."
            )
        normalized = tuple(_validated_seed(seed) for seed in self.seeds)
        if len(normalized) != len(set(normalized)):
            raise ValueError("Source-category seeds must be unique after normalization.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "seeds", normalized)


def _category(name: str, description: str, seeds: tuple[str, ...]) -> SourceCategory:
    return SourceCategory(name=name, description=description, seeds=seeds)


CATEGORIES: Tuple[SourceCategory, ...] = (
    _category(
        "Reference & Encyclopedias",
        "General knowledge resources vetted for editorial oversight.",
        (
            "https://www.wikipedia.org",
            "https://en.wikipedia.org",
            "https://www.britannica.com",
            "https://www.newworldencyclopedia.org",
            "https://www.worldhistory.org",
            "https://www.metmuseum.org/toah",
            "https://www.poetryfoundation.org",
            "https://plato.stanford.edu",
            "https://iep.utm.edu",
            "https://www.loc.gov",
        ),
    ),
    _category(
        "Academic Journals & Publishers",
        "Peer-reviewed publishers and aggregators.",
        (
            "https://www.nature.com",
            "https://www.sciencedirect.com",
            "https://link.springer.com",
            "https://academic.oup.com",
            "https://journals.sagepub.com",
            "https://www.tandfonline.com",
            "https://www.jstor.org",
            "https://www.cell.com",
            "https://www.pnas.org",
            "https://www.annualreviews.org",
            "https://www.mdpi.com",
            "https://www.frontiersin.org",
            "https://www.rsc.org/journals-books-databases",
            "https://dl.acm.org",
            "https://ieeexplore.ieee.org",
        ),
    ),
    _category(
        "Preprint Servers & Scholarly Networks",
        "Open access repositories for early research dissemination.",
        (
            "https://arxiv.org",
            "https://www.biorxiv.org",
            "https://www.medrxiv.org",
            "https://osf.io/preprints",
            "https://hal.science",
            "https://www.researchgate.net",
        ),
    ),
    _category(
        "Education & Open Courseware",
        "Structured learning materials from universities and education platforms.",
        (
            "https://ocw.mit.edu",
            "https://www.khanacademy.org",
            "https://www.edx.org",
            "https://www.coursera.org",
            "https://openstax.org",
            "https://www.open.edu/openlearn",
            "https://www.futurelearn.com",
            "https://www.saylor.org",
            "https://www.carnegielearning.com",
            "https://cs50.harvard.edu",
            "https://www.ted.com/topics/education",
        ),
    ),
    _category(
        "Medical & Health Authorities",
        "Evidence-based medical information and clinical guidance.",
        (
            "https://www.who.int",
            "https://www.cdc.gov",
            "https://www.nih.gov",
            "https://www.ncbi.nlm.nih.gov",
            "https://www.mayoclinic.org",
            "https://www.bmj.com",
            "https://www.medscape.com",
            "https://emedicine.medscape.com",
            "https://www.nhs.uk",
            "https://evidence.nhs.uk",
            "https://www.cochranelibrary.com",
            "https://pubmed.ncbi.nlm.nih.gov",
            "https://clinicaltrials.gov",
        ),
    ),
    _category(
        "Government & Official Statistics",
        "Official data portals, statistical agencies, and government research.",
        (
            "https://www.usa.gov",
            "https://data.gov",
            "https://www.whitehouse.gov",
            "https://www.congress.gov",
            "https://www.gao.gov",
            "https://www.gov.uk",
            "https://www.ons.gov.uk",
            "https://www.parliament.uk",
            "https://www.canada.ca",
            "https://www.statcan.gc.ca",
            "https://www.australia.gov.au",
            "https://www.abs.gov.au",
            "https://www.india.gov.in",
            "https://data.gov.in",
            "https://www.gov.za",
            "https://www.gov.br",
            "https://www.europa.eu",
            "https://data.europa.eu",
            "https://www.worldbank.org",
            "https://openknowledge.worldbank.org",
            "https://unstats.un.org",
            "https://www.imf.org",
            "https://www.oecd.org",
            "https://www.un.org",
        ),
    ),
    _category(
        "Science & Technology Agencies",
        "National laboratories and agencies publishing technical research.",
        (
            "https://www.nasa.gov",
            "https://science.nasa.gov",
            "https://www.jpl.nasa.gov",
            "https://www.nsf.gov",
            "https://www.nist.gov",
            "https://www.energy.gov",
            "https://www.lanl.gov",
            "https://www.sandia.gov",
            "https://www.esa.int",
            "https://www.jaxa.jp",
            "https://www.noaa.gov",
            "https://www.usgs.gov",
        ),
    ),
    _category(
        "Libraries & Archives",
        "Digital library collections and archives.",
        (
            "https://www.gutenberg.org",
            "https://www.hathitrust.org",
            "https://www.archives.gov",
            "https://www.britishmuseum.org",
            "https://digital.library.cornell.edu",
            "https://digitalcommons.unl.edu",
            "https://digital.library.ucla.edu",
            "https://www.si.edu/collections",
            "https://library.si.edu",
            "https://www.loc.gov/collections",
        ),
    ),
    _category(
        "Data Portals & Repositories",
        "Curated datasets for academic and policy research.",
        (
            "https://ourworldindata.org",
            "https://datahub.io",
            "https://catalog.data.gov",
            "https://data.unicef.org",
            "https://humanitarian.atlas",
            "https://data.worldbank.org",
            "https://data.oecd.org",
            "https://www.kaggle.com/datasets",
            "https://zenodo.org",
            "https://figshare.com",
            "https://datadryad.org",
        ),
    ),
    _category(
        "Fact-Checking & Verification",
        "Fact-checked journalism and verification resources.",
        (
            "https://www.reuters.com",
            "https://www.apnews.com/apfactcheck",
            "https://www.factcheck.org",
            "https://www.politifact.com",
            "https://www.snopes.com",
            "https://www.bbc.com/news/reality_check",
        ),
    ),
)

if len(CATEGORIES) > _MAX_CATEGORIES:
    raise ValueError(f"At most {_MAX_CATEGORIES} source categories are supported.")
if len({category.name for category in CATEGORIES}) != len(CATEGORIES):
    raise ValueError("Source-category names must be unique.")


def iter_all_seed_urls() -> Iterable[str]:
    for category in CATEGORIES:
        yield from category.seeds


ALL_TRUSTED_SEEDS: Tuple[str, ...] = tuple(
    sorted(dict.fromkeys(iter_all_seed_urls()))
)
if len(ALL_TRUSTED_SEEDS) > _MAX_TOTAL_SEEDS:
    raise ValueError(f"At most {_MAX_TOTAL_SEEDS} trusted seeds are supported.")


def derive_domain_suffixes(urls: Iterable[str]) -> frozenset[str]:
    if isinstance(urls, (str, bytes, bytearray)):
        raise ValueError("urls must be an iterable of HTTPS seed URLs.")
    try:
        candidates = list(
            itertools.islice(iter(urls), _MAX_TOTAL_SEEDS + 1)
        )
    except Exception as exc:
        raise ValueError("urls must be iterable.") from exc
    if len(candidates) > _MAX_TOTAL_SEEDS:
        raise ValueError(f"At most {_MAX_TOTAL_SEEDS} URLs may be inspected.")
    suffixes: set[str] = set()
    for raw_url in candidates:
        seed = _validated_seed(raw_url)
        hostname = (urlparse(seed).hostname or "").rstrip(".").lower()
        if not hostname:
            continue
        suffixes.add(hostname)
        if hostname.startswith("www."):
            suffixes.add(hostname[4:])
    return frozenset(suffixes)


ALL_TRUSTED_DOMAINS: frozenset[str] = derive_domain_suffixes(
    ALL_TRUSTED_SEEDS
)

_CATEGORY_MAP: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {category.name: category.seeds for category in CATEGORIES}
)


def category_map() -> Mapping[str, Tuple[str, ...]]:
    return _CATEGORY_MAP
