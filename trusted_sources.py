"""Curated list of trusted academic, governmental, and educational sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class SourceCategory:
    name: str
    description: str
    seeds: List[str]


CATEGORIES: List[SourceCategory] = [
    SourceCategory(
        name="Reference & Encyclopedias",
        description="General knowledge resources vetted for editorial oversight.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Academic Journals & Publishers",
        description="Peer-reviewed publishers and aggregators.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Preprint Servers & Scholarly Networks",
        description="Open access repositories for early research dissemination.",
        seeds=[
            "https://arxiv.org",
            "https://www.biorxiv.org",
            "https://www.medrxiv.org",
            "https://osf.io/preprints",
            "https://hal.science",
            "https://www.researchgate.net",
        ],
    ),
    SourceCategory(
        name="Education & Open Courseware",
        description="Structured learning materials from universities and education platforms.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Medical & Health Authorities",
        description="Evidence-based medical information and clinical guidance.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Government & Official Statistics",
        description="Official data portals, statistical agencies, and government research.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Science & Technology Agencies",
        description="National laboratories and agencies publishing technical research.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Libraries & Archives",
        description="Digital library collections and archives.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Data Portals & Repositories",
        description="Curated datasets for academic and policy research.",
        seeds=[
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
        ],
    ),
    SourceCategory(
        name="Fact-Checking & Verification",
        description="Fact-checked journalism and verification resources.",
        seeds=[
            "https://www.reuters.com",
            "https://www.apnews.com/apfactcheck",
            "https://www.factcheck.org",
            "https://www.politifact.com",
            "https://www.snopes.com",
            "https://www.bbc.com/news/reality_check",
        ],
    ),
]


def iter_all_seed_urls() -> Iterable[str]:
    for category in CATEGORIES:
        yield from category.seeds


ALL_TRUSTED_SEEDS: List[str] = sorted(set(iter_all_seed_urls()))


def derive_domain_suffixes(urls: Iterable[str]) -> Set[str]:
    from urllib.parse import urlparse

    suffixes: Set[str] = set()
    for raw_url in urls:
        parsed = urlparse(raw_url)
        host = parsed.netloc.lower()
        if not host:
            continue
        suffixes.add(host)
        if host.startswith("www."):
            suffixes.add(host[4:])
    return suffixes


ALL_TRUSTED_DOMAINS: Set[str] = derive_domain_suffixes(ALL_TRUSTED_SEEDS)


def category_map() -> Dict[str, List[str]]:
    return {category.name: category.seeds for category in CATEGORIES}
