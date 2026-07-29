import pytest

import tools.bib as bib
from tools.bib import export_to_bibtex


def test_incomplete_article_falls_back_to_misc():
    output = export_to_bibtex([
        {
            "entry_type": "article",
            "title": "Study",
            "authors": "Doe, Jane",
            "year": 2026,
        }
    ])

    assert output.startswith("@misc{")
    assert "title = {Study}" in output
    assert "author = {Doe, Jane}" in output


def test_complete_typed_entries_keep_their_type():
    article = export_to_bibtex([
        {
            "entry_type": "article",
            "title": "Study",
            "authors": "Doe, Jane",
            "year": 2026,
            "journal": "Journal of Tests",
        }
    ])
    thesis = export_to_bibtex([
        {
            "entry_type": "phdthesis",
            "title": "Thesis",
            "authors": "Doe, Jane",
            "year": 2026,
            "school": "Example University",
        }
    ])

    assert article.startswith("@article{")
    assert thesis.startswith("@phdthesis{")


def test_all_common_latex_special_characters_are_escaped_once():
    output = export_to_bibtex([
        {
            "entry_type": "misc",
            "title": "A $ B ~ C ^ D # E % F & G _ H {I} backslash \\ end",
        }
    ])

    assert r"\$" in output
    assert r"\textasciitilde{}" in output
    assert r"\textasciicircum{}" in output
    assert r"\#" in output
    assert r"\%" in output
    assert r"\&" in output
    assert r"\_" in output
    assert r"\{I\}" in output
    assert r"\textbackslash{}" in output


def test_direct_export_skips_invalid_items_and_caps_at_one_hundred():
    citations = ["not-a-mapping"] + [
        {"entry_type": "misc", "title": f"Reference {index}"}
        for index in range(150)
    ]

    output = export_to_bibtex(citations)

    assert output.count("@misc{") == 100
    assert "Reference 99" in output
    assert "Reference 100" not in output


def test_infinite_invalid_iterable_is_bounded():
    inspected = []

    def invalid_items():
        index = 0
        while True:
            inspected.append(index)
            yield "not-a-mapping"
            index += 1

    output = export_to_bibtex(invalid_items())

    assert output == ""
    assert len(inspected) == 1000


def test_direct_fields_are_bounded_before_escape_and_key_hashing():
    output = export_to_bibtex([
        {
            "entry_type": "misc",
            "title": "T" * 500_000,
            "authors": "A" * 500_000,
            "url": "https://example.test/" + "u" * 500_000,
        }
    ])

    title_line = next(line for line in output.splitlines() if "title =" in line)
    author_line = next(line for line in output.splitlines() if "author =" in line)
    url_line = next(line for line in output.splitlines() if "url =" in line)
    assert title_line.count("T") == 1000
    assert author_line.count("A") == 3000
    assert len(url_line) < 4200
    assert len(output) < 9000


def test_credentials_paths_and_pii_are_masked_before_bibtex_escape():
    output = export_to_bibtex([
        {
            "entry_type": "misc",
            "title": "Contact analyst@example.com at /private/report.txt",
            "authors": "alice@example.com",
            "url": "https://alice:password@example.test/paper?api_key=secret",
        }
    ])

    assert "analyst@example.com" not in output
    assert "alice@example.com" not in output
    assert "/private" not in output
    assert "password" not in output
    assert "api_key=secret" not in output
    assert "REDACTED" in output


def test_unsupported_or_hostile_field_values_fall_back_without_stringification():
    class Hostile:
        def __bool__(self):
            raise RuntimeError("do not call bool")

        def __str__(self):
            raise RuntimeError("private /secret/path")

    output = export_to_bibtex([
        {
            "entry_type": Hostile(),
            "title": Hostile(),
            "authors": Hostile(),
            "year": True,
            "journal": {"nested": "value"},
        }
    ])

    assert output.startswith("@misc{")
    assert "title = {Untitled}" in output
    assert "author = {Unknown}" in output
    assert "year = {n.d.}" in output
    assert "private" not in output


def test_string_collections_and_noniterable_citations_are_rejected():
    for value in (None, "citation text", b"citation bytes"):
        with pytest.raises(ValueError, match="citations must be an iterable"):
            export_to_bibtex(value)


def test_iterator_failure_is_wrapped_without_private_details():
    class BrokenIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise RuntimeError("private iterator path /secret/state")

    with pytest.raises(ValueError, match="iteration failed") as captured:
        export_to_bibtex(BrokenIterator())
    assert "private iterator" not in str(captured.value)


def test_total_output_ceiling_stops_before_oversized_entry(monkeypatch):
    monkeypatch.setattr(bib, "_MAX_OUTPUT_CHARS", 200)

    output = export_to_bibtex([
        {"entry_type": "misc", "title": "x" * 1000},
        {"entry_type": "misc", "title": "small"},
    ])

    assert output == ""
    assert len(output) <= 200
