import pytest

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


def test_noniterable_citations_are_rejected():
    with pytest.raises(ValueError, match="citations must be an iterable"):
        export_to_bibtex(None)
