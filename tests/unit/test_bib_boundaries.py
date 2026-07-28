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
            "title": r"A $ B ~ C ^ D # E % F & G _ H {I} \",
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

    assert output.count("@misc{") == 99
    assert "Reference 98" in output
    assert "Reference 99" not in output
