from dataclasses import dataclass, field

import pytest

from tools.sparse_fields import build_sparse_fields


@dataclass
class Section:
    title: str
    content: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    id: str = "doc-1"
    title: str = "Trial Report"
    filename: str = "trial.pdf"
    text: str = "fallback body"
    sections: list = field(default_factory=list)


def test_deterministic_field_types_and_provenance():
    document = Document(
        sections=[
            Section("Abstract", "short abstract", 1),
            Section("Methods", "method body", 2),
            Section("Figure 1 caption", "result diagram", 3),
            Section("Table 2", "a | b", 4),
            Section("References", "Doe 2024", 5),
        ]
    )
    first = build_sparse_fields(document)
    second = build_sparse_fields(document)
    assert first == second
    assert [item.field_type for item in first] == [
        "title",
        "heading",
        "abstract",
        "heading",
        "body",
        "heading",
        "caption",
        "heading",
        "table",
        "heading",
        "reference",
    ]
    assert first[2].page_number == 1 and first[2].section == "Abstract"
    assert first[2].metadata["section_index"] == 0
    assert len({item.field_id for item in first}) == len(first)


def test_empty_sections_fall_back_to_document_text():
    fields = build_sparse_fields(Document(sections=[]))
    assert [item.field_type for item in fields] == ["title", "body"]
    assert fields[1].metadata["fallback"] is True


def test_hostile_iterators_controls_and_invalid_pages_fail_closed():
    class Hostile:
        def __iter__(self):
            yield Section("A", "ok")
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        build_sparse_fields(Document(sections=Hostile()))
    with pytest.raises(ValueError, match="control"):
        build_sparse_fields(Document(sections=[Section("A", "bad\x01text")]))
    with pytest.raises(ValueError, match="page_number"):
        build_sparse_fields(Document(sections=[Section("A", "text", True)]))
