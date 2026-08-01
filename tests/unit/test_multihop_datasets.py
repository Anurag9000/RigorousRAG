import json

import pytest

from tools.multihop_datasets import load_multihop_dataset
from tools.multihop_evaluation import (
    MultiHopEvaluationPrediction,
    evaluate_multihop_example,
)


def test_hotpotqa_adapter_preserves_sentence_support_and_aliases(tmp_path):
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "hp-1",
                    "question": "Who founded the company?",
                    "answer": "Ada",
                    "answers": ["Ada Lovelace"],
                    "supporting_facts": [["Company", 0], ["Ada", 2]],
                    "context": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_multihop_dataset(path, dataset="hotpotqa", split="validation")
    example = loaded.examples[0]
    assert loaded.dataset == "hotpotqa"
    assert len(loaded.sha256) == 64
    assert example.answers == ("Ada", "Ada Lovelace")
    assert example.required_hops == 2
    assert {(fact.document_id, fact.locator) for fact in example.support_facts} == {
        ("Company", "sentence:0"),
        ("Ada", "sentence:2"),
    }

    metrics = evaluate_multihop_example(
        example,
        MultiHopEvaluationPrediction(
            answer="Ada Lovelace",
            evidence=(
                {
                    "doc_id": "Company",
                    "hop_id": "q1",
                    "source_id": "company-0",
                    "metadata": {"sentence_id": 0},
                },
                {
                    "doc_id": "Ada",
                    "hop_id": "q2",
                    "source_id": "ada-2",
                    "metadata": {"sentence_index": 2},
                },
            ),
        ),
    )
    assert metrics.path_complete is True
    assert metrics.support_recall == 1.0


def test_2wiki_alias_and_duplicate_support_deduplication(tmp_path):
    path = tmp_path / "2wiki.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "wiki-1",
                        "question": "Question",
                        "answer": "Answer",
                        "supporting_facts": [["Doc", 1], ["Doc", 1]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_multihop_dataset(path, dataset="2wiki", split="dev")
    assert loaded.dataset == "2wikimultihopqa"
    assert loaded.examples[0].support_facts[0].locator == "sentence:1"
    assert len(loaded.examples[0].support_facts) == 1


def test_musique_jsonl_adapter_maps_support_paragraphs_and_steps(tmp_path):
    path = tmp_path / "musique.jsonl"
    record = {
        "id": "mu-1",
        "question": "What links the two people?",
        "answer": "A result",
        "answer_aliases": ["Result"],
        "paragraphs": [
            {"idx": 0, "title": "First", "paragraph_text": "A", "is_supporting": True},
            {"idx": 3, "title": "Second", "paragraph_text": "B", "is_supporting": False},
        ],
        "question_decomposition": [
            {"id": 0, "question": "First?", "answer": "A", "paragraph_support_idx": 0},
            {"id": 1, "question": "Second?", "answer": "B", "paragraph_support_idx": 3},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_multihop_dataset(path, dataset="musique", split="validation")
    example = loaded.examples[0]
    assert example.required_hops == 2
    assert {(fact.document_id, fact.locator) for fact in example.support_facts} == {
        ("First", "paragraph:0"),
        ("Second", "paragraph:3"),
    }
    metrics = evaluate_multihop_example(
        example,
        MultiHopEvaluationPrediction(
            answer="Result",
            evidence=(
                {
                    "doc_id": "First",
                    "hop_id": "q1",
                    "source_id": "first",
                    "metadata": {"paragraph_index": 0},
                },
                {
                    "doc_id": "Second",
                    "hop_id": "q2",
                    "source_id": "second",
                    "metadata": {"paragraph_support_idx": 3},
                },
            ),
        ),
    )
    assert metrics.path_complete is True
    assert metrics.answer_exact_match == 1.0


def test_loader_rejects_duplicate_json_keys_nonstandard_numbers_and_missing_support(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","id":"b","question":"q","answer":"a","paragraphs":[]}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        load_multihop_dataset(path, dataset="musique", split="test")
    path.write_text('{"id":"a","question":"q","answer":"a","score":NaN,"paragraphs":[]}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        load_multihop_dataset(path, dataset="musique", split="test")
    path.write_text(
        json.dumps(
            {
                "id": "a",
                "question": "q",
                "answer": "a",
                "paragraphs": [],
                "question_decomposition": [{"paragraph_support_idx": 4}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="not declared"):
        load_multihop_dataset(path, dataset="musique", split="test")


def test_loader_rejects_symlinks_duplicate_ids_and_boolean_limits(tmp_path):
    target = tmp_path / "data.json"
    payload = [
        {"_id": "same", "question": "q1", "answer": "a", "supporting_facts": []},
        {"_id": "same", "question": "q2", "answer": "a", "supporting_facts": []},
    ]
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_multihop_dataset(target, dataset="hotpotqa", split="test")
    with pytest.raises(ValueError, match="max_examples"):
        load_multihop_dataset(target, dataset="hotpotqa", split="test", max_examples=True)
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        load_multihop_dataset(link, dataset="hotpotqa", split="test")
