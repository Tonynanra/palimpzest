import sys
from pathlib import Path


FINETUNING_DIR = Path(__file__).resolve().parents[2] / "finetuning"
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import generate_cuad_chunk_dataset as chunker  # noqa: E402


def _make_article(context: str, answer_start: int = 0, answer_text: str = ""):
    return {
        "title": "Sample Contract",
        "paragraphs": [
            {
                "context": context,
                "qas": [
                    {
                        "answers": ([] if not answer_text else [{"text": answer_text, "answer_start": answer_start}]),
                        "id": "sample-contract__Exclusivity",
                        "question": "Question",
                        "is_impossible": answer_text == "",
                    }
                ],
            }
        ],
    }


def test_chunk_article_splits_boundary_crossing_answers():
    article = _make_article("abcdefghij", answer_start=3, answer_text="defg")

    chunked = chunker.chunk_article(article, chunk_size=5)

    assert [entry["title"] for entry in chunked] == [
        "Sample Contract [p0-chunk0]",
        "Sample Contract [p0-chunk1]",
    ]
    first_chunk, second_chunk = chunked[0]["paragraphs"][0], chunked[1]["paragraphs"][0]
    assert first_chunk["context"] == "abcde"
    assert second_chunk["context"] == "fghij"
    assert first_chunk["qas"][0]["answers"] == [{"text": "de", "answer_start": 3}]
    assert second_chunk["qas"][0]["answers"] == [{"text": "fg", "answer_start": 0}]
    assert first_chunk["qas"][0]["id"] == "sample-contract__Exclusivity_chunk_0"
    assert second_chunk["qas"][0]["id"] == "sample-contract__Exclusivity_chunk_1"


def test_chunk_article_keeps_short_contract_in_single_chunk():
    article = _make_article("abc", answer_start=0, answer_text="ab")

    chunked = chunker.chunk_article(article, chunk_size=5)

    assert len(chunked) == 1
    assert chunked[0]["title"] == "Sample Contract [p0-chunk0]"
    paragraph = chunked[0]["paragraphs"][0]
    assert paragraph["chunk_index"] == 0
    assert paragraph["chunk_start"] == 0
    assert paragraph["chunk_end"] == 3
    assert paragraph["qas"][0]["answers"] == [{"text": "ab", "answer_start": 0}]


def test_chunk_articles_uses_pool_only_for_long_contracts():
    submitted_lengths = []

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, article, chunk_size):
            submitted_lengths.append(len(article["paragraphs"][0]["context"]))
            return FakeFuture(fn(article, chunk_size))

    short_article = _make_article("abcd", answer_start=1, answer_text="bc")
    long_article = _make_article("abcdefghij", answer_start=4, answer_text="ef")

    chunked_articles = chunker.chunk_articles(
        [short_article, long_article],
        chunk_size=5,
        max_workers=2,
        executor_cls=FakeExecutor,
    )

    assert submitted_lengths == [10]
    assert [article["title"] for article in chunked_articles] == [
        "Sample Contract [p0-chunk0]",
        "Sample Contract [p0-chunk0]",
        "Sample Contract [p0-chunk1]",
    ]
    assert all(len(article["paragraphs"]) == 1 for article in chunked_articles)
