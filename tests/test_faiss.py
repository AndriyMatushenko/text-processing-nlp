import pytest
from main import TextRetriever


def test_faiss_index():
    corpus = ["Це перший документ.", "Другий документ.", "Ще один текст."]
    retriever = TextRetriever(corpus)

    assert retriever.faiss_index.is_trained
    assert retriever.faiss_index.ntotal == len(corpus)
