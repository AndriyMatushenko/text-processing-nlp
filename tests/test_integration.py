import pytest
from main import process_query


def test_process_query():
    query = "Що таке машинне навчання?"
    response, docs = process_query(query)

    # Замість перевірки списку, перевіримо, що docs — це рядок
    assert isinstance(response, str)
    assert len(response) > 0
    assert isinstance(docs, str)  # Було: isinstance(docs, list)
    assert len(docs) > 0
