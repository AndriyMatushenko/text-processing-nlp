import pytest
from main import TextRetriever

def test_search():
    corpus = ["Це тестовий документ.", "Ще один текст.", "Нейронні мережі – це цікаво."]
    retriever = TextRetriever(corpus)

    results = retriever.search("тест", top_n=3)

    # Друк результатів для перевірки
    print("Results:", results)

    assert len(results) >= 1  # Переконаймося, що є хоча б один результат
    assert any("тестовий документ" in res for res in results), "Очікуваний документ не знайдено"
