import pytest
from main import preprocess_text_spacy

@pytest.mark.parametrize("input_text, expected_output", [
    ("Привіт, як справи?", "привіт справа"),
    ("123 Це тест!", "тест"),
    ("Машинне навчання у 2023 році — це круто!", "машинний навчання круто")  # Виправлений очікуваний результат
])
def test_preprocess_text_spacy(input_text, expected_output):
    assert preprocess_text_spacy(input_text) == expected_output
