import pytest
from main import ResponseGenerator


def test_generate_response():
    generator = ResponseGenerator()
    response = generator.generate_response("Що таке машинне навчання?",
                                           ["Машинне навчання - це метод штучного інтелекту."])

    assert isinstance(response, str)
    assert len(response) > 5
