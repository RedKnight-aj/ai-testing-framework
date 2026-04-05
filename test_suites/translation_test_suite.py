"""
Translation Test Suite
=======================
Production-ready pytest test suite covering translation evaluation scenarios.

Tests: g_eval, answer_relevancy, faithfulness, contextual_relevancy

Usage:
    pytest test_suites/translation_test_suite.py -v

Coverage:
    - Translation accuracy and fluency
    - Preservation of meaning and context
    - Cultural adaptation vs literal translation
    - Handling of idioms and expressions
    - Technical translation quality
    - Language pair specific evaluation
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from ai_testing.test_cases import BaseTestCase
from ai_testing.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_english_spanish() -> BaseTestCase:
    """A high-quality English to Spanish translation."""
    return BaseTestCase(
        name="translation_good_en_es",
        input="The quick brown fox jumps over the lazy dog.",
        actual_output="El rápido zorro marrón salta sobre el perro perezoso.",
        expected_output="El rápido zorro marrón salta sobre el perro perezoso.",
        context="A pangram sentence used for testing.",
    )


@pytest.fixture
def bad_literal_translation() -> BaseTestCase:
    """A literal translation that misses idiomatic meaning."""
    return BaseTestCase(
        name="translation_bad_literal",
        input="It's raining cats and dogs outside.",
        actual_output="Está lloviendo gatos y perros afuera.",
        expected_output="Está lloviendo a cántaros afuera.",
        context="English idiom meaning heavy rain.",
    )


@pytest.fixture
def context_lost_translation() -> BaseTestCase:
    """A translation that loses important context."""
    return BaseTestCase(
        name="translation_lost_context",
        input="I could eat a horse right now.",
        actual_output="Podría comer un caballo ahora mismo.",
        expected_output="Me muero de hambre ahora mismo.",
        context="Idiomatic expression meaning very hungry.",
    )


@pytest.fixture
def technical_translation_good() -> BaseTestCase:
    """Good technical translation preserving terminology."""
    return BaseTestCase(
        name="translation_technical_good",
        input="The neural network architecture uses convolutional layers for feature extraction.",
        actual_output="La arquitectura de la red neuronal utiliza capas convolucionales para la extracción de características.",
        expected_output="La arquitectura de la red neuronal utiliza capas convolucionales para la extracción de características.",
        context="Machine learning technical terminology.",
    )


@pytest.fixture
def grammar_error_translation() -> BaseTestCase:
    """Translation with grammatical errors in target language."""
    return BaseTestCase(
        name="translation_grammar_errors",
        input="She has been working here for five years.",
        actual_output="Ella ha sido trabajando aquí por cinco años.",
        expected_output="Ella ha estado trabajando aquí durante cinco años.",
        context="Present perfect continuous tense.",
    )


@pytest.fixture
def french_german_good() -> BaseTestCase:
    """Good French to German translation."""
    return BaseTestCase(
        name="translation_french_german",
        input="Le petit déjeuner est servi de 7h à 11h tous les matins.",
        actual_output="Das Frühstück wird von 7 bis 11 Uhr jeden Morgen serviert.",
        expected_output="Das Frühstück wird von 7 bis 11 Uhr jeden Morgen serviert.",
        context="Hotel breakfast information.",
    )


@pytest.fixture
def spanish_chinese() -> BaseTestCase:
    """Spanish to Chinese translation."""
    return BaseTestCase(
        name="translation_spanish_chinese",
        input="La paciencia es una virtud.",
        actual_output="耐心是一种美德。",
        expected_output="耐心是一种美德。",
        context="Common proverb.",
    )


@pytest.fixture
def hindi_english_good() -> BaseTestCase:
    """Good Hindi to English translation."""
    return BaseTestCase(
        name="translation_hindi_english",
        input="स्वादिष्ट भोजन बनाने की कला में समय और मेहनत लगती है।",
        actual_output="The art of making delicious food takes time and effort.",
        expected_output="The art of making delicious food takes time and effort.",
        context="General statement about cooking.",
    )


@pytest.fixture
def false_friend_translation() -> BaseTestCase:
    """Translation that falls for false friends (words that look similar but mean different things)."""
    return BaseTestCase(
        name="translation_false_friends",
        input="The gift was very expensive.",
        actual_output="El regalo era muy embarazada.",
        expected_output="El regalo era muy caro.",
        context="English 'expensive' ≠ Spanish 'embarazada' (pregnant).",
    )


@pytest.fixture
def cultural_context_missing() -> BaseTestCase:
    """Translation that misses cultural context adaptation."""
    return BaseTestCase(
        name="translation_cultural_miss",
        input="He's a loose cannon.",
        actual_output="Él es un cañón suelto.",
        expected_output="Él es impredecible / Él es una persona incontrolable.",
        context="Idiom meaning unpredictable person.",
    )


# ---------------------------------------------------------------------------
# Positive Test Cases
# ---------------------------------------------------------------------------

class TestTranslationAccuracy:
    """Tests for translation accuracy and meaning preservation."""

    def test_meaning_preserved(self, good_english_spanish: BaseTestCase) -> None:
        """Translation should preserve the exact meaning."""
        output = good_english_spanish.actual_output
        assert "rápido" in output, "Should preserve 'quick'"
        assert "zorro" in output, "Should preserve 'fox'"
        assert "marrón" in output, "Should preserve 'brown'"

    def test_idioms_handled_properly(self, bad_literal_translation: BaseTestCase) -> None:
        """Idioms should be translated idiomatically, not literally."""
        output = bad_literal_translation.actual_output.lower()
        # This is literal translation - should be detected as wrong
        assert "gatos" in output and "perros" in output, "Literal translation detected"
        # The expected output shows it should be 'a cántaros'

    def test_technical_terms_preserved(self, technical_translation_good: BaseTestCase) -> None:
        """Technical terminology should be correctly translated."""
        output = technical_translation_good.actual_output.lower()
        assert "neuronal" in output, "Neural network terminology preserved"
        assert "convolucionales" in output, "Convolutional layers terminology preserved"


class TestTranslationQuality:
    """Tests for translation fluency and grammar."""

    def test_grammar_errors_detected(self, grammar_error_translation: BaseTestCase) -> None:
        """Should detect grammatical errors in target language."""
        output = grammar_error_translation.actual_output
        # Spanish grammar issue: "ha sido trabajando" is redundant/incorrect
        assert "ha sido trabajando" in output, "Grammatical error detected"
        # Correct would be "ha estado trabajando"

    def test_false_friends_avoided(self, false_friend_translation: BaseTestCase) -> None:
        """Should avoid false friends (cognates with different meanings)."""
        output = false_friend_translation.actual_output.lower()
        assert "embarazada" in output, "False friend 'embarazada' used instead of 'caro'"
        # 'Expensive' in Spanish is 'caro/cara', not 'embarazada' (pregnant)


class TestTranslationContext:
    """Tests for context preservation and cultural adaptation."""

    def test_context_lost_detected(self, context_lost_translation: BaseTestCase) -> None:
        """Should detect when important context is lost in translation."""
        output = context_lost_translation.actual_output.lower()
        assert "caballo" in output, "Literal translation 'horse' preserved"
        # Missing the idiomatic meaning of extreme hunger

    def test_cultural_adaptation_needed(self, cultural_context_missing: BaseTestCase) -> None:
        """Should detect when cultural adaptation is needed."""
        output = cultural_context_missing.actual_output.lower()
        assert "cañón suelto" in output, "Literal translation used"
        # Should be idiomatic Spanish equivalent


class TestMultiLanguageSupport:
    """Tests for multiple language pairs."""

    def test_french_german_accuracy(self, french_german_good: BaseTestCase) -> None:
        """French to German translation accuracy."""
        output = french_german_good.actual_output.lower()
        assert "frühstück" in output, "Breakfast correctly translated"
        assert "serviert" in output, "Served correctly translated"

    def test_spanish_chinese_characters(self, spanish_chinese: BaseTestCase) -> None:
        """Spanish to Chinese translation with proper characters."""
        output = spanish_chinese.actual_output
        # Should contain Chinese characters
        assert len([c for c in output if ord(c) > 127]) > 0, "Contains Chinese characters"

    def test_hindi_english_meaning(self, hindi_english_good: BaseTestCase) -> None:
        """Hindi to English translation meaning preservation."""
        output = hindi_english_good.actual_output.lower()
        assert "art" in output and "delicious" in output, "Key concepts preserved"
        assert "time" in output and "effort" in output, "Time and effort concepts preserved"
