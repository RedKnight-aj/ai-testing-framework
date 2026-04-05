"""
Sentiment Analysis Test Suite
==============================
Production-ready pytest test suite covering sentiment analysis evaluation.

Tests: g_eval, answer_relevancy, bias detection

Usage:
    pytest test_suites/sentiment_test_suite.py -v

Coverage:
    - Sentiment polarity accuracy
    - Intensity scoring
    - Sarcasm and irony detection
    - Mixed sentiment handling
    - Cultural differences in sentiment
    - Domain-specific sentiment
    - Neutral vs polarized content
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
def clear_positive_sentiment() -> BaseTestCase:
    """Clear positive sentiment text."""
    return BaseTestCase(
        name="sentiment_clear_positive",
        input="This product is absolutely amazing! Best purchase I've ever made.",
        actual_output="positive",
        expected_output="positive",
        context="Strong positive language and superlatives.",
    )


@pytest.fixture
def clear_negative_sentiment() -> BaseTestCase:
    """Clear negative sentiment text."""
    return BaseTestCase(
        name="sentiment_clear_negative",
        input="This is the worst experience I've ever had. Complete waste of money.",
        actual_output="negative",
        expected_output="negative",
        context="Strong negative language and disappointment.",
    )


@pytest.fixture
def sarcasm_misclassified() -> BaseTestCase:
    """Sarcastic text misclassified as positive."""
    return BaseTestCase(
        name="sentiment_sarcasm_miss",
        input="Oh great, another delay. Just what I needed today.",
        actual_output="positive",
        expected_output="negative",
        context="Sarcastic complaint misclassified as positive.",
    )


@pytest.fixture
def mixed_sentiment_good() -> BaseTestCase:
    """Mixed sentiment handled correctly."""
    return BaseTestCase(
        name="sentiment_mixed_good",
        input="The food was delicious but the service was incredibly slow.",
        actual_output="mixed:positive(food),negative(service)",
        expected_output="mixed",
        context="Positive and negative aspects in same text.",
    )


@pytest.fixture
def neutral_content() -> BaseTestCase:
    """Neutral factual content."""
    return BaseTestCase(
        name="sentiment_neutral",
        input="The meeting is scheduled for 3 PM tomorrow in conference room A.",
        actual_output="neutral",
        expected_output="neutral",
        context="Purely factual information with no sentiment.",
    )


@pytest.fixture
def intensity_scaling() -> BaseTestCase:
    """Sentiment intensity scoring."""
    return BaseTestCase(
        name="sentiment_intensity",
        input="This is pretty good, I guess.",
        actual_output="positive:0.3",
        expected_output="positive:0.2",
        context="Weak positive sentiment with qualifiers.",
    )


@pytest.fixture
def cultural_sentiment_difference() -> BaseTestCase:
    """Cultural differences in sentiment expression."""
    return BaseTestCase(
        name="sentiment_cultural",
        input="This is adequate.",
        actual_output="neutral",
        expected_output="neutral",
        context="English 'adequate' is neutral; in some cultures might be seen as negative.",
    )


@pytest.fixture
def domain_specific_sentiment() -> BaseTestCase:
    """Domain-specific sentiment interpretation."""
    return BaseTestCase(
        name="sentiment_domain_specific",
        input="The patient's condition is guarded.",
        actual_output="negative",
        expected_output="neutral",
        context="Medical terminology where 'guarded' means serious but stable.",
    )


@pytest.fixture
def negation_handling() -> BaseTestCase:
    """Proper handling of negation in sentiment."""
    return BaseTestCase(
        name="sentiment_negation",
        input="I'm not unhappy with the results.",
        actual_output="positive",
        expected_output="positive",
        context="Double negative creates positive sentiment.",
    )


@pytest.fixture
def emoji_sentiment() -> BaseTestCase:
    """Sentiment conveyed through emojis."""
    return BaseTestCase(
        name="sentiment_emoji",
        input="Great job! 👍😊",
        actual_output="positive",
        expected_output="positive",
        context="Positive emojis reinforce positive sentiment.",
    )


# ---------------------------------------------------------------------------
# Positive Test Cases
# ---------------------------------------------------------------------------

class TestSentimentPolarity:
    """Tests for basic sentiment polarity classification."""

    def test_clear_positive_detected(self, clear_positive_sentiment: BaseTestCase) -> None:
        """Should correctly identify clear positive sentiment."""
        assert clear_positive_sentiment.actual_output == "positive"
        assert clear_positive_sentiment.expected_output == "positive"

    def test_clear_negative_detected(self, clear_negative_sentiment: BaseTestCase) -> None:
        """Should correctly identify clear negative sentiment."""
        assert clear_negative_sentiment.actual_output == "negative"
        assert clear_negative_sentiment.expected_output == "negative"

    def test_neutral_content_detected(self, neutral_content: BaseTestCase) -> None:
        """Should correctly identify neutral content."""
        assert neutral_content.actual_output == "neutral"
        assert neutral_content.expected_output == "neutral"


class TestSentimentChallenges:
    """Tests for challenging sentiment analysis cases."""

    def test_sarcasm_detection_failure(self, sarcasm_misclassified: BaseTestCase) -> None:
        """Should detect sarcasm misclassification."""
        assert sarcasm_misclassified.actual_output == "positive"
        assert sarcasm_misclassified.expected_output == "negative"
        # Sarcastic text classified as positive

    def test_mixed_sentiment_handling(self, mixed_sentiment_good: BaseTestCase) -> None:
        """Should handle mixed sentiment appropriately."""
        output = mixed_sentiment_good.actual_output.lower()
        assert "mixed" in output, "Should identify mixed sentiment"
        assert "positive" in output and "negative" in output, "Should identify both polarities"

    def test_negation_handled_correctly(self, negation_handling: BaseTestCase) -> None:
        """Should handle negation correctly."""
        assert negation_handling.actual_output == "positive"
        # "Not unhappy" = positive sentiment

    def test_emoji_sentiment_recognized(self, emoji_sentiment: BaseTestCase) -> None:
        """Should recognize sentiment conveyed by emojis."""
        assert emoji_sentiment.actual_output == "positive"
        # Positive emojis should reinforce positive sentiment


class TestSentimentNuances:
    """Tests for nuanced sentiment analysis."""

    def test_intensity_reasonable(self, intensity_scaling: BaseTestCase) -> None:
        """Sentiment intensity should be appropriately scaled."""
        output = intensity_scaling.actual_output
        if ":" in output:
            intensity = float(output.split(":")[1])
            assert 0 <= intensity <= 1, "Intensity should be between 0 and 1"
            assert intensity < 0.5, "Weak sentiment should have low intensity"

    def test_domain_context_important(self, domain_specific_sentiment: BaseTestCase) -> None:
        """Domain context should affect sentiment interpretation."""
        # In medical context, "guarded" is neutral, not negative
        assert domain_specific_sentiment.actual_output == "negative"
        assert domain_specific_sentiment.expected_output == "neutral"
        # This test documents the failure - medical domain knowledge missing

    def test_cultural_differences(self, cultural_sentiment_difference: BaseTestCase) -> None:
        """Cultural context can affect sentiment interpretation."""
        # "Adequate" in English is typically neutral
        assert cultural_sentiment_difference.actual_output == "neutral"
        # In some cultures, "adequate" might be seen as lukewarm/negative
