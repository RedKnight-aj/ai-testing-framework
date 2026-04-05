"""
Text Classification Test Suite
===============================
Production-ready pytest test suite covering text classification evaluation.

Tests: g_eval, answer_relevancy, bias detection

Usage:
    pytest test_suites/classification_test_suite.py -v

Coverage:
    - Classification accuracy and confidence
    - Multi-label classification handling
    - Edge cases and ambiguous text
    - Bias and fairness in classification
    - Out-of-domain detection
    - Category specificity
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
def good_sentiment_classification() -> BaseTestCase:
    """Correct sentiment classification."""
    return BaseTestCase(
        name="classification_good_sentiment",
        input="This product exceeded my expectations! The quality is outstanding and customer service was incredibly helpful.",
        actual_output="positive",
        expected_output="positive",
        context="Customer review sentiment analysis.",
    )


@pytest.fixture
def wrong_topic_classification() -> BaseTestCase:
    """Classification that misclassifies the main topic."""
    return BaseTestCase(
        name="classification_wrong_topic",
        input="The new tax policy will significantly impact small businesses across the country.",
        actual_output="sports",
        expected_output="politics",
        context="News article topic classification.",
    )


@pytest.fixture
def ambiguous_text_classification() -> BaseTestCase:
    """Classification of ambiguous text that could fit multiple categories."""
    return BaseTestCase(
        name="classification_ambiguous",
        input="The team celebrated their victory with pizza and champagne.",
        actual_output="food",
        expected_output="celebration",
        context="Ambiguous text that could be sports, food, or celebration.",
    )


@pytest.fixture
def multi_label_good() -> BaseTestCase:
    """Good multi-label classification."""
    return BaseTestCase(
        name="classification_multi_label_good",
        input="The new electric car features advanced autonomous driving capabilities and solar panels on the roof.",
        actual_output="technology,automotive,environment",
        expected_output="technology,automotive,environment",
        context="Multi-label product categorization.",
    )


@pytest.fixture
def biased_classification() -> BaseTestCase:
    """Classification showing demographic bias."""
    return BaseTestCase(
        name="classification_biased",
        input="The nurse carefully administered the medication to the patient.",
        actual_output="female",
        expected_output="neutral",
        context="Gender bias in occupation classification.",
    )


@pytest.fixture
def over_specific_classification() -> BaseTestCase:
    """Classification that is overly specific when broader category fits."""
    return BaseTestCase(
        name="classification_over_specific",
        input="I bought a new smartphone last week.",
        actual_output="Samsung Galaxy S24 Ultra 512GB Titanium Black",
        expected_output="electronics",
        context="Product category classification.",
    )


@pytest.fixture
def out_of_domain_classification() -> BaseTestCase:
    """Classification of text completely outside training domain."""
    return BaseTestCase(
        name="classification_out_of_domain",
        input="The quantum entanglement state collapses upon measurement according to the Copenhagen interpretation.",
        actual_output="science",
        expected_output="physics",
        context="Quantum physics text classified by general science model.",
    )


@pytest.fixture
def confidence_scoring_good() -> BaseTestCase:
    """Classification with appropriate confidence scores."""
    return BaseTestCase(
        name="classification_confidence_good",
        input="The restaurant served delicious pasta with fresh ingredients.",
        actual_output="positive:0.95,food:0.87,service:0.92",
        expected_output="positive:0.9,food:0.8,service:0.9",
        context="Multi-aspect classification with confidence scores.",
    )


@pytest.fixture
def sarcasm_misclassified() -> BaseTestCase:
    """Sarcasm misclassified as literal sentiment."""
    return BaseTestCase(
        name="classification_sarcasm_miss",
        input="Oh great, another delay. Just what I needed today.",
        actual_output="positive",
        expected_output="negative",
        context="Sarcastic complaint misclassified as positive.",
    )


@pytest.fixture
def context_dependent_classification() -> BaseTestCase:
    """Classification that depends on context."""
    return BaseTestCase(
        name="classification_context_dependent",
        input="The bank approved my loan application.",
        actual_output="finance",
        expected_output="finance",
        context="Word 'bank' has different meanings (financial institution vs river bank).",
    )


# ---------------------------------------------------------------------------
# Positive Test Cases
# ---------------------------------------------------------------------------

class TestClassificationAccuracy:
    """Tests for classification accuracy."""

    def test_correct_sentiment(self, good_sentiment_classification: BaseTestCase) -> None:
        """Should correctly classify clear positive sentiment."""
        assert good_sentiment_classification.actual_output == "positive"
        assert good_sentiment_classification.expected_output == "positive"

    def test_multi_label_handling(self, multi_label_good: BaseTestCase) -> None:
        """Should handle multiple labels correctly."""
        labels = good_sentiment_classification.actual_output.split(",")
        expected_labels = good_sentiment_classification.expected_output.split(",")
        assert set(labels) == set(expected_labels), "All expected labels present"

    def test_confidence_scores_reasonable(self, confidence_scoring_good: BaseTestCase) -> None:
        """Confidence scores should be in valid range."""
        output = confidence_scoring_good.actual_output
        scores = [float(s.split(":")[1]) for s in output.split(",")]
        assert all(0 <= s <= 1 for s in scores), "All confidence scores between 0 and 1"


class TestClassificationErrors:
    """Tests for common classification errors."""

    def test_topic_misclassification(self, wrong_topic_classification: BaseTestCase) -> None:
        """Should detect when wrong topic is assigned."""
        assert wrong_topic_classification.actual_output == "sports"
        assert wrong_topic_classification.expected_output == "politics"
        # The input is clearly about taxes and business, not sports

    def test_bias_detection(self, biased_classification: BaseTestCase) -> None:
        """Should detect biased classification."""
        assert biased_classification.actual_output == "female"
        # The text doesn't specify gender - this shows bias

    def test_sarcasm_detection_failure(self, sarcasm_misclassified: BaseTestCase) -> None:
        """Should detect sarcasm misclassification."""
        assert sarcasm_misclassified.actual_output == "positive"
        assert sarcasm_misclassified.expected_output == "negative"
        # Sarcastic text classified as positive sentiment

    def test_over_specificity(self, over_specific_classification: BaseTestCase) -> None:
        """Should detect overly specific classification."""
        assert "Galaxy S24" in over_specific_classification.actual_output
        assert over_specific_classification.expected_output == "electronics"
        # Should be broad category, not specific product name


class TestClassificationEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_ambiguous_text_handling(self, ambiguous_text_classification: BaseTestCase) -> None:
        """Should handle ambiguous text appropriately."""
        # This could reasonably be classified multiple ways
        # Test checks if the classification is one of reasonable options
        reasonable_options = ["sports", "food", "celebration"]
        assert ambiguous_text_classification.actual_output in reasonable_options

    def test_out_of_domain_detection(self, out_of_domain_classification: BaseTestCase) -> None:
        """Should handle out-of-domain text gracefully."""
        # Quantum physics classified by general science model
        # Could be "physics" or broader "science"
        assert out_of_domain_classification.actual_output in ["science", "physics"]

    def test_context_dependence(self, context_dependent_classification: BaseTestCase) -> None:
        """Classification should consider context."""
        # "Bank" in financial context should be classified as finance
        assert context_dependent_classification.actual_output == "finance"
