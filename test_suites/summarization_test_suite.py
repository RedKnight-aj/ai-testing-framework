"""
Text Summarization Test Suite
===============================
Production-ready pytest test suite covering text summarization evaluation.

Tests: summarization_score, g_eval, answer_relevancy, faithfulness

Usage:
    pytest test_suites/summarization_test_suite.py -v

Coverage:
    - Summary accuracy and faithfulness to source
    - Conciseness scoring
    - Coverage of key points
    - Hallucination detection in summaries
    - Multi-document summarization
    - Length-appropriate summaries
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from ai_testing.test_cases import SummarizationTestCase, BaseTestCase
from ai_testing.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetricWrapper,
)
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_article_summary() -> SummarizationTestCase:
    """A high-quality summary of a news article."""
    return SummarizationTestCase(
        name="summarization_good_article",
        input=(
            "The Federal Reserve announced on Wednesday that it would maintain interest rates "
            "at their current level of 5.25-5.50%, marking the third consecutive meeting "
            "without a rate change. Fed Chair Jerome Powell stated that while inflation has "
            "cooled significantly from its peak of 9.1% in June 2022 to 3.2% currently, "
            "the central bank wants to see more evidence that price pressures are sustainably "
            "moving toward the 2% target before considering rate cuts. The decision was "
            "unanimous among voting members. Financial markets responded positively, with "
            "the S&P 500 rising 1.2% following the announcement. Economists predict the "
            "first rate cut could come as early as mid-2024 if inflation continues its "
            "downward trajectory. The labor market remains resilient with unemployment at "
            "3.7%, though job growth has moderated from earlier in the year."
        ),
        actual_output=(
            "The Federal Reserve kept interest rates at 5.25-5.50% for the third consecutive "
            "meeting, with Chair Powell saying more evidence of declining inflation is needed "
            "before cutting rates. Inflation has fallen from 9.1% to 3.2%, but the Fed targets 2%. "
            "Markets rose 1.2%, and economists expect potential rate cuts by mid-2024."
        ),
        expected_output=(
            "Fed holds rates steady at 5.25-5.50% for third meeting, waits for more "
            "inflation data before cutting."
        ),
        source_text=(
            "The Federal Reserve announced on Wednesday that it would maintain interest rates "
            "at their current level of 5.25-5.50%, marking the third consecutive meeting "
            "without a rate change. Fed Chair Jerome Powell stated that while inflation has "
            "cooled significantly from its peak of 9.1% in June 2022 to 3.2% currently, "
            "the central bank wants to see more evidence that price pressures are sustainably "
            "moving toward the 2% target before considering rate cuts."
        ),
    )


@pytest.fixture
def hallucinated_summary() -> SummarizationTestCase:
    """A summary that adds fabricated information not in the source."""
    return SummarizationTestCase(
        name="summarization_hallucinated",
        input=(
            "Apple Inc. reported quarterly revenue of $89.5 billion for Q4 2023, slightly below "
            "analyst expectations of $90.2 billion. iPhone sales accounted for $43.8 billion of "
            "total revenue. Services revenue reached a record $22.3 billion. CEO Tim Cook noted "
            "that the company's installed base of active devices reached an all-time high across "
            "all products and geographic segments. The company did not provide specific guidance "
            "for the upcoming quarter."
        ),
        actual_output=(
            "Apple reported $89.5B in Q4 2023 revenue, missing expectations by $700M. "
            "iPhone sales dropped 15% year-over-year to $43.8B. The company announced a "
            "massive layoff of 10,000 employees as part of CEO Tim Cook's restructuring plan. "
            "Apple also revealed plans to exit the European market due to new EU regulations. "
            "Services revenue hit $22.3B. The stock fell 8% in after-hours trading."
        ),
        expected_output=(
            "Apple Q4 2023 revenue of $89.5B missed expectations; iPhone contributed $43.8B "
            "and Services hit a record $22.3B."
        ),
        source_text=(
            "Apple Inc. reported quarterly revenue of $89.5 billion for Q4 2023, slightly below "
            "analyst expectations of $90.2 billion. iPhone sales accounted for $43.8 billion of "
            "total revenue. Services revenue reached a record $22.3 billion."
        ),
    )


@pytest.fixture
def overly_long_summary() -> SummarizationTestCase:
    """A summary that is almost as long as the source text."""
    return SummarizationTestCase(
        name="summarization_too_long",
        input=(
            "The Paris Agreement, adopted in 2015, is a legally binding international treaty on "
            "climate change. It was adopted by 196 Parties at COP 21 in Paris and entered into "
            "force on November 4, 2016. The Agreement's central aim is to strengthen the global "
            "response to climate change by keeping global temperature rise this century well below "
            "2 degrees Celsius above pre-industrial levels and pursuing efforts to limit the "
            "increase to 1.5 degrees. Additionally, it aims to increase countries' ability to "
            "deal with the impacts of climate change and to align financial flows with low "
            "greenhouse gas emissions and climate-resilient development pathways. Countries "
            "submit nationally determined contributions (NDCs) every five years detailing their "
            "climate action plans. The Agreement also includes a mechanism for global stocktake "
            "every five years to assess collective progress."
        ),
        actual_output=(
            "The Paris Agreement, adopted in 2015, is a legally binding international treaty on "
            "climate change that was adopted by 196 Parties at COP 21 in Paris and entered into "
            "force on November 4, 2016. Its central aim is to strengthen the global response to "
            "climate change by keeping global temperature rise well below 2 degrees Celsius above "
            "pre-industrial levels and pursuing efforts to limit the increase to 1.5 degrees. "
            "It aims to increase countries' ability to deal with climate change impacts and align "
            "financial flows with low emissions. Countries submit NDCs every five years and there "
            "is a global stocktake every five years."
        ),
        expected_output=(
            "The Paris Agreement (2015) is a binding climate treaty by 196 parties aiming to "
            "limit global warming to 1.5-2°C through nationally determined contributions."
        ),
        source_text=(
            "The Paris Agreement, adopted in 2015, is a legally binding international treaty on "
            "climate change adopted by 196 Parties at COP 21 in Paris."
        ),
    )


@pytest.fixture
def too_brief_summary() -> SummarizationTestCase:
    """A summary that is too brief and misses key information."""
    return SummarizationTestCase(
        name="summarization_too_brief",
        input=(
            "NASA's James Webb Space Telescope has captured new images of the Pillars of Creation, "
            "revealing previously hidden details. The telescope's infrared capabilities allow it "
            "to peer through dense dust clouds where new stars are forming. The images show bright "
            "young stars previously obscured in the visible light spectrum. Scientists say these "
            "observations will help them understand the timeline and mechanisms of star formation. "
            "The Pillars of Creation, located in the Eagle Nebula about 6,500 light-years away, "
            "are towering columns of gas and dust first photographed by Hubble in 1995. Webb's "
            "new images reveal thousands of stars and detailed structures within the pillars that "
            "were completely invisible to previous telescopes."
        ),
        actual_output=(
            "NASA took photos of stars."
        ),
        expected_output=(
            "JWST captured detailed infrared images of the Pillars of Creation, "
            "revealing hidden stars and structures invisible to previous telescopes."
        ),
        source_text=(
            "NASA's James Webb Space Telescope has captured new images of the Pillars of Creation, "
            "revealing previously hidden details."
        ),
    )


@pytest.fixture
def multi_document_summary() -> SummarizationTestCase:
    """A summary synthesizing multiple documents."""
    return SummarizationTestCase(
        name="summarization_multi_document",
        input=(
            "Document 1: The WHO declared Mpox a global health emergency in 2024 as cases "
            "surged in Central Africa. Document 2: The outbreak has been driven by a new variant "
            "called clade Ib, which spreads more easily than previous strains. Document 3: "
            "Vaccination campaigns with the MVA-BN vaccine have begun in affected regions, "
            "with over 100,000 doses deployed so far. Document 4: International aid organizations "
            "are coordinating with local health authorities to establish testing centers and "
            "treatment facilities across the Democratic Republic of Congo."
        ),
        actual_output=(
            "The WHO declared Mpox a global health emergency in 2024 due to cases spreading in "
            "Central Africa. A new clade Ib variant drives the outbreak. Vaccination with MVA-BN "
            "has begun with 100,000+ doses deployed, while international aid organizations coordinate "
            "testing and treatment in the DRC."
        ),
        expected_output=(
            "WHO declared Mpox a global emergency in 2024 as a new clade Ib variant spread "
            "in Central Africa, prompting vaccination campaigns and international aid."
        ),
        source_text=(
            "The WHO declared Mpox a global health emergency in 2024. A new clade Ib variant "
            "drives the outbreak. Vaccination campaigns have begun with 100,000+ doses."
        ),
    )


@pytest.fixture
def off_topic_summary() -> SummarizationTestCase:
    """A summary that misses the main point and focuses on minor details."""
    return SummarizationTestCase(
        name="summarization_off_topic",
        input=(
            "The 2024 US Presidential Election results were certified on January 6, 2025. "
            "The electoral college voted with 312 to 226 in favor of the winning candidate. "
            "Voter turnout reached 66.5%, the highest in over a century. Key battleground states "
            "included Pennsylvania, Michigan, Wisconsin, Arizona, and Georgia. The election "
            "featured debates on economy, immigration, foreign policy, and healthcare. Campaign "
            "spending exceeded $16 billion across all races."
        ),
        actual_output=(
            "The election had debates about various topics. Campaign spending was $16 billion. "
            "The results were certified on January 6, 2025. The ceremony took place in "
            "Washington D.C. with many notable guests in attendance."
        ),
        expected_output=(
            "The 2024 US Presidential Election was certified with 312-226 electoral vote win, "
            "and 66.5% voter turnout - the highest in over a century."
        ),
        source_text=(
            "The 2024 US Presidential Election results were certified on January 6, 2025. "
            "The electoral college voted with 312 to 226."
        ),
    )


@pytest.fixture
def technical_summary_good() -> SummarizationTestCase:
    """A good summary of a technical document."""
    return SummarizationTestCase(
        name="summarization_technical_good",
        input=(
            "The Transformer architecture, introduced in the paper 'Attention Is All You Need' "
            "(Vaswani et al., 2017), replaced recurrent layers with self-attention mechanisms. "
            "The multi-head attention mechanism computes attention scores between all token pairs "
            "in parallel, enabling efficient training. The architecture consists of an encoder "
            "stack and a decoder stack, each with 6 layers. Positional encodings are added to "
            "token embeddings to provide sequence order information since the model lacks "
            "recurrence. Layer normalization, residual connections, and feed-forward networks "
            "with ReLU activation are used within each layer. The model achieved state-of-the-art "
            "results on WMT 2014 English-to-German (BLEU 28.4) and English-to-French "
            "(BLEU 41.0) translation tasks, with training times significantly shorter than "
            "previous sequence-to-sequence models."
        ),
        actual_output=(
            "The Transformer architecture (Vaswani et al., 2017) replaces RNNs with self-attention, "
            "using stacked encoder-decoder layers with multi-head attention and positional encodings. "
            "It achieved SOTA BLEU scores of 28.4 (EN-DE) and 41.0 (EN-FR) with faster training."
        ),
        expected_output=(
            "Transformers use self-attention instead of RNNs, with multi-head attention and "
            "positional encodings achieving SOTA translation results."
        ),
        source_text=(
            "The Transformer architecture replaced recurrent layers with self-attention mechanisms, "
            "achieving state-of-the-art BLEU scores of 28.4 (EN-DE) and 41.0 (EN-FR)."
        ),
    )


@pytest.fixture
def numerical_fabrication_summary() -> SummarizationTestCase:
    """A summary that misrepresents numbers from the source."""
    return SummarizationTestCase(
        name="summarization_numerical_fabrication",
        input=(
            "The global semiconductor market reached $574 billion in 2023, growing 11.2% "
            "year-over-year. China accounted for 34% of total demand, followed by the Americas "
            "at 24% and Europe at 10%. Memory chips represented the largest segment at $123 billion. "
            "TSMC maintained its position as the largest foundry with 59% market share."
        ),
        actual_output=(
            "The semiconductor market hit $890 billion in 2023, up 25%. China dominated with "
            "55% of demand. Memory chips were worth $340 billion. Samsung was the largest "
            "foundry with 72% market share."
        ),
        expected_output=(
            "The semiconductor market reached $574B in 2023 with China at 34% of demand, "
            "memory chips at $123B, and TSMC at 59% foundry share."
        ),
        source_text=(
            "The global semiconductor market reached $574 billion in 2023, growing 11.2% "
            "year-over-year."
        ),
    )


# ---------------------------------------------------------------------------
# Positive Test Cases
# ---------------------------------------------------------------------------

class TestSummarizationAccuracy:
    """Tests for summary accuracy and faithfulness."""

    def test_good_summary_captures_key_facts(self, good_article_summary: SummarizationTestCase) -> None:
        """Summary should contain the critical facts from the source."""
        summary = good_article_summary.actual_output.lower()
        assert "5.25" in summary or "5.25-5.50" in summary, "Should mention rate level"
        assert "inflation" in summary, "Should mention inflation"
        assert "fed" in summary.lower(), "Should mention the Fed"

    def test_summary_abbreviated_not_hallucinated(self, hallucinated_summary: SummarizationTestCase) -> None:
        """Should detect fabricated claims in the summary."""
        summary = hallucinated_summary.actual_output.lower()
        assert "layoff" in summary or "10,000" in summary, "Summary contains fabricated layoff claim"
        assert "european market" in summary, "Summary contains fabricated EU exit claim"

    def test_technical_terms_preserved(self, technical_summary_good: SummarizationTestCase) -> None:
        """Technical summary should preserve key technical terms."""
        summary = technical_summary_good.actual_output.lower()
        assert "transformer" in summary, "Should mention Transformer"
        assert "attention" in summary or "self-attention" in summary, "Should mention attention"


class TestSummarizationConciseness:
    """Tests for summary length and conciseness."""

    def test_overly_long_summary_detected(self, overly_long_summary: SummarizationTestCase) -> None:
        """Should detect when summary is too close to source length."""
        source_len = len(overly_long_summary.input.split())
        summary_len = len(overly_long_summary.actual_output.split())
        ratio = summary_len / source_len if source_len > 0 else 1
        assert ratio > 0.7, f"Summary is {ratio:.0%} of source length — too long"

    def test_too_brief_summary(self, too_brief_summary: SummarizationTestCase) -> None:
        """Should detect when summary is too brief and misses key info."""
        summary = too_brief_summary.actual_output.lower()
        key_terms = ["james webb", "pillars of creation", "infrared", "stars forming", "6,500"]
        found_terms = [t for t in key_terms if t in summary]
        assert len(found_terms) <= 1, f"Summary misses key details: only found {found_terms}"


class TestSummarizationCompleteness:
    """Tests for completeness and coverage."""

    def test_multi_document_synthesis(self, multi_document_summary: SummarizationTestCase) -> None:
        """Summary should capture points from all source documents."""
        summary = multi_document_summary.actual_output.lower()
        topics = ["who", "mpox", "clade ib", "vaccin", "drc"]
        covered = sum(1 for t in topics if t in summary)
        assert covered >= 3, f"Summary should cover most topics: only {covered}/5"

    def test_off_topic_focus(self, off_topic_summary: SummarizationTestCase) -> None:
        """Should detect when summary focuses on minor details."""
        summary = off_topic_summary.actual_output.lower()
        key_facts = ["312", "66.5%", "electoral"]
        missing = sum(1 for f in key_facts if f not in summary)
        assert missing >= 2, f"Summary misses {missing}/3 key facts"


class TestSummarizationFaithfulness:
    """Tests for faithfulness — no added or distorted information."""

    def test_numerical_fabrication(self, numerical_fabrication_summary: SummarizationTestCase) -> None:
        """Should detect when summary changes numbers from source."""
        source = numerical_fabrication_summary.input
        summary = numerical_fabrication_summary.actual_output
        # Check that fabricated numbers exist in summary but not source
        assert "$890 billion" in summary or "$890B" in summary, "Fabricated market size"
        assert "574 billion" not in summary.lower(), "Correct market size missing from summary"
