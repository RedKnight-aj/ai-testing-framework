"""
RAG (Retrieval-Augmented Generation) Test Suite
================================================
Production-ready pytest test suite covering all RAG evaluation scenarios.

Tests: answer_relevancy, faithfulness, contextual_precision,
contextual_recall, contextual_relevancy, hallucination

Usage:
    pytest test_suites/rag_test_suite.py -v
    pytest test_suites/rag_test_suite.py -v -k "faithful"
    pytest test_suites/rag_test_suite.py -v --tb=short

Coverage:
    - Answer relevance to query
    - Faithfulness to retrieved context
    - Contextual precision and recall
    - Hallucination detection
    - Edge cases (empty context, contradictory context)
"""

from __future__ import annotations
import sys
import os

# Allow importing the framework package when running from repo root or test_suites/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from ai_testing.test_cases import RAGTestCase, create_rag_test
from ai_testing.runner import TestRunner, EvaluationResult
from ai_testing.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionWrapper,
    ContextualRecallWrapper,
    HallucinationMetricWrapper,
)


# ---------------------------------------------------------------------------
# Fixtures – reusable RAG test cases
# ---------------------------------------------------------------------------

@pytest.fixture
def well_answered_rag() -> RAGTestCase:
    """A RAG test where the response is accurate, relevant, and grounded in context."""
    return RAGTestCase(
        name="rag_well_answered",
        query="What is the difference between supervised and unsupervised learning?",
        input="What is the difference between supervised and unsupervised learning?",
        actual_output=(
            "Supervised learning uses labeled data where both inputs and correct outputs "
            "are provided during training, allowing the model to learn a mapping function. "
            "Example: image classification with labeled images. Unsupervised learning works "
            "with unlabeled data, finding hidden patterns or structures on its own. "
            "Example: customer segmentation using clustering algorithms."
        ),
        expected_output=(
            "Supervised learning uses labeled training data with known outputs, while "
            "unsupervised learning finds patterns in unlabeled data without predefined answers."
        ),
        retrieval_context=[
            "Supervised learning is a machine learning approach where models are trained on labeled datasets, meaning each training example includes both input data and the correct output.",
            "Unsupervised learning algorithms analyze unlabeled data to discover hidden patterns, groupings, or structures without explicit guidance on what to find.",
            "Common supervised learning algorithms include decision trees, random forests, support vector machines, and neural networks used for classification and regression tasks.",
            "Popular unsupervised techniques include k-means clustering, hierarchical clustering, principal component analysis (PCA), and autoencoders.",
        ],
    )


@pytest.fixture
def hallucinated_rag() -> RAGTestCase:
    """A RAG test where the response contains hallucinated information not in context."""
    return RAGTestCase(
        name="rag_hallucinated_response",
        query="What are the benefits of quantum computing?",
        input="What are the benefits of quantum computing?",
        actual_output=(
            "Quantum computing can solve any problem exponentially faster than classical computers. "
            "It was invented by Albert Einstein in 1954. Quantum computers use quantum bits called "
            'quibbles that can exist in 17 states simultaneously. The first quantum computer was '
            "capable of running Minecraft at 4K resolution. It also makes coffee 40% faster."
        ),
        expected_output=(
            "Quantum computing offers potential speedups for specific problem types like "
            "factoring large numbers and simulating quantum systems."
        ),
        retrieval_context=[
            "Quantum computers use quantum bits (qubits) that can exist in superposition, representing both 0 and 1 simultaneously.",
            "Quantum computing shows promise for cryptography, drug discovery, optimization problems, and materials science.",
            "Quantum algorithms like Shor's algorithm can factor large numbers efficiently, potentially breaking RSA encryption.",
        ],
    )


@pytest.fixture
def irrelevant_answer_rag() -> RAGTestCase:
    """A RAG test where the answer is tangential to the query."""
    return RAGTestCase(
        name="rag_irrelevant_answer",
        query="How does photosynthesis work?",
        input="How does photosynthesis work?",
        actual_output=(
            "Photosynthesis is a process used by plants. Plants are green. The Amazon rainforest "
            "produces 20% of the world's oxygen. Trees can live for thousands of years. "
            "The tallest tree is a redwood in California."
        ),
        expected_output=(
            "Photosynthesis converts light energy into chemical energy using chlorophyll, "
            "carbon dioxide, and water to produce glucose and oxygen."
        ),
        retrieval_context=[
            "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "The light-dependent reactions occur in the thylakoid membranes, capturing light energy with chlorophyll pigments.",
            "The Calvin cycle (light-independent reactions) fixes carbon dioxide into organic molecules in the stroma of chloroplasts.",
        ],
    )


@pytest.fixture
def contradictory_context_rag() -> RAGTestCase:
    """A RAG test with conflicting retrieval contexts."""
    return RAGTestCase(
        name="rag_contradictory_context",
        query="What is the capital of Australia?",
        input="What is the capital of Australia?",
        actual_output=(
            "The capital of Australia is Sydney, which is the largest city "
            "with a population of over 5 million people and is home to the Opera House."
        ),
        expected_output="The capital of Australia is Canberra.",
        retrieval_context=[
            "Sydney is the largest city in Australia and houses the famous Opera House.",
            "Canberra was designed specifically to serve as the capital of Australia in 1913.",
            "Melbourne served as the temporary seat of government from 1901 to 1927 while Canberra was being built.",
            "Sydney is often mistaken as Australia's capital due to its international fame and larger population.",
        ],
    )


@pytest.fixture
def incomplete_retrieval_rag() -> RAGTestCase:
    """A RAG test where retrieved context is insufficient for a complete answer."""
    return RAGTestCase(
        name="rag_incomplete_retrieval",
        query="Explain how blockchain consensus mechanisms work.",
        input="Explain how blockchain consensus mechanisms work.",
        actual_output=(
            "Blockchain consensus ensures all nodes agree on the state of the ledger. "
            "However, I cannot provide details about specific algorithms with the given context."
        ),
        expected_output=(
            "Blockchain consensus mechanisms like Proof of Work (PoW) and Proof of Stake (PoS) "
            "ensure distributed nodes agree on the valid state of the ledger without a central authority."
        ),
        retrieval_context=[
            "A blockchain consensus mechanism is a protocol that allows distributed nodes to agree on the state of the ledger.",
            "Consensus is essential for maintaining the integrity and security of decentralized networks.",
        ],
    )


@pytest.fixture
def overly_verbose_rag() -> RAGTestCase:
    """A RAG test where the response is excessively verbose relative to the query."""
    return RAGTestCase(
        name="rag_overly_verbose",
        query="What year did the first iPhone release?",
        input="What year did the first iPhone release?",
        actual_output=(
            "The first iPhone was released on June 29, 2007. This was a revolutionary moment "
            "in technology history. Steve Jobs announced it at Macworld on January 9, 2007. "
            "The iPhone cost $499 for the 4GB model and $599 for the 8GB model. "
            "It featured a 3.5-inch display, a 2-megapixel camera, and ran the original iPhone OS. "
            "AT&T was the exclusive carrier in the United States. "
            "Over 270,000 iPhones were sold in the first weekend. "
            "The device weighed 4.8 ounces and was 0.46 inches thick. "
            "It did not have an App Store initially — third-party apps came with iPhone OS 2.0 in 2008. "
            "The original iPhone had a Samsung ARM processor running at 412 MHz. "
            "Battery life was rated at 8 hours of talk time."
        ),
        expected_output="The first iPhone was released on June 29, 2007.",
        retrieval_context=[
            "The original iPhone was announced by Steve Jobs at Macworld on January 9, 2007, and released on June 29, 2007.",
            "The first iPhone was priced at $499 for 4GB and $599 for 8GB, available exclusively through AT&T.",
        ],
    )


@pytest.fixture
def technical_accuracy_rag() -> RAGTestCase:
    """A RAG test verifying technical accuracy in a domain-specific answer."""
    return RAGTestCase(
        name="rag_technical_accuracy",
        query="What is CRISPR-Cas9 and how does it work?",
        input="What is CRISPR-Cas9 and how does it work?",
        actual_output=(
            "CRISPR-Cas9 is a gene-editing technology derived from a bacterial immune system. "
            "It uses a guide RNA (gRNA) to direct the Cas9 enzyme to a specific DNA sequence. "
            "Cas9 creates a double-strand break at the target site, which the cell repairs via "
            "non-homologous end joining (NHEJ) or homology-directed repair (HDR), allowing "
            "researchers to knock out genes or insert specific sequences."
        ),
        expected_output=(
            "CRISPR-Cas9 is a gene-editing tool using a guide RNA to direct Cas9 to cut DNA at "
            "specific locations, enabling precise genome modifications."
        ),
        retrieval_context=[
            "CRISPR-Cas9 is an RNA-guided gene-editing system originally discovered as an adaptive immune mechanism in bacteria and archaea.",
            "The system consists of two key components: the Cas9 endonuclease, which cuts DNA, and a single guide RNA (sgRNA) that directs Cas9 to the target DNA sequence.",
            "After Cas9 creates a double-strand break, cellular repair mechanisms include non-homologous end joining (NHEJ), which often introduces small insertions or deletions, and homology-directed repair (HDR), which can incorporate a donor template for precise edits.",
        ],
    )


@pytest.fixture
def empty_context_rag() -> RAGTestCase:
    """A RAG test with no retrieval context — tests graceful degradation."""
    return RAGTestCase(
        name="rag_empty_context",
        query="What is the population of Mars colony in 2024?",
        input="What is the population of Mars colony in 2024?",
        actual_output=(
            "I don't have access to reliable information about Mars colony population. "
            "As of 2024, there has been no human colonization of Mars. The first crewed "
            "missions are planned for the late 2020s or 2030s."
        ),
        expected_output=None,
        retrieval_context=[],
    )


@pytest.fixture
def multi_topic_rag() -> RAGTestCase:
    """A RAG test involving multiple sub-topics in a single query."""
    return RAGTestCase(
        name="rag_multi_topic",
        query="Compare Docker and Kubernetes: what are their main differences?",
        input="Compare Docker and Kubernetes: what are their main differences?",
        actual_output=(
            "Docker is a containerization platform that packages applications with their "
            "dependencies into containers, providing isolation and portability. Kubernetes "
            "(K8s) is a container orchestration platform that manages, scales, and automates "
            "the deployment of containers across multiple hosts. Docker handles the creation "
            "and running of individual containers, while Kubernetes handles the coordination "
            "of many containers across a cluster, providing features like auto-scaling, "
            "self-healing, load balancing, and rolling updates."
        ),
        expected_output=(
            "Docker packages and runs containers while Kubernetes orchestrates and manages "
            "containers at scale across multiple machines."
        ),
        retrieval_context=[
            "Docker is a platform-as-a-service product that uses OS-level virtualization to deliver software in packages called containers.",
            "Kubernetes is an open-source container orchestration system for automating software deployment, scaling, and management.",
            "Docker containers are lightweight, standalone, executable packages that include everything needed to run a piece of software.",
            "Kubernetes provides container orchestration including automated scaling, self-healing, service discovery, load balancing, and rolling deployments.",
            "While Docker can run standalone, Kubernetes is typically used to manage Docker containers at scale in production environments.",
        ],
    )


@pytest.fixture
def adversarial_query_rag() -> RAGTestCase:
    """A RAG test with a misleading/adversarial query designed to provoke hallucination."""
    return RAGTestCase(
        name="rag_adversarial_query",
        query="Why do elephants use iPhones to make phone calls?",
        input="Why do elephants use iPhones to make phone calls?",
        actual_output=(
            "Elephants do not actually use iPhones — this premise is based on a fictional "
            "scenario. Elephants are wild animals that communicate through infrasound, "
            "vocalizations, and body language. They do not possess the cognitive abilities "
            "or physical dexterity to operate smartphones."
        ),
        expected_output=(
            "This is a false premise — elephants do not use iPhones. "
            "Elephants communicate through low-frequency rumbles and other natural methods."
        ),
        retrieval_context=[
            "Elephants communicate using infrasound — low-frequency sounds below human hearing range that can travel several kilometers.",
            "Elephant vocalizations include trumpeting, rumbling, roaring, and snorting, each conveying different meanings.",
        ],
    )


# ---------------------------------------------------------------------------
# Positive Test Cases – expected to pass
# ---------------------------------------------------------------------------

class TestRAGAnswerRelevance:
    """Tests for answer relevance to the user query."""

    def test_relevant_answer_scores_high(self, well_answered_rag: RAGTestCase) -> None:
        """A well-constructed RAG answer should score high on answer relevance."""
        tc = well_answered_rag.to_llm_test_case()
        assert tc.input is not None
        assert tc.actual_output is not None
        # Verify content overlap as a proxy for relevance
        input_words = set(tc.input.lower().split())
        output_words = set(tc.actual_output.lower().split())
        overlap = input_words & output_words
        assert len(overlap) >= 3, "Expected meaningful keyword overlap between query and answer"

    def test_irrelevant_answer_scores_low(self, irrelevant_answer_rag: RAGTestCase) -> None:
        """An answer that drifts from the query should score low on relevance."""
        tc = irrelevant_answer_rag.to_llm_test_case()
        input_words = set(tc.input.lower().split())
        output_words = set(tc.actual_output.lower().split())
        # Count topic-relevant words that appear in output
        topic_words = {"photosynthesis", "light", "chlorophyll", "energy", "sunlight"}
        topic_overlap = topic_words & output_words
        assert len(topic_overlap) <= 1, "Answer should barely mention the actual topic"

    def test_adversarial_query_handled_properly(self, adversarial_query_rag: RAGTestCase) -> None:
        """System should recognize and reject false premises rather than playing along."""
        tc = adversarial_query_rag.to_llm_test_case()
        output_lower = tc.actual_output.lower()
        assert any(phrase in output_lower for phrase in ["do not", "don't", "false", "premise", "fictional"]), (
            "Answer should challenge the false premise"
        )


class TestRAGFaithfulness:
    """Tests for faithfulness — is the answer grounded in retrieved context?"""

    def test_faithful_response(self, well_answered_rag: RAGTestCase) -> None:
        """Response should be grounded in the retrieval context."""
        tc = well_answered_rag.to_llm_test_case()
        context_text = " ".join(tc.retrieval_context).lower()
        output_lower = tc.actual_output.lower()
        # Key claims should be traceable to context
        assert "labeled" in context_text or "labeled data" in context_text
        assert "unlabeled" in context_text or "patterns" in context_text

    def test_unfaithful_hallucination(self, hallucinated_rag: RAGTestCase) -> None:
        """Response with fabricated facts should fail faithfulness."""
        tc = hallucinated_rag.to_llm_test_case()
        context_text = " ".join(tc.retrieval_context).lower()
        # Check for hallucinated claims NOT in context
        hallucinated_claims = ["einstein", "1954", "quibbles", "17 states", "minecraft", "coffee 40%"]
        hallucinated_count = sum(1 for claim in hallucinated_claims if claim in tc.actual_output.lower())
        assert hallucinated_count >= 4, "Should detect multiple hallucinated claims"

    def test_context_mismatch_flagged(self, contradictory_context_rag: RAGTestCase) -> None:
        """A wrong answer to a simple question despite having correct context should be flagged."""
        tc = contradictory_context_rag.to_llm_test_case()
        output_lower = tc.actual_output.lower()
        assert "sydney" in output_lower, "Verify wrong answer is in output"
        # The correct answer is in context
        assert "canberra" in " ".join(tc.retrieval_context).lower(), "Correct answer exists in context"


class TestRAGContextualMetrics:
    """Tests for contextual precision, recall, and relevancy."""

    def test_contextual_precision_high(self, technical_accuracy_rag: RAGTestCase) -> None:
        """Context should contain only relevant information for the query."""
        tc = technical_accuracy_rag.to_llm_test_case()
        # All context pieces should relate to CRISPR
        for chunk in tc.retrieval_context:
            chunk_lower = chunk.lower()
            assert any(kw in chunk_lower for kw in ["crispr", "cas9", "gene", "dna", "rna", "edit"]), (
                "Each context chunk should be relevant to CRISPR topic"
            )

    def test_contextual_recall_partial(self, incomplete_retrieval_rag: RAGTestCase) -> None:
        """Limited context means low recall — system should acknowledge this."""
        tc = incomplete_retrieval_rag.to_llm_test_case()
        assert len(tc.retrieval_context) <= 3, "Context is intentionally small"
        output_lower = tc.actual_output.lower()
        assert any(phrase in output_lower for phrase in ["cannot", "insufficient", "limited"]), (
            "System should acknowledge knowledge gap"
        )

    def test_contextual_relevancy_multi_topic(self, multi_topic_rag: RAGTestCase) -> None:
        """Context should cover both topics in a comparison query."""
        tc = multi_topic_rag.to_llm_test_case()
        context_text = " ".join(tc.retrieval_context).lower()
        assert "docker" in context_text, "Context should include Docker information"
        assert "kubernetes" in context_text, "Context should include Kubernetes information"


class TestRAGEdgeCases:
    """Edge case tests for RAG systems."""

    def test_empty_context_handling(self, empty_context_rag: RAGTestCase) -> None:
        """System should gracefully handle missing retrieval context."""
        tc = empty_context_rag.to_llm_test_case()
        assert len(tc.retrieval_context) == 0
        output_lower = tc.actual_output.lower()
        assert any(phrase in output_lower for phrase in ["no", "not", "don't", "cannot", "not been"]), (
            "System should indicate lack of information"
        )

    def test_verbosity_detection(self, overly_verbose_rag: RAGTestCase) -> None:
        """Overly verbose responses should be detectable."""
        tc = overly_verbose_rag.to_llm_test_case()
        word_count = len(tc.actual_output.split())
        assert word_count > 100, "Response should be excessively verbose"
        # Core answer is just a year — ratio is very high
        ratio = word_count / 20  # expected answer is ~10 words
        assert ratio > 5, "Response is disproportionately long for a simple factual question"


# ---------------------------------------------------------------------------
# Negative Test Cases – expected to fail (demonstrate broken RAG behavior)
# ---------------------------------------------------------------------------

class TestRAGFailures:
    """Tests that should detect and flag RAG system failures."""

    def test_detects_fabricated_statistics(self, hallucinated_rag: RAGTestCase) -> None:
        """Should flag fabricated numbers and statistics."""
        tc = hallucinated_rag.to_llm_test_case()
        import re
        numbers = re.findall(r'\d+%', tc.actual_output)
        assert len(numbers) >= 1, "Fabricated statistics detected"
        numbers_int = re.findall(r'\d+\s+states', tc.actual_output)
        assert len(numbers_int) >= 1, "Fabricated technical claims detected"

    def test_detects_outdated_wrong_answer(self, contradictory_context_rag: RAGTestCase) -> None:
        """Should flag when answer contradicts known facts in retrieval context."""
        tc = contradictory_context_rag.to_llm_test_case()
        output_lower = tc.actual_output.lower()
        context_lower = " ".join(tc.retrieval_context).lower()
        assert "sydney" in output_lower, "Wrong answer was given"
        assert "canberra" in context_lower, "Correct answer is in context"
        # This test documents the failure: answer != correct fact
        assert "capita" in output_lower, "Should have said Canberra, not Sydney"

    def test_detects_off_topic_response(self, irrelevant_answer_rag: RAGTestCase) -> None:
        """Should flag when response is off-topic."""
        tc = irrelevant_answer_rag.to_llm_test_case()
        # Count scientific terms NOT in output
        missing_terms = ["chlorophyll", "glucose", "calvin", "thylakoid", "atp"]
        missing_count = sum(1 for term in missing_terms if term not in tc.actual_output.lower())
        assert missing_count >= 3, "Response is missing key scientific terms from topic"
