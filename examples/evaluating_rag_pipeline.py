"""
Evaluating RAG Pipelines
========================
Complete example of evaluating a Retrieval-Augmented Generation (RAG) pipeline.
"""

from typing import List, Dict, Any
from ai_testing.test_cases import RAGTestCase, create_rag_test
from ai_testing.runner import TestRunner
from ai_testing.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionWrapper,
    ContextualRecallWrapper,
    HallucinationMetricWrapper,
)

class MockRAGSystem:
    """Mock RAG system for demonstration."""

    def __init__(self):
        # Mock knowledge base
        self.knowledge_base = {
            "machine_learning": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "Supervised learning uses labeled training data, while unsupervised learning finds patterns in unlabeled data.",
                "Deep learning uses neural networks with multiple layers to process complex data patterns.",
                "Common ML algorithms include decision trees, random forests, support vector machines, and neural networks."
            ],
            "climate_change": [
                "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
                "Greenhouse gases like CO2, methane, and nitrous oxide trap heat in the atmosphere, causing global warming.",
                "Effects include rising sea levels, more extreme weather events, biodiversity loss, and impacts on agriculture.",
                "Mitigation strategies include reducing emissions, transitioning to renewable energy, and improving energy efficiency."
            ],
            "quantum_computing": [
                "Quantum computing uses quantum bits (qubits) that can exist in superposition and entanglement.",
                "Qubits can represent both 0 and 1 simultaneously, enabling parallel computation.",
                "Quantum algorithms like Shor's algorithm can factor large numbers exponentially faster than classical computers.",
                "Current challenges include qubit stability, error correction, and scaling to practical quantum computers."
            ]
        }

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Mock retrieval function."""
        query_lower = query.lower()

        # Simple keyword matching for demonstration
        if "machine learning" in query_lower or "ml" in query_lower:
            return self.knowledge_base["machine_learning"][:top_k]
        elif "climate" in query_lower or "global warming" in query_lower:
            return self.knowledge_base["climate_change"][:top_k]
        elif "quantum" in query_lower:
            return self.knowledge_base["quantum_computing"][:top_k]
        else:
            return ["General knowledge: This topic requires specific domain expertise."][:top_k]

    def generate_response(self, query: str, context: List[str]) -> str:
        """Mock response generation."""
        context_text = " ".join(context)

        # Simple template-based generation (in real RAG, this would be an LLM)
        if "machine learning" in query.lower():
            return f"Based on the retrieved information: {context_text[:200]}... Machine learning enables computers to learn patterns from data through various algorithms and techniques."
        elif "climate" in query.lower():
            return f"Climate change information: {context_text[:200]}... This leads to significant environmental and societal impacts that require urgent mitigation efforts."
        elif "quantum" in query.lower():
            return f"Quantum computing explanation: {context_text[:200]}... This revolutionary technology has the potential to solve complex problems beyond classical computing capabilities."
        else:
            return f"General response: {context_text[:100]}... This topic covers important concepts in modern technology and science."

def evaluate_rag_pipeline():
    """Evaluate a complete RAG pipeline."""

    # Initialize RAG system and test runner
    rag_system = MockRAGSystem()
    runner = TestRunner(
        model="gpt-4",
        threshold=0.7
    )

    # Test cases for different domains
    test_queries = [
        {
            "query": "What is machine learning and how does it work?",
            "expected": "Machine learning is a subset of AI that enables computers to learn from data. It works through algorithms that identify patterns in data, using techniques like supervised and unsupervised learning."
        },
        {
            "query": "How does climate change affect the environment?",
            "expected": "Climate change causes global warming through greenhouse gases, leading to rising sea levels, extreme weather, biodiversity loss, and agricultural impacts."
        },
        {
            "query": "What are the advantages of quantum computing?",
            "expected": "Quantum computing can solve certain problems exponentially faster than classical computers through superposition and entanglement of qubits."
        },
        {
            "query": "Explain the concept of artificial intelligence.",
            "expected": "This falls back to general knowledge as the query doesn't match specific domains."
        }
    ]

    results = []

    print("RAG Pipeline Evaluation Results:")
    print("=" * 60)

    for i, test_case in enumerate(test_queries, 1):
        # Retrieve relevant context
        retrieved_context = rag_system.retrieve(test_case["query"])

        # Generate response using retrieved context
        generated_response = rag_system.generate_response(test_case["query"], retrieved_context)

        # Create RAG test case
        rag_test = RAGTestCase(
            name=f"rag_test_{i}",
            query=test_case["query"],
            input=test_case["query"],
            actual_output=generated_response,
            expected_output=test_case["expected"],
            retrieval_context=retrieved_context
        )

        # Evaluate with RAG-specific metrics
        evaluation_result = runner.evaluate(
            test_case=rag_test,
            metrics=[
                "answer_relevancy",
                "faithfulness",
                "contextual_precision",
                "contextual_recall",
                "hallucination"
            ]
        )

        results.append(evaluation_result)

        # Print detailed results
        print(f"\\nTest {i}: {test_case['query'][:50]}...")
        print(f"Retrieved {len(retrieved_context)} context chunks")
        print(".2f")
        print(f"Passed: {evaluation_result.passed}")
        print(f"Relevancy: {evaluation_result.metrics.get('answer_relevancy', 0):.2f}")
        print(f"Faithfulness: {evaluation_result.metrics.get('faithfulness', 0):.2f}")
        print(f"Context Precision: {evaluation_result.metrics.get('contextual_precision', 0):.2f}")
        print(f"Context Recall: {evaluation_result.metrics.get('contextual_recall', 0):.2f}")
        print(f"Hallucination: {evaluation_result.metrics.get('hallucination', 0):.2f}")

        # Show sample of retrieved context and response
        print(f"Sample context: {retrieved_context[0][:80]}...")
        print(f"Response: {generated_response[:80]}...")

    # Overall summary
    print("\\n" + "=" * 60)
    print("OVERALL SUMMARY:")
    print(f"Total tests: {len(results)}")
    passed = sum(1 for r in results if r.passed)
    print(f"Passed: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

    avg_score = sum(r.score for r in results) / len(results)
    print(".2f")

    # Average scores for key metrics
    metrics_to_average = ["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall"]
    for metric in metrics_to_average:
        scores = [r.metrics.get(metric, 0) for r in results]
        avg_metric = sum(scores) / len(scores)
        print(".2f")

    return results

def evaluate_rag_components_separately():
    """Evaluate retrieval and generation components separately."""

    rag_system = MockRAGSystem()
    runner = TestRunner(model="gpt-4", threshold=0.7)

    test_query = "What is machine learning?"

    # Evaluate retrieval quality
    retrieved_context = rag_system.retrieve(test_query)
    generated_response = rag_system.generate_response(test_query, retrieved_context)

    # Test retrieval quality (context relevance)
    retrieval_test = RAGTestCase(
        name="retrieval_quality_test",
        query=test_query,
        input=test_query,
        actual_output=" ".join(retrieved_context),  # Treat retrieved context as "output"
        expected_output="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        retrieval_context=retrieved_context
    )

    # Test generation quality (response given context)
    generation_test = RAGTestCase(
        name="generation_quality_test",
        query=test_query,
        input=test_query,
        actual_output=generated_response,
        expected_output="Machine learning enables computers to learn from data through algorithms that identify patterns and make predictions.",
        retrieval_context=retrieved_context
    )

    print("Component-Level RAG Evaluation:")
    print("-" * 40)

    # Evaluate retrieval
    retrieval_result = runner.evaluate(
        retrieval_test,
        ["contextual_precision", "contextual_relevancy"]
    )
    print(f"Retrieval Quality: {retrieval_result.score:.2f}")
    print(f"  Precision: {retrieval_result.metrics.get('contextual_precision', 0):.2f}")
    print(f"  Relevancy: {retrieval_result.metrics.get('contextual_relevancy', 0):.2f}")

    # Evaluate generation
    generation_result = runner.evaluate(
        generation_test,
        ["answer_relevancy", "faithfulness", "hallucination"]
    )
    print(f"Generation Quality: {generation_result.score:.2f}")
    print(f"  Relevancy: {generation_result.metrics.get('answer_relevancy', 0):.2f}")
    print(f"  Faithfulness: {generation_result.metrics.get('faithfulness', 0):.2f}")
    print(f"  Hallucination: {generation_result.metrics.get('hallucination', 0):.2f}")

    return retrieval_result, generation_result

if __name__ == "__main__":
    # Full pipeline evaluation
    evaluate_rag_pipeline()

    # Component-level evaluation
    evaluate_rag_components_separately()