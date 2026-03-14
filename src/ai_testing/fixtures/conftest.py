"""
Pytest Fixtures - Reusable test setup/teardown
"""

import pytest
from typing import List, Dict, Any
from .test_cases import (
    BaseTestCase,
    RAGTestCase,
    ChatbotTestCase,
    AgentTestCase,
)


@pytest.fixture
def sample_rag_test_case() -> RAGTestCase:
    """Sample RAG test case for testing."""
    return RAGTestCase(
        name="sample_rag",
        query="What is artificial intelligence?",
        input="What is artificial intelligence?",
        actual_output="AI stands for Artificial Intelligence, which enables machines to learn.",
        expected_output="Artificial Intelligence (AI) is a field of computer science...",
        retrieval_context=[
            "Artificial Intelligence is intelligence demonstrated by machines.",
            "AI enables computers to learn from experience."
        ]
    )


@pytest.fixture
def sample_chatbot_test_case() -> ChatbotTestCase:
    """Sample chatbot test case for testing."""
    return ChatbotTestCase(
        name="sample_chatbot",
        input="Hello, I need help with my order",
        actual_output="Hello! I'd be happy to help you with your order.",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"}
        ]
    )


@pytest.fixture
def sample_agent_test_case() -> AgentTestCase:
    """Sample agent test case for testing."""
    return AgentTestCase(
        name="sample_agent",
        input="Book a flight from NYC to London",
        actual_output="I've booked flight AA123 from NYC to London.",
        tools_called=[
            {"name": "search_flights", "args": {"from": "NYC", "to": "London"}},
            {"name": "book_flight", "args": {"flight_id": "AA123"}}
        ],
        steps_taken=2,
        goal_achieved=True
    )


@pytest.fixture
def test_runner():
    """Sample test runner configuration."""
    from ..runner import TestRunner
    return TestRunner(model="gpt-4", threshold=0.5)


@pytest.fixture
def sample_rag_queries() -> List[str]:
    """Sample RAG queries for batch testing."""
    return [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "Define supervised learning",
    ]


@pytest.fixture
def sample_retrieval_contexts() -> List[List[str]]:
    """Sample retrieval contexts for batch testing."""
    return [
        ["Machine learning is a subset of AI...", "ML algorithms learn from data..."],
        ["Neural networks are inspired by biological neurons...", "They consist of layers..."],
        ["Deep learning uses neural networks with multiple layers...", "It powers modern AI..."],
        ["Supervised learning uses labeled data...", "The model learns from examples..."],
    ]


@pytest.fixture
def sample_test_data() -> Dict[str, Any]:
    """General sample test data."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "threshold": 0.5,
    }


__all__ = [
    "sample_rag_test_case",
    "sample_chatbot_test_case", 
    "sample_agent_test_case",
    "test_runner",
    "sample_rag_queries",
    "sample_retrieval_contexts",
    "sample_test_data",
]
