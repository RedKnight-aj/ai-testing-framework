"""
Test Cases - Reusable test objects (Page Object Pattern)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from deepeval.test_case import LLMTestCase, LLMTestCaseParams


@dataclass
class BaseTestCase:
    """
    Base test case - similar to Page Object in Playwright.
    
    Provides reusable test case structure for AI evaluation.
    """
    
    name: str
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_llm_test_case(self) -> LLMTestCase:
        """
        Convert to DeepEval LLMTestCase.
        
        Returns:
            LLMTestCase for DeepEval evaluation
        """
        return LLMTestCase(
            input=self.input,
            actual_output=self.actual_output,
            expected_output=self.expected_output,
            retrieval_context=self.retrieval_context,
            context=self.context,
        )
    
    @property
    def params(self) -> List[LLMTestCaseParams]:
        """Get available parameters for GEval."""
        params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        if self.expected_output:
            params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
        if self.retrieval_context:
            params.append(LLMTestCaseParams.RETRIEVAL_CONTEXT)
        if self.context:
            params.append(LLMTestCaseParams.CONTEXT)
        return params


@dataclass
class RAGTestCase(BaseTestCase):
    """
    Test case for RAG (Retrieval-Augmented Generation) evaluation.
    
    Usage:
        test_case = RAGTestCase(
            name="test_rag_1",
            input="What is machine learning?",
            actual_output="Machine learning is...",
            expected_output="Machine learning is a type of AI...",
            retrieval_context=["ML is a subset of AI...", "ML enables computers to learn..."]
        )
    """
    
    query: str = ""
    retrieved_chunks: List[str] = field(default_factory=list)
    generation: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = "rag_test"
        if self.retrieval_context is None:
            self.retrieval_context = []
    
    def to_llm_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.input or self.query,
            actual_output=self.actual_output or self.generation,
            expected_output=self.expected_output,
            retrieval_context=self.retrieval_context or self.retrieved_chunks,
        )


@dataclass
class ChatbotTestCase(BaseTestCase):
    """
    Test case for Chatbot/Multi-turn conversation evaluation.
    
    Usage:
        test_case = ChatbotTestCase(
            name="chatbot_support",
            input="Hello, I need help",
            actual_output="Hello! How can I help you today?",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            expected_output="Hello! I'd be happy to help...",
        )
    """
    
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    turn_number: int = 0
    
    def __post_init__(self):
        if not self.name:
            self.name = "chatbot_test"
        if self.context is None:
            self.context = self._format_history()
    
    def _format_history(self) -> str:
        """Format conversation history as context."""
        if not self.conversation_history:
            return ""
        return "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history
        )
    
    def add_turn(self, role: str, content: str):
        """Add a turn to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        self.context = self._format_history()
        self.turn_number += 1


@dataclass
class AgentTestCase(BaseTestCase):
    """
    Test case for AI Agent evaluation.
    
    Usage:
        test_case = AgentTestCase(
            name="agent_booking",
            input="Book a flight to NYC",
            actual_output="I've booked your flight...",
            tools_called=[
                {"name": "search_flights", "args": {"destination": "NYC"}},
                {"name": "book_flight", "args": {"flight_id": "123"}}
            ],
            steps_taken=3,
            goal_achieved=True,
        )
    """
    
    tools_called: List[Dict[str, Any]] = field(default_factory=list)
    steps_taken: int = 0
    goal_achieved: bool = False
    tool_outputs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name:
            self.name = "agent_test"
    
    def add_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Add a tool call to the test case."""
        self.tools_called.append({"name": tool_name, "args": args})
        self.steps_taken += 1


@dataclass
class SummarizationTestCase(BaseTestCase):
    """
    Test case for Summarization evaluation.
    
    Usage:
        test_case = SummarizationTestCase(
            name="summarize_article",
            input="Long article text...",
            actual_output="Summary of the article...",
            expected_output="Brief summary...",
        )
    """
    
    source_text: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = "summarization_test"
        if not self.expected_output and self.source_text:
            # For summarization, context is the source
            self.retrieval_context = [self.source_text]


@dataclass
class CodeTestCase(BaseTestCase):
    """
    Test case for Code generation/understanding evaluation.
    
    Usage:
        test_case = CodeTestCase(
            name="code_generation",
            input="Write a function to sort a list",
            actual_output="def sort_list(lst): return sorted(lst)",
            expected_output="def sort_list(lst): ...",
            language="python",
        )
    """
    
    language: str = "python"
    code_output: str = ""
    expected_code: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = "code_test"
        if self.code_output:
            self.actual_output = self.code_output
        if self.expected_code:
            self.expected_output = self.expected_code


# Convenience factory functions
def create_rag_test(
    query: str,
    response: str,
    expected: str,
    context: List[str],
    name: str = "rag_test"
) -> RAGTestCase:
    """Factory function to create RAG test case."""
    return RAGTestCase(
        name=name,
        query=query,
        input=query,
        actual_output=response,
        expected_output=expected,
        retrieval_context=context,
    )


def create_chatbot_test(
    user_input: str,
    bot_response: str,
    history: List[Dict[str, str]] = None,
    name: str = "chatbot_test"
) -> ChatbotTestCase:
    """Factory function to create chatbot test case."""
    return ChatbotTestCase(
        name=name,
        input=user_input,
        actual_output=bot_response,
        conversation_history=history or [],
    )


__all__ = [
    "BaseTestCase",
    "RAGTestCase", 
    "ChatbotTestCase",
    "AgentTestCase",
    "SummarizationTestCase",
    "CodeTestCase",
    "create_rag_test",
    "create_chatbot_test",
]
