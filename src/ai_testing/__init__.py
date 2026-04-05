"""
AI Testing Framework v2.0.0
Production-grade LLM evaluation framework using DeepEval

This framework provides comprehensive testing capabilities for AI applications including:
- RAG (Retrieval-Augmented Generation) evaluation
- Chatbot/Conversational AI assessment
- AI Agent tool use and goal achievement
- Quality gates and compliance checks
- Benchmark datasets and metrics

Example:
    >>> from ai_testing import TestRunner, RAGTestCase
    >>> runner = TestRunner()
    >>> test_case = RAGTestCase(
    ...     name="rag_eval",
    ...     input="What is machine learning?",
    ...     actual_output="ML is a subset of AI...",
    ...     retrieval_context=["ML definition...", "AI context..."]
    ... )
    >>> result = runner.evaluate(test_case, ["answer_relevancy", "faithfulness"])
"""

from typing import TYPE_CHECKING

__version__ = "2.0.0"

# Core classes
from .runner import TestRunner, EvaluationResult
from .benchmark import BenchmarkRunner, BenchmarkResult
from .gates import QualityGate, QualityGateResult
from .cli import main as cli_main

# Test case classes
from .test_cases import (
    BaseTestCase,
    RAGTestCase,
    ChatbotTestCase,
    AgentTestCase,
    SummarizationTestCase,
    CodeTestCase,
    create_rag_test,
    create_chatbot_test,
)

# Metrics
from .metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    SummarizationMetric,
    ToxicityMetric,
    BiasMetric,
    HallucinationMetric,
    GEvalMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    GoalAccuracyMetric,
    StepEfficiencyMetric,
    create_custom_metric,
)

# Reporters
from .reporters import (
    BaseReporter,
    JSONReporter,
    HTMLReporter,
    MarkdownReporter,
    SlackReporter,
    ConsoleReporter,
)

# Configuration
from .config import Settings, default_settings

# Utilities
from .utils import load_benchmark_dataset, load_quality_gate_config

if TYPE_CHECKING:
    from typing import List, Dict, Any, Optional, Union

__all__ = [
    # Version
    "__version__",

    # Core functionality
    "TestRunner",
    "EvaluationResult",
    "BenchmarkRunner",
    "BenchmarkResult",
    "QualityGate",
    "QualityGateResult",
    "cli_main",

    # Test cases
    "BaseTestCase",
    "RAGTestCase",
    "ChatbotTestCase",
    "AgentTestCase",
    "SummarizationTestCase",
    "CodeTestCase",
    "create_rag_test",
    "create_chatbot_test",

    # Metrics
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    "SummarizationMetric",
    "ToxicityMetric",
    "BiasMetric",
    "HallucinationMetric",
    "GEvalMetric",
    "TaskCompletionMetric",
    "ToolCorrectnessMetric",
    "GoalAccuracyMetric",
    "StepEfficiencyMetric",
    "create_custom_metric",

    # Reporters
    "BaseReporter",
    "JSONReporter",
    "HTMLReporter",
    "MarkdownReporter",
    "SlackReporter",
    "ConsoleReporter",

    # Configuration
    "Settings",
    "default_settings",

    # Utilities
    "load_benchmark_dataset",
    "load_quality_gate_config",
]