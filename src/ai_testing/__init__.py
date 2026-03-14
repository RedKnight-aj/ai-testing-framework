"""
AI Testing Framework
Production-ready LLM evaluation framework using DeepEval
"""

from .runner import TestRunner, EvaluationResult
from .test_cases import BaseTestCase, RAGTestCase, ChatbotTestCase, AgentTestCase
from .metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    GEvalMetric,
)
from .reporters import JSONReporter, HTMLReporter, SlackReporter
from .config import Settings

__version__ = "1.0.0"

__all__ = [
    "TestRunner",
    "EvaluationResult",
    "BaseTestCase",
    "RAGTestCase",
    "ChatbotTestCase",
    "AgentTestCase",
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "HallucinationMetric",
    "ToxicityMetric",
    "BiasMetric",
    "GEvalMetric",
    "JSONReporter",
    "HTMLReporter",
    "SlackReporter",
    "Settings",
]
