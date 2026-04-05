"""
AI Testing Framework v2.0.0
Production-grade LLM evaluation framework using DeepEval CLI.

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

from __future__ import annotations
from typing import TYPE_CHECKING

__version__ = "2.0.0"

# ------------------------------------------------------------------
# Core execution — runs DeepEval via CLI (not direct imports)
# ------------------------------------------------------------------
from .engine import (
    DeepEvalEngine,
    DeepEvalRunResult,
    MetricScore,
    TestCaseResult,
)
from .runner import TestRunner, EvaluationResult

# ------------------------------------------------------------------
# Quality gates
# ------------------------------------------------------------------
from .gates import QualityGate, QualityGateResult

# ------------------------------------------------------------------
# Optional sub-modules (may not exist in minimal installs)
# ------------------------------------------------------------------
if TYPE_CHECKING:
    from typing import List, Dict, Any, Optional, Union

__all__: list[str] = [
    # Version
    "__version__",

    # Engine (DeepEval CLI wrapper)
    "DeepEvalEngine",
    "DeepEvalRunResult",
    "MetricScore",
    "TestCaseResult",

    # Core functionality
    "TestRunner",
    "EvaluationResult",
    "QualityGate",
    "QualityGateResult",
]


def __getattr__(name: str):
    """Lazy-import optional sub-modules to avoid import errors.

    The framework core (engine + runner) no longer imports DeepEval
    directly, so this module is always loadable.  Extended modules
    (benchmark, test_cases, metrics, reporters, config, cli, utils)
    are loaded on-demand.
    """
    lazy_map = {
        "benchmark":   (".benchmark",   ["BenchmarkRunner", "BenchmarkResult"]),
        "cli":         (".cli",         ["main"]),
        "test_cases":  (".test_cases",  [
            "BaseTestCase", "RAGTestCase", "ChatbotTestCase",
            "AgentTestCase", "SummarizationTestCase", "CodeTestCase",
            "create_rag_test", "create_chatbot_test",
        ]),
        "metrics":     (".metrics", [
            "AnswerRelevancyMetric", "FaithfulnessMetric",
            "ContextualPrecisionMetric", "ContextualRecallMetric",
            "ContextualRelevancyMetric", "SummarizationMetric",
            "ToxicityMetric", "BiasMetric", "HallucinationMetric",
            "GEvalMetric", "TaskCompletionMetric", "ToolCorrectnessMetric",
            "GoalAccuracyMetric", "StepEfficiencyMetric", "create_custom_metric",
        ]),
        "reporters":   (".reporters", [
            "BaseReporter", "JSONReporter", "HTMLReporter",
            "MarkdownReporter", "SlackReporter", "ConsoleReporter",
        ]),
        "config":      (".config", ["Settings", "default_settings"]),
        "utils":       (".utils", ["load_benchmark_dataset", "load_quality_gate_config"]),
    }

    for mod_modpath, mod_names in lazy_map.values():
        if name in mod_names:
            import importlib
            mod = importlib.import_module(mod_modpath, __package__)
            value = getattr(mod, name)
            globals()[name] = value
            import sys
            sys.modules.get(__name__).__dict__[name] = value  # type: ignore[union-attr]
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """All exported names (including lazy) for IDE autocomplete."""
    return list(__all__) + ["BenchmarkRunner", "BenchmarkResult"]