"""
Metrics - Wrapper around DeepEval metrics
"""

from typing import Optional, List
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    SummarizationMetric,
    GEval,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    GoalAccuracyMetric,
    StepEfficiencyMetric,
)
from deepeval.metrics.base import BaseMetric
from deepeval.test_case import LLMTestCaseParams


# RAG Metrics
class AnswerRelevancyMetricWrapper(AnswerRelevancyMetric):
    """Wrapper for Answer Relevancy metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class FaithfulnessMetricWrapper(FaithfulnessMetric):
    """Wrapper for Faithfulness metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class ContextualPrecisionWrapper(ContextualPrecisionMetric):
    """Wrapper for Contextual Precision metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class ContextualRecallWrapper(ContextualRecallMetric):
    """Wrapper for Contextual Recall metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


# Agent Metrics
class TaskCompletionMetricWrapper(TaskCompletionMetric):
    """Wrapper for Task Completion metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class ToolCorrectnessMetricWrapper(ToolCorrectnessMetric):
    """Wrapper for Tool Correctness metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class GoalAccuracyMetricWrapper(GoalAccuracyMetric):
    """Wrapper for Goal Accuracy metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


# Quality Metrics
class HallucinationMetricWrapper(HallucinationMetric):
    """Wrapper for Hallucination metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class ToxicityMetricWrapper(ToxicityMetric):
    """Wrapper for Toxicity metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class BiasMetricWrapper(BiasMetric):
    """Wrapper for Bias metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


class SummarizationMetricWrapper(SummarizationMetric):
    """Wrapper for Summarization metric."""
    
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


# Custom GEval
def create_custom_metric(
    name: str,
    criteria: str,
    evaluation_params: List[LLMTestCaseParams],
    threshold: float = 0.5,
    **kwargs
) -> GEval:
    """
    Create a custom GEval metric.
    
    Args:
        name: Metric name
        criteria: Evaluation criteria
        evaluation_params: Parameters to evaluate
        threshold: Pass threshold
        **kwargs: Additional parameters
        
    Returns:
        GEval metric instance
    """
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=evaluation_params,
        threshold=threshold,
        **kwargs
    )


# Convenience imports
AnswerRelevancyMetric = AnswerRelevancyMetricWrapper
FaithfulnessMetric = FaithfulnessMetricWrapper
HallucinationMetric = HallucinationMetricWrapper
ToxicityMetric = ToxicityMetricWrapper
BiasMetric = BiasMetricWrapper
GEvalMetric = GEval


__all__ = [
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionWrapper",
    "ContextualRecallWrapper",
    "TaskCompletionMetricWrapper",
    "ToolCorrectnessMetricWrapper",
    "GoalAccuracyMetricWrapper",
    "HallucinationMetric",
    "ToxicityMetric",
    "BiasMetric",
    "SummarizationMetricWrapper",
    "GEvalMetric",
    "create_custom_metric",
]
