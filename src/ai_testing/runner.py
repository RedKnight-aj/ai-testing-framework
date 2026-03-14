"""
Test Runner - Core execution engine
Follows DeepEval's official patterns
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json

from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
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
)
from deepeval.metrics.base import BaseMetric


@dataclass
class EvaluationResult:
    """Result of a test evaluation."""
    
    test_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    passed: bool = False
    threshold: float = 0.5
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def score(self) -> float:
        """Average score across all metrics."""
        if not self.metrics:
            return 0.0
        return sum(self.metrics.values()) / len(self.metrics)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "metrics": self.metrics,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "details": self.details,
        }


class TestRunner:
    """
    Main test runner following DeepEval's pytest-style patterns.
    
    Usage:
        runner = TestRunner(model="gpt-4", threshold=0.7)
        result = runner.evaluate(test_case, metrics=["answer_relevancy", "faithfulness"])
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        cloud_enabled: bool = False,
        **kwargs
    ):
        """
        Initialize the test runner.
        
        Args:
            model: Model to use for evaluation
            threshold: Minimum score to pass
            cloud_enabled: Use Confident AI cloud
            **kwargs: Additional DeepEval parameters
        """
        self.model = model
        self.threshold = threshold
        self.cloud_enabled = cloud_enabled
        self.kwargs = kwargs
    
    def _get_metric(self, metric_name: str, **kwargs) -> BaseMetric:
        """Get metric instance by name."""
        metrics_map = {
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": FaithfulnessMetric,
            "hallucination": HallucinationMetric,
            "toxicity": ToxicityMetric,
            "bias": BiasMetric,
            "summarization": SummarizationMetric,
            "contextual_precision": ContextualPrecisionMetric,
            "contextual_recall": ContextualRecallMetric,
            "contextual_relevancy": ContextualRelevancyMetric,
        }
        
        if metric_name not in metrics_map:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric_class = metrics_map[metric_name]
        return metric_class(threshold=self.threshold, **kwargs)
    
    def evaluate(
        self,
        test_case: "BaseTestCase",
        metrics: List[str],
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a test case.
        
        Args:
            test_case: The test case to evaluate
            metrics: List of metric names to use
            **kwargs: Additional parameters
            
        Returns:
            EvaluationResult with scores
        """
        # Convert to DeepEval LLMTestCase
        llm_test_case = test_case.to_llm_test_case()
        
        # Get metrics
        eval_metrics = [self._get_metric(m, **kwargs) for m in metrics]
        
        # Run evaluation
        try:
            assert_test(llm_test_case, eval_metrics)
            passed = True
        except AssertionError:
            passed = False
        
        # Collect scores
        metric_scores = {}
        for metric in eval_metrics:
            metric_scores[metric.name] = metric.score
        
        return EvaluationResult(
            test_name=test_case.name,
            metrics=metric_scores,
            passed=passed,
            threshold=self.threshold,
            details={"model": self.model},
        )
    
    def evaluate_batch(
        self,
        test_cases: List["BaseTestCase"],
        metrics: List[str],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases
            metrics: List of metric names
            
        Returns:
            List of evaluation results
        """
        results = []
        for test_case in test_cases:
            result = self.evaluate(test_case, metrics)
            results.append(result)
        return results
    
    def evaluate_with_custom_criteria(
        self,
        test_case: "BaseTestCase",
        criteria: str,
        evaluation_params: List[LLMTestCaseParams],
    ) -> EvaluationResult:
        """
        Evaluate with custom GEval criteria.
        
        Args:
            test_case: Test case to evaluate
            criteria: Custom evaluation criteria
            evaluation_params: Parameters to evaluate
            
        Returns:
            EvaluationResult
        """
        metric = GEval(
            name="Custom",
            criteria=criteria,
            evaluation_params=evaluation_params,
            threshold=self.threshold,
        )
        
        llm_test_case = test_case.to_llm_test_case()
        
        try:
            assert_test(llm_test_case, [metric])
            passed = True
        except AssertionError:
            passed = False
        
        return EvaluationResult(
            test_name=test_case.name,
            metrics={"custom": metric.score},
            passed=passed,
            threshold=self.threshold,
        )


__all__ = ["TestRunner", "EvaluationResult"]
