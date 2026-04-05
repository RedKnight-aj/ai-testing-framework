"""
Test Runner — Core execution engine via DeepEval CLI.

This module orchestrates LLM evaluations by delegating the heavy lifting
to the DeepEval CLI (``deepeval test run``).  The runner is the *brain*:
it generates test configs, invokes the engine, parses results, and applies
quality gates.

Usage::

    runner = TestRunner(model="gpt-4", threshold=0.7)
    result = runner.evaluate(test_case, metrics=["answer_relevancy", "faithfulness"])
"""

from __future__ import annotations

import os
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .engine import (
    DeepEvalEngine,
    DeepEvalRunResult,
    MetricScore,
    TestCaseResult,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of a single test evaluation."""

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "metrics": self.metrics,
            "passed": self.passed,
            "score": self.score,
            "threshold": self.threshold,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Main test runner that delegates evaluation to the DeepEval CLI engine.

    Typical workflow::

        runner = TestRunner(model="gpt-4", threshold=0.7)
        # Use a test case object (from test_cases module)
        result = runner.evaluate(my_test_case, metrics=["answer_relevancy"])

    Args:
        model: Model identifier for metadata (passed to engine config).
        threshold: Default minimum score to pass (0-1).
        cloud_enabled: Whether to enable DeepEval Confident AI cloud reporting.
        timeout: Maximum seconds for a single evaluation run.
        engine: Optional pre-configured :class:`DeepEvalEngine``.
    """

    KNOWN_METRICS = [
        "answer_relevancy",
        "faithfulness",
        "hallucination",
        "toxicity",
        "bias",
        "summarization",
        "contextual_precision",
        "contextual_recall",
        "contextual_relevancy",
    ]

    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        cloud_enabled: bool = False,
        timeout: int = 600,
        engine: Optional[DeepEvalEngine] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.cloud_enabled = cloud_enabled
        self.timeout = timeout
        self.extra_kwargs = kwargs
        self._engine = engine or DeepEvalEngine(timeout=timeout)

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def check_dependencies(self) -> tuple[bool, str]:
        """Check whether the DeepEval CLI engine is available.

        Returns:
            ``(True, message)`` if available, ``(False, error_message)`` otherwise.
        """
        return self._engine.check_dependencies()

    # ------------------------------------------------------------------
    # Public: evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_case: "BaseTestCase",
        metrics: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a single test case via the DeepEval CLI.

        Args:
            test_case: Test case object providing input/output/context.
            metrics: Metric names to evaluate.
                     Defaults to ``["answer_relevancy"]``.
            **kwargs: Additional metric configuration.

        Returns:
            :class:`EvaluationResult` with scores and pass/fail status.

        Raises:
            RuntimeError: If the DeepEval CLI is not available.
        """
        metrics = metrics or ["answer_relevancy"]
        threshold = kwargs.pop("threshold", self.threshold)
        llm_tc = test_case.to_llm_test_case()

        result = self._engine.run_test(
            test_name=test_case.name,
            input_text=llm_tc.get("input", ""),
            actual_output=llm_tc.get("actual_output", ""),
            expected_output=llm_tc.get("expected_output", ""),
            context=llm_tc.get("context", []),
            retrieval_context=llm_tc.get("retrieval_context", []),
            metrics=metrics,
            threshold=threshold,
        )

        return self._cli_result_to_evaluation(result)

    def evaluate_batch(
        self,
        test_cases: List["BaseTestCase"],
        metrics: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple test cases by generating a benchmark file.

        Args:
            test_cases: List of test cases.
            metrics: Metric names for all cases.

        Returns:
            List of :class:`EvaluationResult` objects.
        """
        metrics = metrics or ["answer_relevancy"]
        threshold = self.threshold

        # Build dataset from test cases
        dataset = []
        for tc in test_cases:
            llm_tc = tc.to_llm_test_case()
            dataset.append({
                "name": tc.name,
                "input": llm_tc.get("input", ""),
                "actual_output": llm_tc.get("actual_output", ""),
                "expected_output": llm_tc.get("expected_output"),
                "context": llm_tc.get("context", []),
                "retrieval_context": llm_tc.get("retrieval_context", []),
            })

        # Write temp dataset
        fd, ds_path = tempfile.mkstemp(suffix=".json", prefix="deepeval_batch_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        try:
            result = self._engine.run_benchmark(
                dataset_path=ds_path,
                metrics=metrics,
                threshold=threshold,
            )
        finally:
            try:
                os.unlink(ds_path)
            except OSError:
                pass

        # Map results back to evaluation results
        if not result.results:
            return [EvaluationResult(test_name=tc.name) for tc in test_cases]

        eval_results: List[EvaluationResult] = []
        for i, tc_result in enumerate(result.results):
            name = tc_result.name
            if i < len(test_cases):
                name = test_cases[i].name
            eval_results.append(
                EvaluationResult(
                    test_name=name,
                    metrics={m.name: m.score for m in tc_result.metrics},
                    passed=tc_result.passed,
                    threshold=threshold,
                    details={"model": self.model, "error": tc_result.error},
                )
            )
        return eval_results

    def evaluate_with_custom_criteria(
        self,
        test_case: "BaseTestCase",
        criteria: str,
        evaluation_params: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate using custom GEval criteria via the CLI engine.

        Args:
            test_case: Test case to evaluate.
            criteria: Human-readable evaluation criteria string.
            evaluation_params: Parameter names to evaluate against.

        Returns:
            :class:`EvaluationResult`.
        """
        llm_tc = test_case.to_llm_test_case()

        # Generate a custom test file with GEval
        test_content = (
            "from deepeval.test_case import LLMTestCase\n"
            "from deepeval.metrics import GEval\n\n\n"
            "def test_custom_geval():\n"
            f'    criteria = """{criteria}"""\n'
            "    metric = GEval(\n"
            '        name="Custom",'
            "        criteria=criteria,\n"
            f"        evaluation_params={evaluation_params or []},\n"
            f"        threshold={self.threshold},\n"
            "    )\n"
            f'    tc = LLMTestCase(\n'
            f'        input={json.dumps(llm_tc.get("input", ""))},\n'
            f'        actual_output={json.dumps(llm_tc.get("actual_output", ""))},\n'
            f'        expected_output={json.dumps(llm_tc.get("expected_output"))},\n'
            f'        context={json.dumps(llm_tc.get("context", []))},\n'
            f'    )\n'
            "    metric.measure(tc)\n"
            "    assert metric.is_successful(), f'Custom GEval failed: {{metric.score}}'\n"
        )

        fd, tf_path = tempfile.mkstemp(suffix="_custom.py", prefix="deepeval_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(test_content)

        try:
            result = self._engine.run(tf_path)
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass

        return self._cli_result_to_evaluation(result)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        dataset_path: str | Path,
        *,
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Run a full benchmark evaluation against a dataset file.

        Args:
            dataset_path: Path to JSON dataset file.
            metrics: List of metric names.

        Returns:
            Aggregate :class:`EvaluationResult` for the entire benchmark.
        """
        metrics = metrics or ["answer_relevancy"]

        result = self._engine.run_benchmark(
            dataset_path=str(dataset_path),
            metrics=metrics,
            threshold=self.threshold,
        )

        # Aggregate all results into one
        all_metrics: Dict[str, List[float]] = {}
        total_passed = 0
        total_tests = 0

        for r in result.results:
            total_tests += 1
            if r.passed:
                total_passed += 1
            for m in r.metrics:
                all_metrics.setdefault(m.name, []).append(m.score)

        avg_metrics = {}
        for m_name, scores in all_metrics.items():
            avg_metrics[m_name] = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            test_name=f"benchmark:{dataset_path}",
            metrics=avg_metrics,
            passed=result.success,
            threshold=self.threshold,
            details={
                "model": self.model,
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "pass_rate": total_passed / total_tests if total_tests else 0.0,
                "raw": result.to_dict(),
            },
        )

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_last_result(self) -> Optional[DeepEvalRunResult]:
        """Return the raw engine result from the most recent run."""
        return self._engine.get_results()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cli_result_to_evaluation(cli_result: DeepEvalRunResult) -> EvaluationResult:
        """Convert a :class:`DeepEvalRunResult` to :class:`EvaluationResult`.

        Args:
            cli_result: Parsed result from the CLI engine.

        Returns:
            :class:`EvaluationResult`.
        """
        # Aggregate all metrics from all test case results
        all_metrics: Dict[str, float] = {}
        any_passed = cli_result.success
        errors: List[str] = []

        for r in cli_result.results:
            for m in r.metrics:
                # Keep highest score per metric name or latest
                all_metrics[m.name] = m.score
                if not r.passed and r.error:
                    errors.append(r.error)
            if not r.passed:
                any_passed = False

        # Determine test name
        test_name = "unknown"
        if cli_result.results:
            test_name = cli_result.results[0].name
            # Remove common prefixes
            for prefix in ("test_", "t_"):
                if test_name.startswith(prefix):
                    test_name = test_name[len(prefix):]
                    break

        return EvaluationResult(
            test_name=test_name,
            metrics=all_metrics,
            passed=any_passed,
            details={
                "error": "; ".join(errors) if errors else "",
                "return_code": cli_result.return_code,
            },
        )


__all__ = ["TestRunner", "EvaluationResult"]
