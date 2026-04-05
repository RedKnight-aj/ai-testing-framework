"""
DeepEval CLI Engine — subprocess wrapper for DeepEval evaluation.

This module runs DeepEval as an external CLI tool (``deepeval test run``)
so the framework acts as the *brain* (configuration, result parsing, quality
gates) rather than the *muscle* (the actual evaluation engine).

Usage::

    engine = DeepEvalEngine()
    engine.check_dependencies()
    results = engine.run_test_file("tests/test_my_model.py")
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricScore:
    """Score reported by a single metric.

    Attributes:
        name: Metric name (e.g. ``"answer_relevancy"``).
        score: Numeric score 0-1.
        threshold: Configured pass threshold.
        passed: ``True`` if *score >= threshold*.
        reason: Optional explanation from DeepEval.
    """

    name: str
    score: float = 0.0
    threshold: float = 0.5
    passed: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class TestCaseResult:
    """Result for a single test case.

    Attributes:
        name: Test case identifier.
        passed: ``True`` if the test case passed all metrics.
        metrics: List of individual metric scores.
        error: Optional error message.
    """

    name: str
    passed: bool = True
    metrics: List[MetricScore] = field(default_factory=list)
    error: str = ""
    stdout: str = ""
    stderr: str = ""

    @property
    def avg_score(self) -> float:
        """Average metric score across all metrics."""
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "test_name": self.name,
            "metrics": {m.name: m.score for m in self.metrics},
            "passed": self.passed,
            "score": self.avg_score,
            "threshold": self.metrics[0].threshold if self.metrics else 0.5,
            "details": {
                "error": self.error,
                "metric_details": [m.to_dict() for m in self.metrics],
            },
        }


@dataclass
class DeepEvalRunResult:
    """Complete result of a DeepEval CLI run.

    Attributes:
        success: ``True`` if the CLI returned exit code 0.
        results: Parsed per-test-case results.
        return_code: Raw subprocess exit code.
        stdout: Full stdout from the subprocess.
        stderr: Full stderr from the subprocess.
        extra: Any additional metadata captured.
    """

    success: bool = False
    results: List[TestCaseResult] = field(default_factory=list)
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Average score across all results."""
        if not self.results:
            return 0.0
        return sum(r.avg_score for r in self.results) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "success": self.success,
            "overall_score": self.overall_score,
            "results": [r.to_dict() for r in self.results],
            "return_code": self.return_code,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DeepEvalEngine:
    """CLI wrapper for DeepEval.

    Invokes ``deepeval test run`` as a subprocess, collects structured output,
    and returns parsed :class:`DeepEvalRunResult` objects.

    Args:
        timeout: Maximum seconds to wait for a subprocess call.
        verbose: Whether to capture and return full stdout/stderr.
    """

    def __init__(self, timeout: int = 600, verbose: bool = False) -> None:
        self._timeout = timeout
        self._verbose = verbose
        self._last_result: Optional[DeepEvalRunResult] = None

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    def check_dependencies(self) -> Tuple[bool, str]:
        """Verify that the ``deepeval`` CLI is available.

        Returns:
            Tuple of ``(is_available, message)``.
        """
        bin_path = self._find_deepeval()
        if bin_path is None:
            return False, (
                "DeepEval CLI not found on PATH.\n"
                "Install it with:  pip install 'deepeval>=2.0'\n"
                "Then ensure it is on your PATH."
            )
        try:
            result = subprocess.run(
                [bin_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            version_output = result.stdout.strip() or result.stderr.strip()
            return True, f"DeepEval found at {bin_path} ({version_output})"
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"DeepEval found but could not execute: {exc}"

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _find_deepeval() -> Optional[str]:
        """Locate the ``deepeval`` binary on the system."""
        # 1. Environment override
        env_path = os.environ.get("DEEPEVAL_BIN")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                return str(p)

        # 2. PATH lookup
        binary = shutil.which("deepeval")
        if binary:
            return binary

        # 3. Common pip bin locations
        for search_path in [
            os.path.expanduser("~/.local/bin"),
            "/usr/local/bin",
        ]:
            candidate = Path(search_path) / "deepeval"
            if candidate.is_file():
                return str(candidate)
        return None

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(
        self,
        config_path: str | Path,
        *,
        extra_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> DeepEvalRunResult:
        """Run ``deepeval test run <config_path>`` via subprocess.

        The *config_path* should point to a pytest-compatible file that uses
        DeepEval's ``@deepeval`` markers / fixtures.

        Args:
            config_path: Path to a test file or config.
            extra_args: Additional CLI arguments (e.g. ``["-n", "4"]`` for parallelism).
            timeout: Per-run timeout override (default: instance default).

        Returns:
            :class:`DeepEvalRunResult` with parsed results.
        """
        deepeval_bin = self._find_deepeval()
        if deepeval_bin is None:
            return DeepEvalRunResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="deepeval not installed or not on PATH",
                extra={"error": "engine_not_found"},
            )

        timeout = timeout or self._timeout
        extra_args = extra_args or []

        cmd: List[str] = [
            deepeval_bin,
            "test",
            "run",
            str(config_path),
        ]
        # If there's a JSON report plugin, capture its output
        if "--json-report-file" not in extra_args:
            extra_args.extend(["-q"])

        cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return DeepEvalRunResult(
                success=False,
                return_code=-1,
                stdout=str(exc.stdout or ""),
                stderr=f"Timeout after {timeout}s: {exc.stderr or ''}",
                extra={"error": "timeout"},
            )
        except OSError as exc:
            return DeepEvalRunResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(exc),
                extra={"error": "os_error"},
            )

        # Also try to capture pytest JSON report if produced
        json_report = None
        json_path = None
        for arg in extra_args:
            if arg.endswith(".json") and Path(arg).exists():
                json_path = Path(arg)
                break
        if json_path:
            try:
                json_report = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass

        parsed = self._parse_output(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            json_report=json_report,
        )
        self._last_result = parsed
        return parsed

    def run_test(
        self,
        *,
        test_name: str,
        input_text: str,
        actual_output: str,
        expected_output: str = "",
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> DeepEvalRunResult:
        """Generate a temporary DeepEval pytest file and run it.

        This is the primary entry point for single-test evaluations.  It builds
        a minimal pytest-compatible file, invokes ``deepeval test run``, and
        returns structured results.

        Args:
            test_name: Identifier for the test case.
            input_text: Input prompt/query.
            actual_output: Model's actual output.
            expected_output: Expected / reference answer.
            context: Context passages (for RAG evaluation).
            retrieval_context: Documents retrieved before generation.
            metrics: List of metric names to evaluate.
            threshold: Minimum score to pass (0-1).

        Returns:
            :class:`DeepEvalRunResult` with evaluation results.
        """
        metrics = metrics or ["answer_relevancy"]
        context = context or []
        retrieval_context = retrieval_context or []

        test_file = self.generate_test_file(
            test_name=test_name,
            input_text=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
            metrics=metrics,
            threshold=threshold,
        )

        result = self.run(test_file)

        # Clean up temp file
        try:
            os.unlink(test_file)
        except OSError:
            pass

        return result

    # ------------------------------------------------------------------
    # Public: benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        dataset_path: str | Path,
        *,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
        extra_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> DeepEvalRunResult:
        """Run a DeepEval benchmark against a dataset.

        The dataset should be a JSON file with an array of test objects, each
        containing ``input``, ``actual_output``, and optionally
        ``expected_output`` and ``retrieval_context``.

        Args:
            dataset_path: Path to JSON benchmark dataset.
            metrics: Metrics to evaluate with.
            threshold: Pass threshold.
            extra_args: Extra CLI flags.
            timeout: Timeout override.

        Returns:
            :class:`DeepEvalRunResult` with all results.
        """
        dataset = Path(dataset_path)
        if not dataset.exists():
            return DeepEvalRunResult(
                success=False,
                return_code=-1,
                stderr=f"Dataset not found: {dataset}",
                extra={"error": "file_not_found"},
            )

        # Generate a test file from the dataset
        test_file = self.generate_benchmark_file(
            dataset_path=str(dataset),
            metrics=metrics or ["answer_relevancy"],
            threshold=threshold,
        )

        result = self.run(test_file, extra_args=extra_args, timeout=timeout)

        # Clean up
        try:
            os.unlink(test_file)
        except OSError:
            pass

        return result

    # ------------------------------------------------------------------
    # Public: results access
    # ------------------------------------------------------------------

    def get_results(self) -> Optional[DeepEvalRunResult]:
        """Return the results from the most recent ``run()`` call.

        Returns:
            :class:`DeepEvalRunResult` or ``None`` if no run yet.
        """
        return self._last_result

    # ------------------------------------------------------------------
    # Test file generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_test_file(
        *,
        test_name: str,
        input_text: str,
        actual_output: str,
        expected_output: str = "",
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> str:
        """Generate a pytest-compatible DeepEval test file.

        The returned file is a Python module that DeepEval's test runner can
        execute.

        Args:
            test_name: Unique name for the test.
            input_text: Input prompt.
            actual_output: Model response to evaluate.
            expected_output: Expected / reference output.
            context: Context passages.
            retrieval_context: Retrieved documents.
            metrics: Metrics to evaluate.
            threshold: Pass threshold.

        Returns:
            Path to the generated file.
        """
        context = context or []
        retrieval_context = retrieval_context or []
        metrics = metrics or ["answer_relevancy"]

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", test_name)
        if not safe_name[0:1].isalpha():
            safe_name = "t_" + safe_name

        # Build metric instantiation code
        metric_lines: List[str] = []
        for m in metrics:
            m_clean = re.sub(r"[^a-zA-Z0-9_]", "_", m)
            metric_lines.append(f"    metrics.append({m_clean}(threshold={threshold}))")
        metric_code = "\n".join(metric_lines)

        # Safe string helpers
        def _py_str(s: str) -> str:
            return json.dumps(s)

        def _py_list(lst: List[str]) -> str:
            return json.dumps(lst)

        # Build expected_output arg
        expected_arg = _py_str(expected_output) if expected_output else "None"

        content = (
            f"# Auto-generated DeepEval test — do not edit\n"
            f"from deepeval.test_case import LLMTestCase\n"
            f"from deepeval.metrics import (\n"
            f"    AnswerRelevancyMetric,\n"
            f"    FaithfulnessMetric,\n"
            f"    HallucinationMetric,\n"
            f"    ToxicityMetric,\n"
            f"    BiasMetric,\n"
            f"    SummarizationMetric,\n"
            f"    GEval,\n"
            f"    ContextualPrecisionMetric,\n"
            f"    ContextualRecallMetric,\n"
            f"    ContextualRelevancyMetric,\n"
            f")\n\n\n"
            f"def test_{safe_name}():\n"
            f"    test_case = LLMTestCase(\n"
            f"        input={_py_str(input_text)},\n"
            f"        actual_output={_py_str(actual_output)},\n"
            f"        expected_output={expected_arg},\n"
            f"        context={_py_list(context)},\n"
            f"        retrieval_context={_py_list(retrieval_context)},\n"
            f"    )\n"
            f"    metrics = []\n"
            f"{metric_code}\n"
            f"    for metric in metrics:\n"
            f"        metric.measure(test_case)\n"
            f"        assert metric.is_successful(), (\n"
            f'            f"{{metric.name}} failed: score={{metric.score:.2f}} "\n'
            f'            f"< threshold={{metric.threshold}}"\n'
            f"        )\n"
        )

        fd, path = tempfile.mkstemp(suffix="_test.py", prefix=f"deepeval_{safe_name}_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def generate_benchmark_file(
        self,
        *,
        dataset_path: str,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> str:
        """Generate a pytest file from a benchmark dataset.

        Args:
            dataset_path: Path to JSON dataset file.
            metrics: Metrics to run per case.
            threshold: Pass threshold.

        Returns:
            Path to the generated test file.
        """
        metrics = metrics or ["answer_relevancy"]
        dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))

        if isinstance(dataset, dict) and "test_cases" in dataset:
            cases = dataset["test_cases"]
        elif isinstance(dataset, list):
            cases = dataset
        else:
            raise ValueError(
                f"Dataset must be a list or dict with 'test_cases' key, "
                f"got {type(dataset).__name__}"
            )

        lines: List[str] = [
            "# Auto-generated DeepEval benchmark — do not edit",
            "from deepeval.test_case import LLMTestCase",
            "from deepeval.metrics import (",
            "    AnswerRelevancyMetric,",
            "    FaithfulnessMetric,",
            "    HallucinationMetric,",
            "    ToxicityMetric,",
            "    BiasMetric,",
            "    SummarizationMetric,",
            "    GEval,",
            "    ContextualPrecisionMetric,",
            "    ContextualRecallMetric,",
            "    ContextualRelevancyMetric,",
            ")",
            "",
        ]

        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            case_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(case.get("name", f"case_{idx}")))
            if not case_name[0:1].isalpha():
                case_name = "t_" + case_name

            lines.append(f"def test_bench_{case_name}():")
            lines.append(f"    test_case = LLMTestCase(")
            lines.append(f"        input={json.dumps(case.get('input', ''))},")
            lines.append(f"        actual_output={json.dumps(case.get('actual_output', ''))},")
            lines.append(f"        expected_output={json.dumps(case.get('expected_output'))},")
            lines.append(f"        context={json.dumps(case.get('context', []))},")
            lines.append(f"        retrieval_context={json.dumps(case.get('retrieval_context', []))},")
            lines.append(f"    )")

            for m in metrics:
                m_clean = re.sub(r"[^a-zA-Z0-9_]", "_", m)
                lines.append(f"    metric = {m_clean}(threshold={threshold})")
                lines.append(f"    metric.measure(test_case)")
                lines.append(f"    assert metric.is_successful(), f'{{metric.name}} failed on {case_name}'")

            lines.append("")

        fd, path = tempfile.mkstemp(suffix="_bench.py", prefix="deepeval_benchmark_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self,
        *,
        stdout: str,
        stderr: str,
        return_code: int,
        json_report: Optional[Dict[str, Any]] = None,
    ) -> DeepEvalRunResult:
        """Parse DeepEval CLI output into :class:`DeepEvalRunResult`.

        Strategy:
        1. If a pytest JSON report was captured, parse that directly.
        2. Otherwise, heuristically parse the textual CLI output.

        Args:
            stdout: Subprocess stdout.
            stderr: Subprocess stderr.
            return_code: Exit code.
            json_report: Optional parsed JSON report.

        Returns:
            Parsed result.
        """
        # Prefer structured JSON report if available
        if json_report:
            return self._parse_json_report(json_report, stdout, stderr, return_code)

        # Fall back to text parsing
        return self._parse_text_output(stdout, stderr, return_code)

    @staticmethod
    def _build_result(
        test_results: List[TestCaseResult],
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> DeepEvalRunResult:
        """Build a :class:`DeepEvalRunResult` from parsed test results."""
        return DeepEvalRunResult(
            success=return_code == 0 and all(r.passed for r in test_results),
            results=test_results,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            extra={"metric_count": sum(len(r.metrics) for r in test_results)},
        )

    def _parse_json_report(
        self,
        report: Dict[str, Any],
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> DeepEvalRunResult:
        """Parse a pytest-JSON report into result objects."""
        test_results: List[TestCaseResult] = []
        tests = report.get("tests", [])
        for test in tests:
            outcome = test.get("outcome", "unknown")
            name = test.get("nodeid", test.get("name", "unknown"))
            passed = outcome == "passed"

            metric_scores: List[MetricScore] = []
            for call in test.get("call", test.get("phases", [])):
                if isinstance(call, dict):
                    dur = call.get("duration", 0)
                    if "metric" in str(call):
                        metric_scores.append(
                            MetricScore(name="execution", score=1.0 if passed else 0.0, reason=f"duration: {dur:.2f}s")
                        )
            if not metric_scores:
                # Assume a single composite metric from the pass/fail
                metric_scores.append(
                    MetricScore(
                        name="composite",
                        score=1.0 if passed else 0.0,
                        threshold=0.5,
                        passed=passed,
                    )
                )

            test_results.append(
                TestCaseResult(
                    name=name,
                    passed=passed,
                    metrics=metric_scores,
                    error=str(test.get("call", {}).get("crash", test.get("longrepr", "")))[:500]
                    if not passed
                    else "",
                )
            )

        return self._build_result(test_results, stdout, stderr, return_code)

    @staticmethod
    def _parse_text_output(
        stdout: str,
        stderr: str,
        return_code: int,
    ) -> DeepEvalRunResult:
        """Heuristically parse DeepEval CLI text output."""
        test_results: List[TestCaseResult] = []
        combined = stdout + "\n" + stderr

        # Look for deepeval-style score lines
        # Pattern: "Answer Relevancy: 0.85" or "metric_name: score"
        score_pattern = re.compile(
            r"^(?:\s*[-\u2502\s]*)?([A-Za-z][A-Za-z\s_]+?):\s*([\d.]+)",
            re.MULTILINE,
        )

        # Look for PASSED/FAILED lines
        result_pattern = re.compile(
            r"^(test_\w+).*?\b(PASSED|FAILED)\b",
            re.MULTILINE,
        )

        # Extract named test results
        for match in result_pattern.finditer(combined):
            name = match.group(1)
            passed = match.group(2) == "PASSED"
            test_results.append(
                TestCaseResult(
                    name=name,
                    passed=passed,
                    metrics=[
                        MetricScore(
                            name="composite",
                            score=1.0 if passed else 0.0,
                            threshold=0.5,
                            passed=passed,
                        )
                    ],
                )
            )

        # If no named tests found, look for metric scores in the output
        if not test_results:
            scores: List[MetricScore] = []
            for match in score_pattern.finditer(combined):
                metric_name = match.group(1).strip().lower().replace(" ", "_")
                try:
                    score_val = float(match.group(2))
                    scores.append(
                        MetricScore(
                            name=metric_name,
                            score=score_val,
                            threshold=0.5,
                            passed=score_val >= 0.5,
                        )
                    )
                except ValueError:
                    continue
            if scores:
                test_results.append(
                    TestCaseResult(
                        name="aggregate",
                        passed=all(s.passed for s in scores),
                        metrics=scores,
                    )
                )

        # If still no results, create a minimal result from return code
        if not test_results:
            test_results.append(
                TestCaseResult(
                    name="execution",
                    passed=return_code == 0,
                    metrics=[],
                    error=stderr.strip()[:500] if return_code != 0 else "",
                )
            )

        return DeepEvalRunResult(
            success=return_code == 0 and all(r.passed for r in test_results),
            results=test_results,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            extra={"metric_count": sum(len(r.metrics) for r in test_results)},
        )
