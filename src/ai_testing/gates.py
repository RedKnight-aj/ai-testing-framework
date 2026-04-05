"""
Quality Gates
Evaluates test results against predefined quality standards and thresholds.
"""

import yaml
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from .config import Settings


class GateStatus(Enum):
    """Quality gate evaluation status."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class GateFailure:
    """Details of a quality gate failure."""

    rule: str
    description: str
    actual_value: Union[float, int, str]
    expected_value: Union[float, int, str]
    severity: str = "error"  # error, warning


@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation."""

    gate_name: str
    config_name: str
    status: GateStatus
    passed: bool
    overall_score: float = 0.0
    failures: List[GateFailure] = field(default_factory=list)
    warnings: List[GateFailure] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_name": self.gate_name,
            "config_name": self.config_name,
            "status": self.status.value,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "failures": [
                {
                    "rule": f.rule,
                    "description": f.description,
                    "actual_value": f.actual_value,
                    "expected_value": f.expected_value,
                    "severity": f.severity,
                }
                for f in self.failures
            ],
            "warnings": [
                {
                    "rule": w.rule,
                    "description": w.description,
                    "actual_value": w.actual_value,
                    "expected_value": w.expected_value,
                    "severity": w.severity,
                }
                for w in self.warnings
            ],
            "metadata": self.metadata,
        }


class QualityGate:
    """
    Evaluates test results against quality standards.

    Supports multiple predefined gate configurations and custom rules.

    Example:
        >>> gate = QualityGate("strict")
        >>> result = gate.evaluate(evaluation_results)
        >>> print(f"Status: {result.status.value}")
    """

    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        settings: Optional[Settings] = None
    ):
        """
        Initialize quality gate.

        Args:
            config: Gate configuration name (str) or config dict
            settings: Evaluation settings
        """
        self.settings = settings or Settings()

        if isinstance(config, str):
            self.config_name = config
            self.config = self._load_config(config)
        else:
            self.config_name = "custom"
            self.config = config

        self.gate_name = self.config.get("description", self.config_name)

    def evaluate(self, results: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> QualityGateResult:
        """
        Evaluate results against quality gate criteria.

        Args:
            results: Evaluation results (file path, dict, or list of results)

        Returns:
            QualityGateResult with evaluation outcome
        """
        # Load results data
        results_data = self._load_results(results)

        # Calculate overall statistics
        stats = self._calculate_statistics(results_data)

        # Evaluate against gate criteria
        failures = []
        warnings = []
        status = GateStatus.PASS

        # Check fail conditions
        for condition, config in self.config.get("fail_conditions", {}).items():
            if self._evaluate_condition(condition, config, stats):
                status = GateStatus.FAIL
                failures.append(GateFailure(
                    rule=condition,
                    description=f"Failed condition: {condition}",
                    actual_value=stats.get(condition.replace("_", " "), 0),
                    expected_value=config,
                    severity="error"
                ))

        # Check metric thresholds
        for metric, threshold in self.config.get("minimum_scores", {}).items():
            actual_score = stats.get(f"avg_{metric}", 0)
            if actual_score < threshold:
                status = GateStatus.FAIL
                failures.append(GateFailure(
                    rule=f"{metric}_threshold",
                    description=f"{metric} score below threshold",
                    actual_value=actual_score,
                    expected_value=threshold,
                    severity="error"
                ))

        # Check warnings (less strict conditions)
        for condition, config in self.config.get("warn_conditions", {}).items():
            if self._evaluate_condition(condition, config, stats):
                if status == GateStatus.PASS:
                    status = GateStatus.WARN
                warnings.append(GateFailure(
                    rule=condition,
                    description=f"Warning condition: {condition}",
                    actual_value=stats.get(condition.replace("_", " "), 0),
                    expected_value=config,
                    severity="warning"
                ))

        # Check required metrics presence
        required_metrics = set(self.config.get("required_metrics", []))
        available_metrics = set()
        for result in results_data:
            available_metrics.update(result.get("metrics", {}).keys())

        missing_metrics = required_metrics - available_metrics
        if missing_metrics:
            status = GateStatus.FAIL
            for metric in missing_metrics:
                failures.append(GateFailure(
                    rule="missing_metric",
                    description=f"Required metric not found: {metric}",
                    actual_value="missing",
                    expected_value="present",
                    severity="error"
                ))

        result = QualityGateResult(
            gate_name=self.gate_name,
            config_name=self.config_name,
            status=status,
            passed=status in [GateStatus.PASS, GateStatus.WARN],
            overall_score=stats.get("overall_average", 0),
            failures=failures,
            warnings=warnings,
            metadata={
                "total_tests": stats.get("total_tests", 0),
                "pass_rate": stats.get("pass_rate", 0),
                "config": self.config,
                "statistics": stats,
            }
        )

        return result

    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """Load quality gate configuration."""
        config_path = Path(__file__).parent / "data" / "quality_gates.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Quality gates config not found: {config_path}")

        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)

        if config_name not in configs:
            available = list(configs.keys())
            raise ValueError(f"Unknown gate config '{config_name}'. Available: {available}")

        return configs[config_name]

    def _load_results(self, results: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Load evaluation results data."""
        if isinstance(results, (str, Path)):
            # Load from file
            results_path = Path(results)
            if not results_path.exists():
                raise FileNotFoundError(f"Results file not found: {results_path}")

            with open(results_path, 'r') as f:
                if results_path.suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported results file format: {results_path.suffix}")

            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("Results file must contain a 'results' array or be an array")

        elif isinstance(results, dict):
            # Results dict
            if "results" in results:
                return results["results"]
            else:
                return [results]

        elif isinstance(results, list):
            # List of results
            return results

        else:
            raise ValueError("Invalid results format")

    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from evaluation results."""
        if not results:
            return {}

        stats = {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.get("passed", False)),
            "failed_tests": len(results) - sum(1 for r in results if r.get("passed", False)),
        }

        stats["pass_rate"] = stats["passed_tests"] / stats["total_tests"] if stats["total_tests"] > 0 else 0

        # Calculate metric averages
        metric_sums = {}
        metric_counts = {}

        for result in results:
            metrics = result.get("metrics", {})
            for metric, score in metrics.items():
                metric_sums[metric] = metric_sums.get(metric, 0) + score
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        for metric in metric_sums:
            stats[f"avg_{metric}"] = metric_sums[metric] / metric_counts[metric]

        # Overall average
        all_scores = []
        for result in results:
            score = result.get("score", 0)
            all_scores.append(score)

        stats["overall_average"] = sum(all_scores) / len(all_scores) if all_scores else 0

        return stats

    def _evaluate_condition(self, condition: str, config: Any, stats: Dict[str, Any]) -> bool:
        """Evaluate a gate condition against statistics."""
        if condition == "any_metric_below_threshold":
            # Check if any required metric is below its threshold
            required_metrics = self.config.get("required_metrics", [])
            min_scores = self.config.get("minimum_scores", {})

            for metric in required_metrics:
                avg_score = stats.get(f"avg_{metric}", 0)
                threshold = min_scores.get(metric, 0)
                if avg_score < threshold:
                    return True
            return False

        elif condition == "overall_score_below":
            return stats.get("overall_average", 0) < config

        elif condition == "pass_rate_below":
            return stats.get("pass_rate", 0) < config

        elif condition == "critical_metrics_below_threshold":
            # Similar to any_metric_below_threshold but for critical metrics only
            critical_metrics = self.config.get("critical_metrics", [])
            min_scores = self.config.get("minimum_scores", {})

            for metric in critical_metrics:
                avg_score = stats.get(f"avg_{metric}", 0)
                threshold = min_scores.get(metric, 0)
                if avg_score < threshold:
                    return True
            return False

        elif condition.endswith("_below"):
            # Generic below condition
            metric_name = condition.replace("_below", "")
            threshold = config
            actual_value = stats.get(metric_name, 0)
            return actual_value < threshold

        elif condition.endswith("_above"):
            # Generic above condition
            metric_name = condition.replace("_above", "")
            threshold = config
            actual_value = stats.get(metric_name, 0)
            return actual_value > threshold

        else:
            # Unknown condition
            return False


# Convenience functions
def evaluate_quality_gate(
    config_name: str,
    results: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    settings: Optional[Settings] = None
) -> QualityGateResult:
    """
    Convenience function to evaluate a quality gate.

    Args:
        config_name: Quality gate configuration name
        results: Evaluation results
        settings: Optional settings

    Returns:
        QualityGateResult
    """
    gate = QualityGate(config_name, settings)
    return gate.evaluate(results)


def list_available_gates() -> List[str]:
    """List all available quality gate configurations."""
    config_path = Path(__file__).parent / "data" / "quality_gates.yml"

    if not config_path.exists():
        return []

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    return list(configs.keys())


def get_gate_config(gate_name: str) -> Dict[str, Any]:
    """Get quality gate configuration details."""
    config_path = Path(__file__).parent / "data" / "quality_gates.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Quality gates config not found: {config_path}")

    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    if gate_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown gate config '{gate_name}'. Available: {available}")

    return configs[gate_name]