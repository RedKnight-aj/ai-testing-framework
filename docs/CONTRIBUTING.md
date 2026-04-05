# Contributing

> How to contribute to the AI Testing Framework

Thank you for your interest in improving the AI Testing Framework! This guide will help you set up your development environment and contribute new features.

## 🛠️ Development Setup

### Prerequisites

- **Python 3.9+** (3.10+ recommended)
- **Git**
- **An LLM API key** (OpenAI, Anthropic, or an Ollama instance for local testing)

### Quick Start

```bash
# 1. Fork and clone the repository
git clone https://github.com/RedKnight-aj/ai-testing-framework.git
cd ai-testing-framework

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install the package with dev dependencies
pip install -e ".[dev]"

# 4. Set up your API key
export OPENAI_API_KEY="sk-..."
# Or use Ollama for local testing:
# export OPENAI_API_BASE_URL="http://localhost:11434/v1"

# 5. Run the test suite (68 tests)
pytest tests/ -v

# 6. Run a pre-built test suite
pytest test_suites/rag_test_suite.py -v
```

### Code Quality Tools

```bash
# Format code with Black
black src/ test_suites/ tests/ examples/

# Check with Ruff
ruff check src/ test_suites/ tests/

# Type checking with mypy
mypy src/ai_testing/

# Run all quality checks
ruff check src/ && black --check src/ && mypy src/
```

## 📝 Adding a New Test Suite

Adding a test suite for a new AI domain takes ~30 minutes. Here's the step-by-step guide.

### Step 1: Create the Test Suite File

Create a new file in `test_suites/`:

```bash
touch test_suites/your_domain_test_suite.py
```

### Step 2: Write the Test Suite

Follow this template:

```python
"""
Your Domain Test Suite
======================
Description of what this suite evaluates.

Tests: metric1, metric2, metric3

Usage:
    pytest test_suites/your_domain_test_suite.py -v
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from ai_testing.test_cases import YourTestCase
from ai_testing.runner import TestRunner


@pytest.fixture
def runner():
    """Create a test runner with default settings."""
    return TestRunner(model="gpt-4", threshold=0.7)


class TestYourDomain:
    """Tests for your new AI domain."""

    def test_metric_1(self, runner):
        """Test description."""
        test_case = YourTestCase(
            name="test_case_1",
            input="What is X?",
            actual_output="X is a concept that...",
            expected_output="X is a concept that...",
        )

        result = runner.evaluate(
            test_case=test_case,
            metrics=["metric_1", "metric_2"]
        )

        assert result.passed
        assert result.metrics["metric_1"] >= 0.7

    def test_metric_2(self, runner):
        """Test another metric."""
        # ... your test case and assertions

    def test_edge_case(self, runner):
        """Test an edge case (empty input, etc.)."""
        # ...


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 3: Create the Test Case Class

If you need a custom test case class, add it to `src/ai_testing/test_cases/your_domain.py`:

```python
"""Test case class for your new domain."""

from dataclasses import dataclass
from .base import BaseTestCase


@dataclass
class YourTestCase(BaseTestCase):
    """Test case for your domain."""
    # Add domain-specific fields as needed
    custom_field: str = ""

    def to_llm_test_case(self):
        """Convert to DeepEval's LLMTestCase format."""
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=self.input,
            actual_output=self.actual_output,
            expected_output=self.expected_output,
        )
```

### Step 4: Export from Package

Update the test cases `__init__.py`:

```python
# src/ai_testing/test_cases/__init__.py
from .your_domain import YourTestCase

__all__ = [..., "YourTestCase"]
```

### Step 5: Add Documentation

Add your new suite to `docs/CLI.md` under the available suites section and update `README.md`.

### Step 6: Write Tests for Your Suite

Add tests to `tests/` that verify your suite works correctly:

```python
# tests/test_your_domain_suite.py
import pytest

def test_your_domain_suite_runs():
    """Verify the test suite can be imported."""
    from test_suites.your_domain_test_suite import TestYourDomain
    assert TestYourDomain is not None
```

## 📊 Adding Benchmark Datasets

### Step 1: Create the JSON File

Create a new file in `benchmarks/`:

```bash
touch benchmarks/your_domain_benchmark.json
```

### Step 2: Structure Your Benchmark Data

```json
[
  {
    "id": "your_domain_001",
    "query": "What is X?",
    "context": ["X is defined as...", "Another source says..."],
    "expected_answer": "X is a concept that...",
    "category": "your_category",
    "difficulty": "easy",
    "metrics": ["answer_relevancy", "faithfulness"],
    "thresholds": {
      "answer_relevancy": 0.8,
      "faithfulness": 0.9
    }
  }
]
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `query` | string | The input/query/prompt |
| `context` | string[] | Optional retrieval context (for RAG-style tests) |
| `expected_answer` | string | The expected/golden response |
| `category` | string | Domain category for grouping |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `metrics` | string[] | List of metric names to evaluate |
| `thresholds` | object | Per-metric minimum scores |

Recommended: **20+ cases** with a mix of difficulties (30% easy, 50% medium, 20% hard).

### Step 3: Register the Benchmark

Add your benchmark to the list in `docs/CLI.md` and `README.md`.

## 📏 Adding Custom Metrics

### Method 1: Using GEval (Recommended)

For domain-specific evaluation, use DeepEval's GEval with custom criteria:

```python
from ai_testing.runner import TestRunner
from ai_testing.test_cases import BaseTestCase

runner = TestRunner(model="gpt-4")

result = runner.evaluate_with_custom_criteria(
    test_case=BaseTestCase(
        name="custom_eval",
        input="Summarize this article",
        actual_output="The article discusses..."
    ),
    criteria="Evaluate the clarity and conciseness of the summary. "
             "Rate 0.0-1.0 based on how well it captures key points.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
)
```

### Method 2: Creating a Custom Metric Module

For reusable custom metrics:

```python
# src/ai_testing/metrics/custom_eval.py

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.base import BaseMetric


def create_custom_accuracy_metric(
    name: str = "custom_accuracy",
    criteria: str = "Evaluate factual accuracy.",
    threshold: float = 0.7
) -> BaseMetric:
    """Create a custom GEval metric."""
    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
    )
```

Then register it in the runner's `_get_metric` method.

## 🚦 Adding Quality Gates

### Adding a New Gate Profile

Edit `data/quality_gates.yml` and add a new profile:

```yaml
# Your custom gate profile
my_custom_gate:
  description: "Custom gate for my specific use case"
  required_metrics:
    - answer_relevancy
    - faithfulness
    - toxicity
  minimum_scores:
    answer_relevancy: 0.85
    faithfulness: 0.8
    toxicity: 0.15
  fail_conditions:
    - any_metric_below_threshold: true
    - overall_score_below: 0.75
  on_failure:
    action: "warn"
    notify_channels: ["slack"]
    message: "Custom gate check failed"
```

### Gate Profile Structure

| Key | Type | Description |
|-----|------|-------------|
| `description` | string | Human-readable description |
| `required_metrics` | string[] | Metrics that must be evaluated |
| `minimum_scores` | object | Min score per metric |
| `fail_conditions` | array | Rules that trigger failure |
| `on_failure.action` | string | `block_deploy`, `warn`, or `log` |
| `on_failure.notify_channels` | string[] | Where to send alerts |
| `on_failure.message` | string | Explanation message |

### Testing Your Gate

```bash
ai-test gate --profile my_custom_gate --suite rag
```

## 🔄 Pull Request Guidelines

### Before Submitting

1. **Fork the repository** and create a feature branch
2. **Follow the code style** — Black formatting, Ruff linting, mypy type hints
3. **Write tests** — All new code must have corresponding tests
4. **Update documentation** — Update README, CLI.md, and ARCHITECTURE.md as needed
5. **Run the full test suite**:

```bash
pytest tests/ -v --cov=ai_testing
```

### Branch Naming

```
feature/rag-benchmarks
fix/quality-gate-parsing
docs/cli-reference
test/summarization-suite
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add summarization test suite
fix: resolve quality gate threshold comparison
docs: update CLI reference with new options
test: add benchmark dataset tests
```

### Pull Request Template

```markdown
## Summary
<!-- What does this PR do? -->

## Changes
<!-- Key changes in bullet points -->

## Testing
<!-- How did you test this? -->
- [ ] Ran existing test suite (`pytest tests/ -v`)
- [ ] Added new tests
- [ ] Updated documentation

## Related Issues
Closes #XX
```

### Review Process

1. All PRs require at least one approving review
2. All CI checks must pass (linting, type checks, tests)
3. Documentation must be updated for user-facing changes
4. Semantic versioning will be applied to releases

## 🏗️ Architecture Overview

For in-depth understanding of how the framework works, see [ARCHITECTURE.md](ARCHITECTURE.md).

Key directories:

```
src/ai_testing/          ← Core framework code
test_suites/             ← Pre-built pytest test suites
benchmarks/              ← JSON benchmark datasets
data/                    ← Quality gates + metrics catalog
examples/                ← Real-world usage examples
tests/                   ← Unit tests (68 and counting)
docs/                    ← Documentation
```

## 🙏 Thank You

Your contributions help make AI evaluation accessible to everyone. Whether it's fixing a typo, adding a test suite, or improving documentation — every contribution matters.

- **Star** the repo if you find it useful ⭐
- **Share** it with your team
- **Report** issues you encounter
- **Contribute** test suites, benchmarks, or documentation

---

Questions? Open an [issue](https://github.com/RedKnight-aj/ai-testing-framework/issues) or start a [discussion](https://github.com/RedKnight-aj/ai-testing-framework/discussions).
