# Architecture

> How the AI Testing Framework works under the hood

## Overview

The AI Testing Framework is a **thin, opinionated layer** on top of DeepEval's Python API. It takes the raw evaluation power of DeepEval and wraps it with:

1. **Pre-built test suite templates** — ready-to-run pytest files for 10 AI domains
2. **Benchmark datasets** — 120+ curated evaluation cases
3. **Quality gate decision engine** — deployment block/warn/log rules
4. **Multiple reporters** — JSON, HTML, Markdown, Slack, Console
5. **CLI interface** — `ai-test` command with four sub-commands

## Design Philosophy

- **Zero config to start** — sensible defaults for everything
- **Convention over configuration** — naming patterns, directory layouts, threshold recommendations
- **Composable** — mix and match suites, benchmarks, gates, reporters
- **Self-hosted first** — no SaaS dependency; cloud integrations are optional additions

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                            │
│  ┌──────────────────┐     ┌──────────────────┐                       │
│  │   ai-test CLI    │     │   Python API     │                       │
│  │  run/benchmark/  │     │  TestRunner      │                       │
│  │  gate/metrics    │     │  BenchmarkRunner │                       │
│  └────────┬─────────┘     │  QualityGate     │                       │
│           │               └────────┬─────────┘                       │
└───────────┼──────────────────────┬─┼─────────────────────────────────┘
            │                      │ │
            │  ┌──────────────────┐│ │
            └─►│  Test Suite      ││ │ 10 pytest files
               │  Templates       ││ │ RAG, Chatbot, Agent, etc.
               │                  ││ │
               └────────┬─────────┘│ │
                        │          │ │
            ┌───────────┴──────────▼─▼─────────┐
            │         TestRunner               │
            │  ┌────────────────────────────┐  │
            │  │ _get_metric(name)          │  │ Maps → DeepEval
            │  │ evaluate(case, metrics)    │  │ metrics by name
            │  │ evaluate_batch(cases, ...) │  │
            │  │ evaluate_custom(criteria)  │  │
            │  └────────────┬───────────────┘  │
            └───────────────┼──────────────────┘
                            │
            ┌───────────────▼──────────────────┐
            │         DeepEval API              │
            │  ┌────────────────────────────┐  │
            │  │ AnswerRelevancyMetric      │  │
            │  │ FaithfulnessMetric         │  │
            │  │ HallucinationMetric        │  │
            │  │ ToxicityMetric             │  │
            │  │ BiasMetric                 │  │
            │  │ ContextualPrecisionMetric  │  │
            │  │ ContextualRecallMetric     │  │
            │  │ ContextualRelevancyMetric  │  │
            │  │ SummarizationMetric        │  │
            │  │ GEval                      │  │
            │  └────────────────────────────┘  │
            └───────────────┬──────────────────┘
                            │
            ┌───────────────▼──────────────────┐
            │    EvaluationResult              │
            │    {metrics, passed, score}      │
            └───────┬──────────────┬───────────┘
                    │              │
    ┌───────────────▼─┐   ┌───────▼────────────┐
    │   QualityGate   │   │     Reporters      │
    │  ┌───────────┐  │   │ ┌────────────────┐ │
    │  │ strict    │  │   │ │ JSONReporter   │ │
    │  │ moderate  │  │   │ │ HTMLReporter   │ │
    │  │ quick     │  │   │ │ MarkdownRep.   │ │
    │  │ rag_prod  │  │   │ │ SlackReporter  │ │
    │  │ + 5 more  │  │   │ │ ConsoleReporter│ │
    │  └───────────┘  │   │ └────────────────┘ │
    └─────────────────┘   └────────────────────┘
```

## Core Classes

### `TestRunner` (`src/ai_testing/runner.py`)

The main execution engine. Wraps DeepEval's `assert_test` and `evaluate` with a Pythonic API.

```python
class TestRunner:
    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        cloud_enabled: bool = False,
        **kwargs
    )

    def evaluate(test_case, metrics: List[str]) -> EvaluationResult
    def evaluate_batch(test_cases, metrics) -> List[EvaluationResult]
    def evaluate_custom_criteria(test_case, criteria, params) -> EvaluationResult
```

**Key method — `_get_metric(metric_name)`:** Maps string metric names to DeepEval metric instances with the configured threshold.

```python
metrics_map = {
    "answer_relevancy":   AnswerRelevancyMetric,
    "faithfulness":       FaithfulnessMetric,
    "hallucination":      HallucinationMetric,
    "toxicity":           ToxicityMetric,
    "bias":               BiasMetric,
    "summarization":      SummarizationMetric,
    "contextual_precision": ContextualPrecisionMetric,
    "contextual_recall":    ContextualRecallMetric,
    "contextual_relevancy": ContextualRelevancyMetric,
}
```

### `EvaluationResult` (dataclass)

```python
@dataclass
class EvaluationResult:
    test_name: str
    metrics: Dict[str, float]       # {"answer_relevancy": 0.95, ...}
    passed: bool
    threshold: float
    details: Dict[str, Any]

    @property
    def score(self) -> float:       # Average across all metrics
```

## Test Suite Organization

Test suites live in `test_suites/` as standalone pytest files. Each suite:

1. **Defines test cases** with realistic inputs, expected outputs, and retrieval contexts
2. **Imports fixtures** from ai-testing-framework
3. **Runs `assert_test`** against DeepEval metrics
4. **Reports results** via pytest's output

### Suite Structure

```
test_suites/rag_test_suite.py
├── Import test case classes (RAGTestCase, etc.)
├── Define fixture functions (optional)
├── test_answer_relevancy_rag()
├── test_faithfulness_rag()
├── test_contextual_precision_rag()
├── test_hallucination_rag()
└── test_empty_context_edge_case()
```

10 suites cover:
- RAG, Chatbot, Agent, Code Gen
- Summarization, Translation, Classification
- Sentiment, Embedding, Q&A

## Benchmarks

Benchmarks are JSON files in `benchmarks/` containing structured evaluation cases.

### Benchmark File Structure

```json
[
  {
    "id": "rag_001",
    "query": "What is supervised learning?",
    "context": ["Supervised learning uses labeled data..."],
    "expected_answer": "Supervised learning uses labeled data for training.",
    "category": "machine_learning_basics",
    "difficulty": "easy",
    "metrics": ["answer_relevancy", "faithfulness", "contextual_precision"],
    "thresholds": {
      "answer_relevancy": 0.8,
      "faithfulness": 0.9,
      "contextual_precision": 0.85
    }
  }
]
```

### How Benchmarks Run

```
BenchmarkRunner
    │
    ├─ Load JSON benchmark file
    ├─ For each case:
    │   ├─ Create test case object
    │   ├─ Map metric names → DeepEval instances
    │   ├─ Run assert_test()
    │   ├─ Collect scores
    │   └─ Check against case-specific thresholds
    ├─ Aggregate results (avg score, pass rate)
    └─ Return BenchmarkResult
```

### BenchmarkResult

```python
class BenchmarkResult:
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    results: List[EvaluationResult]
```

## Quality Gates

### Overview

Quality gates are **deployment decision rules** configured in `data/quality_gates.yml`. Each gate profile defines:

- **`required_metrics`** — which metrics must be evaluated
- **`minimum_scores`** — minimum acceptable score per metric
- **`fail_conditions`** — rules that trigger failure
- **`on_failure`** — action (block/warn/log) + notification channels

### Gate Profiles

10 profiles: `strict`, `moderate`, `quick`, `rag_production`, `chatbot_production`, `codegen_production`, `agent_production`, `content_moderation`, `academic_research`.

### Gate Decision Engine

```python
class QualityGate:
    def __init__(self, profile: str)     # Load profile from YAML

    def evaluate(metrics: Dict[str, float]) -> QualityGateResult:
        │
        ├─ Check: Are all required_metrics present?
        ├─ Check: Is each metric >= minimum_scores[name]?
        ├─ Check fail_conditions:
        │   ├─ any_metric_below_threshold?
        │   ├─ overall_score_below X.X?
        │   ├─ faithfulness_below X.X?
        │   ├─ toxicity_above X.X?
        │   └─ hallucination_above X.X?
        ├─ Set passed=True/False
        ├─ Set message + action (block/warn/log)
        └─ Return QualityGateResult
```

### QualityGateResult

```python
@dataclass
class QualityGateResult:
    passed: bool          # True = deploy is safe
    action: str           # "block_deploy" | "warn" | "log"
    message: str          # Human-readable explanation
    failed_checks: List[str]  # Which conditions failed
    notify_channels: List[str]  # ["slack", "email", etc.]
```

## Reporters

Five reporter implementations in `ai_testing/reporters/`:

| Reporter | Output | Use Case |
|----------|--------|----------|
| `ConsoleReporter` | Terminal output | Local development |
| `JSONReporter` | `.json` file | Downstream processing, CI |
| `HTMLReporter` | `.html` file | Stakeholder reports |
| `MarkdownReporter` | `.md` file | Documentation |
| `SlackReporter` | Slack channel | Team notifications |

All reporters follow the same interface:

```python
class BaseReporter:
    def generate(results) -> str    # Return formatted output
    def save(results, path)          # Write to file
    def send(results)                # Send to channel (e.g., Slack webhook)
```

## Data Flow Summary

```
ai-test run --suite rag
    │
    ├─ CLI parses args
    ├─ Loads test_suites/rag_test_suite.py
    ├─ pytest collects test functions
    │
    └─ For each test:
        ├─ Create TestCase object
        ├─ TestRunner.evaluate() → DeepEval metrics
        ├─ DeepEval calls LLM for scoring
        ├─ Collect metrics → EvaluationResult
        └─ pytest reports pass/fail

ai-test benchmark --name rag
    │
    ├─ Load benchmarks/rag_benchmark.json
    ├─ For each case → test case → evaluate → score
    └─ Aggregate → BenchmarkResult → Report

ai-test gate --profile rag_production
    │
    ├─ Load data/quality_gates.yml → rag_production profile
    ├─ Run evaluation
    ├─ QualityGate.evaluate(metrics)
    ├─ passed? → exit 0
    └─ failed? → action=block_deploy → exit 1
```

## Configuration Layers

The framework uses a layered configuration approach:

1. **Code defaults** — `TestRunner(threshold=0.5)` etc.
2. **Environment variables** — `OPENAI_API_KEY`, `DEEPEVAL_API_KEY`
3. **Settings object** — `Settings(model="gpt-4", threshold=0.7)`
4. **CLI arguments** — `--model gpt-4 --threshold 0.8`

CLI arguments override env vars, which override code defaults.

## Dependencies

```
ai-testing-framework
├── deepeval>=0.21.0     # Core evaluation metrics
├── pytest>=7.0.0        # Test framework / runner backbone
├── rich>=13.0.0         # Beautiful terminal output
├── pyyaml>=6.0          # Quality gate config parsing
├── click>=8.0.0         # CLI argument parsing
└── pydantic>=2.0.0      # Data validation
```

---

See [CLI.md](CLI.md) for command reference and [CONTRIBUTING.md](CONTRIBUTING.md) for extending the framework.
