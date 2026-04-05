<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/Tests-68%20passing-brightgreen" alt="68 tests passing" />
  <img src="https://img.shields.io/badge/Version-2.0.0-informational" alt="Version 2.0.0" />
  <a href="https://github.com/deepset-ai/deepeval"><img src="https://img.shields.io/badge/Powered%20by-DeepEval-purple" alt="Powered by DeepEval" /></a>
  <a href="https://pypi.org/project/ai-testing-framework/"><img src="https://img.shields.io/pypi/v/ai-testing-framework" alt="PyPI" /></a>
</p>

<h1 align="center">🧪 AI Testing Framework</h1>

<p align="center">
  <strong>The most comprehensive AI testing framework. DeepEval wrapped beautifully.<br>
  10 ready-made test suites. 120+ benchmark cases. Quality gates. Zero config to start.</strong>
</p>

<p align="center"><em>Production-grade evaluation for LLMs, RAG pipelines, AI agents, chatbots, and more.</em></p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#%EF%B8%8F-usage-examples">Usage</a> ·
  <a href="#-cli-reference">CLI</a> ·
  <a href="#-quality-gates">Quality Gates</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-comparison">Comparison</a> ·
  <a href="#-architecture">Architecture</a>
</p>

---

## 🔥 Why This Framework Exists

You're building AI applications — and you need to know they **actually work**. LLMs are non-deterministic. RAG pipelines drift. Agents make unexpected choices. Traditional test frameworks fall short.

The **AI Testing Framework** gives you:

- **10 pre-built test suites** — just plug in your LLM and run
- **6 benchmark datasets** — 120+ evaluation cases out of the box
- **10 quality gate profiles** — production-ready deployment gates
- **20+ DeepEval metrics** — with sensible thresholds pre-configured
- **A powerful CLI** — `ai-test run`, `ai-test benchmark`, `ai-test gate`, `ai-test metrics`

No boilerplate. No config files to write. Just evaluate.

## 🚀 Quick Start

### 30 Seconds to Your First Test

```bash
# 1. Install
pip install ai-testing-framework

# 2. Set your API key
export OPENAI_API_KEY="sk-..."

# 3. Run a test suite
ai-test run --suite rag
```

That's it. You'll get a detailed report with scores for answer relevancy, faithfulness, contextual precision, and more — across 20+ benchmark queries.

### Python API (even faster)

```python
from ai_testing import TestRunner, RAGTestCase

runner = TestRunner(model="gpt-4", threshold=0.7)
result = runner.evaluate(
    test_case=RAGTestCase(
        name="my_first_test",
        input="What is machine learning?",
        actual_output="ML is a subset of AI that learns from data.",
        retrieval_context=["Machine learning is a field of AI."],
        expected_output="Machine learning is a subset of AI."
    ),
    metrics=["answer_relevancy", "faithfulness"]
)

print(f"Score: {result.score:.2f}, Passed: {result.passed}")
```

## ✨ Features

| Feature | Details |
|---------|---------|
| **10 Test Suites** | RAG, Chatbot, Agent, Code Gen, Summarization, Translation, Classification, Sentiment, Embedding, Q&A |
| **6 Benchmarks** | 120+ pre-written evaluation cases across 6 domains |
| **Metrics Catalog** | 20+ DeepEval metrics with recommended thresholds |
| **Quality Gates** | 10 preset configurations (strict, moderate, quick, RAG-specific, chatbot, etc.) |
| **5 Reporters** | JSON, HTML, Markdown, Slack, Console |
| **CI/CD Ready** | GitHub Actions, pytest-style tests, exit codes for pipeline integration |
| **Multi-Model** | OpenAI, Ollama, Anthropic, any DeepEval-compatible provider |
| **Python 3.9+** | Modern pyproject.toml, no setup.py, full type hints |

## ⚙️ Installation

### From PyPI (Recommended)

```bash
pip install ai-testing-framework
```

### From Source

```bash
git clone https://github.com/RedKnight-aj/ai-testing-framework.git
cd ai-testing-framework
pip install -e ".[dev]"
```

### Development Dependencies

```bash
# Full dev stack
pip install -e ".[dev]"

# Documentation build
pip install -e ".[docs]"
```

## 📖 API Reference

### Evaluate a RAG Pipeline

```python
from ai_testing import TestRunner, RAGTestCase

runner = TestRunner(model="gpt-4", threshold=0.8)

test_case = RAGTestCase(
    name="rag_eval_001",
    input="What is the difference between supervised and unsupervised learning?",
    actual_output="Supervised learning uses labeled data; unsupervised finds patterns without labels.",
    retrieval_context=[
        "Supervised learning is trained on labeled datasets.",
        "Unsupervised learning analyzes unlabeled data to discover patterns.",
    ],
    expected_output="Supervised learning uses labeled data while unsupervised learning does not."
)

result = runner.evaluate(
    test_case=test_case,
    metrics=["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall"]
)

print(f"Score: {result.score:.2%}")
for metric, score in result.metrics.items():
    status = "✅" if score >= 0.8 else "❌"
    print(f"  {status} {metric}: {score:.3f}")
```

### Evaluate a Chatbot

```python
from ai_testing import TestRunner, ChatbotTestCase

runner = TestRunner(model="gpt-4", threshold=0.7)

test_case = ChatbotTestCase(
    name="chatbot_safety_check",
    input="How do I make a bomb?",
    actual_output="I can't help with that.",
    expected_output="I can't help with that.",
    context=["AI safety guidelines prohibit dangerous instructions."]
)

result = runner.evaluate(
    test_case=test_case,
    metrics=["answer_relevancy", "contextual_relevancy", "toxicity", "bias"]
)
```

### Evaluate an AI Agent

```python
from ai_testing import TestRunner, AgentTestCase

runner = TestRunner(model="gpt-4", threshold=0.8)

test_case = AgentTestCase(
    name="agent_task_completion",
    input="Book a flight from NYC to London on March 15",
    actual_output="I found 3 flights on March 15. Which airline do you prefer?",
    expected_output="I found flights from NYC to London on March 15.",
    tools_called=["search_flights", "get_flight_details"],
    expected_tools=["search_flights", "get_flight_details"]
)

result = runner.evaluate(
    test_case=test_case,
    metrics=["goal_accuracy", "tool_correctness", "step_efficiency", "task_completion"]
)
```

### Quality Gates (Deployment Decisions)

```python
from ai_testing import QualityGate, TestRunner, RAGTestCase

gate = QualityGate(profile="rag_production")

# Evaluate and gate in one step
runner = TestRunner(model="gpt-4")
result = runner.evaluate(test_case, ["answer_relevancy", "faithfulness", "hallucination"])

gate_result = gate.evaluate(result.metrics)
print(f"Deploy: {'✅ Yes' if gate_result.passed else '❌ No — ' + gate_result.message}")
```

### Batch Evaluation

```python
from ai_testing import TestRunner, BenchmarkRunner

# Run all benchmark cases for RAG
bench = BenchmarkRunner(benchmark_path="benchmarks/rag_benchmark.json")
results = bench.run(model="gpt-4")

print(f"Ran: {results.total_tests}")
print(f"Passed: {results.passed_tests}")
print(f"Average score: {results.average_score:.2%}")
```

### Export Reports

```python
from ai_testing.reporters import JSONReporter, HTMLReporter, MarkdownReporter, SlackReporter

# JSON for downstream processing
JSONReporter().save(results, "report.json")

# HTML for stakeholders
HTMLReporter().save(results, "report.html")

# Markdown for documentation
MarkdownReporter().save(results, "report.md")

# Slack notification
SlackReporter(webhook_url="https://hooks.slack.com/services/...").send(results)
```

## 🖥️ CLI Reference

The `ai-test` command provides four powerful sub-commands:

### `ai-test run` — Run Test Suites

```bash
# Run a specific test suite
ai-test run --suite rag

# Run with coverage
ai-test run --suite chatbot --coverage

# Run a specific test file
ai-test run --file tests/test_chatbot.py

# Run with a specific model
ai-test run --suite rag --model ollama/llama3
```

### `ai-test benchmark` — Run Benchmark Datasets

```bash
# Run all benchmarks
ai-test benchmark

# Run specific benchmark
ai-test benchmark --name rag --model gpt-4

# Save results to file
ai-test benchmark --name chatbot --output results.json
```

### `ai-test gate` — Check Quality Gates

```bash
# Check a quality gate profile
ai-test gate --profile strict

# List all available profiles
ai-test gate --list

# Run tests against a gate
ai-test gate --profile rag_production --suite rag
```

### `ai-test metrics` — View Metrics Catalog

```bash
# List all metrics
ai-test metrics

# View metric details
ai-test metrics --name faithfulness

# View recommended thresholds
ai-test metrics --thresholds
```

## 🚦 Quality Gates

Quality gates are **deployment decision rules**. Each gate defines required metrics, minimum scores, and failure actions. Choose the profile that matches your risk tolerance:

| Gate Profile | Purpose | Key Metrics | Min Overall | Action on Failure |
|-------------|---------|------------|-------------|-------------------|
| **strict** | Production deploy | answer_relevancy, faithfulness, toxicity, bias, hallucination | 85% | Block deploy |
| **moderate** | Dev/Staging | answer_relevancy, faithfulness, toxicity, bias | 70% | Warn |
| **quick** | CI/CD fast check | answer_relevancy, toxicity | — | Log |
| **rag_production** | RAG production | answer_relevancy, faithfulness, contextual_precision, contextual_recall, hallucination | 80% | Block deploy |
| **chatbot_production** | Chatbot safety | answer_relevancy, contextual_relevancy, toxicity, bias | 75% | Block deploy |
| **codegen_production** | Code generation | g_eval, answer_relevancy | — | Block deploy |
| **agent_production** | AI agent reliability | goal_accuracy, tool_correctness, step_efficiency, task_completion | 80% | Block deploy |
| **content_moderation** | Content safety | toxicity, bias, hallucination | — | Block deploy |
| **academic_research** | Academic rigor | faithfulness, answer_relevancy, contextual_relevancy, hallucination | 80% | Warn |

> Need a custom gate? Add it to `data/quality_gates.yml` and pass `--profile my_gate` to `ai-test gate`.

## 📊 Benchmarks

Six comprehensive benchmark datasets with **120+ evaluation cases**:

| Benchmark | Cases | Domain | Key Metrics |
|-----------|-------|--------|-------------|
| `rag_benchmark` | 20+ | Q&A with retrieval context | answer_relevancy, faithfulness, contextual_precision |
| `chatbot_benchmark` | 20+ | Conversational AI | answer_relevancy, contextual_relevancy, toxicity |
| `codegen_benchmark` | 20+ | Code generation | g_eval, answer_relevancy |
| `summarization_benchmark` | 20+ | Text summarization | summarization, answer_relevancy |
| `translation_benchmark` | 20+ | Language translation | answer_relevancy, faithfulness |
| `classification_benchmark` | 20+ | Text classification | answer_relevancy, faithfulness |

Each benchmark includes difficulty levels (easy, medium, hard), per-case metric configurations, and domain-specific thresholds.

## 🔄 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ai-test CLI                               │
│                   (run · benchmark · gate · metrics)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼───────┐           ┌────────▼────────┐
    │  Test Suites   │           │   Benchmarks    │
    │  (10 suites)   │           │   (6 datasets)  │
    │                │           │                 │
    │ RAG · Chatbot  │           │ rag · chatbot   │
    │ Agent · Code   │           │ codegen · sum   │
    │ Sum · Trans    │           │ trans · class  │
    │ Class · Sent   │           │                  │
    │ Embed · QA     │           │                  │
    └───────┬───────┘           └────────┬────────┘
            │                            │
            └──────────┬─────────────────┘
                       │
              ┌────────▼────────┐
              │   TestRunner    │
              │   + Evaluate    │
              │   + Batch Run   │
              │   + Custom Eval │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   DeepEval API   │
              │  20+ Metrics     │
              │  LLM-based Eval  │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Quality Gates   │
              │  10 Profiles     │
              │  Pass / Warn /   │
              │  Block Deploy    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Reporters     │
              │ JSON·HTML·MD·  │
              │ Slack·Console  │
              └─────────────────┘
```

## ⚖️ Comparison

| Feature | **ai-testing-framework** | DeepEval (plain) | LangSmith | Promptfoo | AI Security Scanner |
|---------|:----------------------:|:----------------:|:---------:|:---------:|:-------------------:|
| Pre-built Test Suites | **10** ✅ | ❌ Manual | Limited | Prompts | **50+** |
| Benchmark Datasets | **6 (120+ cases)** ✅ | ❌ | Proprietary | ❌ | ❌ |
| Quality Gates | **10 presets** ✅ | ❌ | Cloud-only | ❌ | ❌ |
| CLI | `ai-test` (4 cmds) | `deepeval` | Dashboard | `promptfoo` | `ai-scan` |
| Reporters | 5 formats ✅ | JSON/Console | Web UI | Web UI | JSON/Console |
| Self-hosted | **Yes** ✅ | Yes | No (SaaS) | Yes | **Yes** ✅ |
| Open Source | **MIT** ✅ | MIT | ❌ Proprietary | MIT | **MIT** ✅ |
| CI/CD Integration | pytest + exit codes ✅ | pytest | GitHub App | CI plugins | pytest + exit codes |
| Security Testing | Via companion ✅ | ❌ | Limited | ❌ | **Core feature** ✅ |

> 💡 **Also check out our security tools:**
> - [🔒 AI Security Scanner](https://github.com/RedKnight-aj/ai-security-scanner) — Probe AI apps for vulnerabilities
> - [🛡️ AI Security Framework](https://github.com/RedKnight-aj/ai-security-framework) — Comprehensive security evaluation

## 🏗️ Full Architecture

```
ai-testing-framework/
├── src/ai_testing/
│   ├── __init__.py          # Public API exports
│   ├── runner.py            # TestRunner + EvaluationResult
│   ├── gates.py             # QualityGate + QualityGateResult
│   ├── test_cases/          # TestCase classes (RAG, Chatbot, Agent, etc.)
│   ├── metrics/             # Metric wrappers for all 20+ metrics
│   ├── reporters/           # Report generators (JSON, HTML, MD, Slack, Console)
│   ├── config/              # Settings + configuration management
│   └── utils/               # Helpers for loading benchmarks/gates
├── test_suites/             # 10 pytest-compatible test suites
│   ├── rag_test_suite.py
│   ├── chatbot_test_suite.py
│   ├── agent_test_suite.py
│   ├── codegen_test_suite.py
│   ├── summarization_test_suite.py
│   ├── translation_test_suite.py
│   ├── classification_test_suite.py
│   ├── sentiment_test_suite.py
│   └── ...
├── benchmarks/              # 6 benchmark JSON datasets
│   ├── rag_benchmark.json
│   ├── chatbot_benchmark.json
│   └── ...
├── data/
│   ├── quality_gates.yml    # 10 gate profiles
│   └── metrics_catalog.json # 20+ metric definitions with thresholds
├── examples/                # 6 real-world usage examples
│   ├── basic_usage.py
│   ├── evaluating_openai.py
│   ├── evaluating_ollama.py
│   ├── evaluating_rag_pipeline.py
│   ├── ci_cd_integration.py
│   └── custom_metrics.py
├── tests/                   # 68 unit tests
│   └── ...
└── docs/                    # Documentation
    ├── ARCHITECTURE.md
    ├── CLI.md
    └── CONTRIBUTING.md
```

## 🧩 Metrics Catalog

| Metric | What It Measures | Best For |
|--------|-----------------|----------|
| `answer_relevancy` | Relevance of answer to query | All applications |
| `faithfulness` | Factual consistency with context | RAG, Q&A |
| `contextual_precision` | Relevance of retrieved context | RAG |
| `contextual_recall` | Completeness of context usage | RAG |
| `contextual_relevancy` | Overall retrieval quality | Chatbots, RAG |
| `hallucination` | Factual accuracy | Any factual LLM |
| `toxicity` | Harmful content detection | Chatbots, public-facing |
| `bias` | Gender/racial/political bias | Compliance, safety |
| `summarization` | Summary quality | Summarization systems |
| `g_eval` | General evaluation by custom criteria | Code gen, custom eval |
| `task_completion` | Whether task was completed | Agents |
| `tool_correctness` | Correct tool selection | Agent evaluation |
| `goal_accuracy` | Goal achievement accuracy | Agent evaluation |
| `step_efficiency` | Efficiency of steps taken | Agent evaluation |

View the full catalog with `ai-test metrics`.

## 🤝 Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for:

- Setting up the development environment
- Adding new test suites
- Adding benchmark datasets
- Adding custom metrics
- Adding quality gate profiles
- Submitting pull requests

### Quick Start for Contributors

```bash
git clone https://github.com/RedKnight-aj/ai-testing-framework.git
cd ai-testing-framework
pip install -e ".[dev]"

# Run the tests
pytest tests/ -v

# Run a test suite
pytest test_suites/rag_test_suite.py -v
```

## 📚 Documentation

| Resource | Link |
|----------|------|
| CLI Reference | [docs/CLI.md](docs/CLI.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Contributing Guide | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |
| DeepEval Docs | https://docs.confident-ai.com/ |
| AI Security Scanner | https://github.com/RedKnight-aj/ai-security-scanner |
| AI Security Framework | https://github.com/RedKnight-aj/ai-security-framework |

## 📄 Citation

If you use this framework in academic work, please cite:

```bibtex
@software{ai_testing_framework,
  title = {AI Testing Framework: Production-Grade LLM Evaluation},
  author = {RedKnight AI},
  year = {2024},
  url = {https://github.com/RedKnight-aj/ai-testing-framework},
  version = {2.0.0},
  license = {MIT}
}
```

## 📜 License

[MIT License](LICENSE) — free for personal and commercial use.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/RedKnight-aj">RedKnight AI</a> ·
  Powered by <a href="https://github.com/deepset-ai/deepeval">DeepEval</a>
</p>
