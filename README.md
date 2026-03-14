# AI Testing Framework 🧪

> Production-ready LLM evaluation framework using DeepEval

[![DeepEval](https://img.shields.io/badge/Powered%20by-DeepEval-blue)](https://github.com/confident-ai/deepeval)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Professional AI testing framework for evaluating LLM applications. Built on DeepEval with best practices.

## Features

- 🎯 **50+ Metrics** - RAG, Agent, Chatbot, Multimodal evaluation
- 🔄 **pytest-style** - Familiar testing patterns
- 🔗 **Framework Integrations** - LangChain, LlamaIndex, OpenAI Agents
- ☁️ **Cloud Platform** - Confident AI integration
- 📊 **Multiple Reports** - JSON, HTML, Slack
- 🚀 **CI/CD Ready** - GitHub Actions integration

## Installation

```bash
# Core package
pip install -U deepeval

# Optional: For cloud features
deepeval login

# Install framework
pip install ai-testing-framework
```

## Quick Start

### 1. Basic Test

```python
from ai_testing_framework import TestRunner, Metric
from ai_testing_framework.test_cases import RAGTestCase

# Define test case
test_case = RAGTestCase(
    input="What is AI?",
    expected_output="AI stands for Artificial Intelligence",
    retrieval_context=["AI is Artificial Intelligence"]
)

# Run evaluation
runner = TestRunner()
result = runner.evaluate(test_case, metrics=["answer_relevancy", "faithfulness"])
print(result.score)
```

### 2. RAG Pipeline Test

```python
from ai_testing_framework import RAGTester

tester = RAGTester(model="gpt-4")

# Test entire RAG pipeline
results = tester.evaluate_pipeline(
    queries=["What is machine learning?", "Explain neural networks"],
    expected_outputs=["...", "..."],
    retrieval_contexts=[["..."], ["..."]]
)

# Get detailed metrics
print(results.get_metrics())
```

### 3. Agent Evaluation

```python
from ai_testing_framework import AgentTester

tester = AgentTester()

# Test agent task completion
result = tester.evaluate_task_completion(
    task="Book a flight from NYC to London",
    actual_output="I found flights...",
    steps_taken=["search_flights", "select_flight", "confirm_booking"]
)
```

## Architecture

```
ai-testing-framework/
├── src/
│   └── ai_testing/
│       ├── __init__.py          # Main exports
│       ├── runner.py            # Test execution engine
│       ├── test_cases/          # Reusable test cases (Page Objects)
│       │   ├── __init__.py
│       │   ├── base.py          # BaseTestCase
│       │   ├── rag.py           # RAG test cases
│       │   ├── chatbot.py       # Chatbot test cases
│       │   └── agent.py         # Agent test cases
│       ├── metrics/             # Metric wrappers
│       │   ├── __init__.py
│       │   ├── rag_metrics.py   # RAG metrics
│       │   ├── agent_metrics.py # Agent metrics
│       │   └── custom.py        # Custom GEval
│       ├── fixtures/            # Test fixtures & data
│       │   ├── __init__.py
│       │   ├── conftest.py      # Pytest fixtures
│       │   └── test_data.py     # Sample data
│       ├── reporters/           # Report generators
│       │   ├── __init__.py
│       │   ├── json_reporter.py
│       │   ├── html_reporter.py
│       │   └── slack_reporter.py
│       └── config/              # Configuration
│           ├── __init__.py
│           └── settings.py       # Config management
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── configs/                    # Config files
└── docs/                       # Documentation
```

## Available Metrics

### RAG Metrics
| Metric | Description |
|--------|-------------|
| `answer_relevancy` | How relevant is the answer to the query? |
| `faithfulness` | Is answer grounded in retrieval context? |
| `contextual_precision` | Are relevant chunks ranked higher? |
| `contextual_recall` | Does retrieval include needed information? |
| `contextual_relevancy` | Overall retrieval relevance |

### Agent Metrics
| Metric | Description |
|--------|-------------|
| `task_completion` | Did the agent complete the task? |
| `tool_correctness` | Were correct tools called? |
| `goal_accuracy` | Was the goal achieved accurately? |
| `step_efficiency` | Were unnecessary steps taken? |
| `plan_adherence` | Did agent follow the plan? |

### General Metrics
| Metric | Description |
|--------|-------------|
| `hallucination` | Factually correct output? |
| `toxicity` | Harmful content detection |
| `bias` | Gender/racial/political bias |
| `summarization` | Summary quality |
| `json_correctness` | Valid JSON output? |

## Configuration

### Using Configuration File

```python
from ai_testing_framework.config import Settings

settings = Settings(
    model="gpt-4",
    threshold=0.7,
    cloud_enabled=True
)
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export DEEPEVAL_API_KEY="..."
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/ai-test.yml
name: AI Testing

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -U deepeval
          pip install ai-testing-framework
      
      - name: Run tests
        run: deepeval test run tests/
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: results/
```

## Examples

See [`examples/`](examples/) for complete examples:

- `examples/rag_evaluation.py` - RAG pipeline testing
- `examples/chatbot_evaluation.py` - Chatbot testing
- `examples/agent_evaluation.py` - Agent testing
- `examples/custom_metrics.py` - Custom GEval metrics

## Reporting

### JSON Report

```python
from ai_testing_framework.reporters import JSONReporter

reporter = JSONReporter()
reporter.save(results, "report.json")
```

### HTML Report

```python
from ai_testing_framework.reporters import HTMLReporter

reporter = HTMLReporter()
reporter.save(results, "report.html")
```

### Slack Notification

```python
from ai_testing_framework.reporters import SlackReporter

reporter = SlackReporter(webhook_url="https://hooks.slack.com/...")
reporter.send(results)
```

## Development

```bash
# Clone repository
git clone https://github.com/RedKnight-aj/ai-testing-framework.git
cd ai-testing-framework

# Install development dependencies
pip install -e ".[dev]"

# Run tests
deepeval test run tests/

# Run with coverage
pytest --cov=src tests/
```

## Documentation

- [DeepEval Docs](https://docs.confident-ai.com/)
- [Metric Reference](https://deepeval.com/docs/metrics)
- [Confident AI Platform](https://app.confident-ai.com)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**RedKnight AI** - [GitHub](https://github.com/RedKnight-ai)

---

Built with ❤️ using [DeepEval](https://github.com/confident-ai/deepeval)
