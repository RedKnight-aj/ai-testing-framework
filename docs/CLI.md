# CLI Reference

> Complete reference for the `ai-test` command-line interface

## Overview

The `ai-test` CLI is your primary interface for evaluating AI applications. It provides four sub-commands:

```
ai-test <command> [options]

Commands:
  run         Run test suites
  benchmark   Run benchmark datasets
  gate        Check quality gates
  metrics     View metrics catalog
```

## `ai-test run`

Run pre-built test suites with DeepEval metrics.

### Synopsis

```bash
ai-test run --suite <suite_name> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--suite` | Test suite to run | *(required)* |
| `--model` | LLM model for evaluation | `"gpt-4"` |
| `--threshold` | Minimum pass threshold | `0.5` |
| `--file` | Run specific test file | — |
| `--coverage` | Run with pytest coverage | `False` |
| `--output` | Output file path | — |
| `--reporter` | Reporter type (json/html/md/slack/console) | `console` |
| `--verbose` | Verbose output | `False` |
| `-v`, `-vv` | pytest verbosity (repeat for more detail) | — |

### Examples

**Run the RAG test suite:**

```bash
ai-test run --suite rag
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       AI Testing Framework — RAG         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Test: test_answer_relevancy_rag          ┃
┃   ✅ answer_relevancy:  0.942            ┃
┃   ✅ faithfulness:      0.891            ┃
┃   ✅ contextual_prec:   0.856            ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Tests passed: 12/12                      ┃
┃ Average score: 0.87                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Run with a specific model:**

```bash
ai-test run --suite chatbot --model ollama/llama3
```

**Run with lower threshold for quick feedback:**

```bash
ai-test run --suite agent --threshold 0.6
```

**Run a specific test file:**

```bash
ai-test run --file test_suites/rag_test_suite.py
```

**Run with coverage reporting:**

```bash
ai-test run --suite rag --coverage
```

**Run with HTML report output:**

```bash
ai-test run --suite rag --output report.html --reporter html
```

**Verbose pytest output:**

```bash
ai-test run --suite rag -vvv
```

### Available Suites

| `--suite` | What It Tests | Metrics Used |
|-----------|--------------|-------------|
| `rag` | RAG pipelines | answer_relevancy, faithfulness, contextual_precision, contextual_recall, hallucination |
| `chatbot` | Conversational AI | answer_relevancy, contextual_relevancy, toxicity, bias |
| `agent` | AI agents | goal_accuracy, tool_correctness, step_efficiency, task_completion |
| `codegen` | Code generation | g_eval, answer_relevancy |
| `summarization` | Text summarization | summarization, answer_relevancy |
| `translation` | Translation quality | answer_relevancy, faithfulness |
| `classification` | Classification accuracy | answer_relevancy, faithfulness |
| `sentiment` | Sentiment analysis | answer_relevancy, bias |
| `embedding` | Embedding quality | answer_relevancy, faithfulness |
| `qa` | Q&A systems | answer_relevancy, faithfulness, hallucination |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All tests passed, quality gate satisfied |
| `1` | One or more tests failed |
| `2` | Invalid arguments / configuration error |

---

## `ai-test benchmark`

Run benchmark datasets — curated evaluation cases with expected outputs and per-metric thresholds.

### Synopsis

```bash
ai-test benchmark --name <benchmark_name> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Benchmark to run | — |
| `--model` | LLM model for evaluation | `"gpt-4"` |
| `--output` | Save results to file | — |
| `--format` | Output format (json/md/console) | `console` |
| `--threshold` | Override per-case thresholds | *(uses benchmark defaults)* |
| `--verbose` | Detailed per-case output | `False` |

### Examples

**Run a specific benchmark:**

```bash
ai-test benchmark --name rag
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     Benchmark — RAG Dataset              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Total Cases:     20                      ┃
┃ Passed:          16                      ┃
┃ Failed:          4                       ┃
┃ Pass Rate:       80.0%                   ┃
┃ Avg Score:       0.84                    ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ [PASS] rag_001  answer_rel: 0.95  faith: 0.92
┃ [PASS] rag_002  answer_rel: 0.88  faith: 0.85
┃ [FAIL] rag_003  answer_rel: 0.62  faith: 0.71
┃ ...
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Run and save results:**

```bash
ai-test benchmark --name chatbot --output results_chatbot.json --format json
```

**Run all benchmarks:**

```bash
ai-test benchmark
```

**Verbose mode — see every case detail:**

```bash
ai-test benchmark --name codegen --verbose
```

### Available Benchmarks

| `--name` | Cases | Domain | Key Metrics |
|----------|-------|--------|-------------|
| `rag` | 20+ | Q&A with retrieval context | answer_relevancy, faithfulness, contextual_precision |
| `chatbot` | 20+ | Conversational AI | answer_relevancy, contextual_relevancy, toxicity |
| `codegen` | 20+ | Code generation | g_eval, answer_relevancy |
| `summarization` | 20+ | Text summarization | summarization, answer_relevancy |
| `translation` | 20+ | Language translation | answer_relevancy, faithfulness |
| `classification` | 20+ | Text classification | answer_relevancy, faithfulness |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All benchmark cases passed |
| `1` | One or more cases failed |
| `2` | Invalid arguments / benchmark file not found |

---

## `ai-test gate`

Check quality gates — deployment decision rules that evaluate whether your AI application meets production standards.

### Synopsis

```bash
ai-test gate --profile <profile_name> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--profile` | Quality gate profile | *(required)* |
| `--suite` | Run test suite before gate check | — |
| `--metrics` | Comma-separated metric names | — |
| `--list` | List all available profiles | `False` |
| `--output` | Save gate result to file | — |
| `--verbose` | Detailed gate evaluation output | `False` |

### Examples

**List all quality gate profiles:**

```bash
ai-test gate --list
```

Output:
```
Available Quality Gate Profiles:
  strict               — Production deployment gate (highest standards)
  moderate             — Development gate (balanced quality/speed)
  quick                — CI/CD quick check
  rag_production       — RAG production gate
  chatbot_production   — Chatbot safety gate
  codegen_production   — Code generation gate
  agent_production     — AI agent reliability gate
  content_moderation   — Content safety gate
  academic_research    — Academic rigor gate
```

**Check a quality gate profile:**

```bash
ai-test gate --profile strict
```

**Run a gate after a test suite (full CI/CD pipeline):**

```bash
ai-test gate --profile rag_production --suite rag
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     Quality Gate — rag_production        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Required Metrics:  5 checked             ┃
┃ answer_relevancy:  0.942  ✅ >= 0.85     ┃
┃ faithfulness:      0.891  ❌ >= 0.90     ┃
┃ ctx_precision:     0.856  ✅ >= 0.80     ┃
┃ ctx_recall:        0.812  ✅ >= 0.75     ┃
┃ hallucination:     0.080  ✅ <= 0.10     ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ ❌ GATE FAILED                           ┃
┃ faithfulness (0.891) below threshold (0.90)
┃ Action: BLOCK_DEPLOY                     ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Notify: slack, email                     ┃
┃ Message: RAG quality gate failed         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Available Profiles

| Profile | Use Case | Key Action |
|---------|----------|------------|
| `strict` | Production deploy | Block on any failure |
| `moderate` | Dev/staging | Warn on failure |
| `quick` | CI/CD pipeline | Log only |
| `rag_production` | RAG systems | Block on faithfulness < 0.90 |
| `chatbot_production` | Chatbots | Block on safety issues |
| `codegen_production` | Code generation | Block on quality |
| `agent_production` | AI agents | Block on reliability |
| `content_moderation` | Content safety | Block on toxicity |
| `academic_research` | Academic use | Warn on factual issues |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Gate passed — deployment is safe |
| `1` | Gate failed — action required (block/warn/log) |
| `2` | Invalid arguments / unknown profile |

---

## `ai-test metrics`

View the metrics catalog — all available evaluation metrics with descriptions, thresholds, and usage guidance.

### Synopsis

```bash
ai-test metrics [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--name` | Show details for specific metric | — |
| `--thresholds` | Show recommended thresholds | `False` |
| `--category` | Filter by category (rag/agent/safety/general) | — |
| `--format` | Output format (console/json) | `console` |

### Examples

**List all metrics:**

```bash
ai-test metrics
```

Output:
```
Available Metrics (20+):
  answer_relevancy         Measures how relevant the response is to the input
  faithfulness             Measures factual consistency with context
  contextual_precision     Evaluates if all retrieved context is relevant
  contextual_recall        Measures if all relevant context is used
  contextual_relevancy     Overall retrieval relevance
  hallucination            Detects if output is factually fabricated
  toxicity                 Detects harmful or offensive content
  bias                     Detects gender/racial/political bias
  summarization            Measures summary quality
  g_eval                   General evaluation with custom criteria
  task_completion          Did agent complete the task?
  tool_correctness         Were correct tools called?
  goal_accuracy            Was goal achieved accurately?
  step_efficiency          Were unnecessary steps taken?
```

**View metric details:**

```bash
ai-test metrics --name faithfulness
```

Output:
```
Metric: Faithfulness
Description: Measures whether the response is factually consistent with context
Score Range: 0.0 – 1.0
Recommended Thresholds:
  High Quality:  >= 0.90
  Acceptable:    >= 0.80
  Minimum:       >= 0.70
Best For: RAG systems, factual Q&A, knowledge-based responses
Cost: Medium (LLM-based with context comparison)
```

**Show all recommended thresholds:**

```bash
ai-test metrics --thresholds
```

Output:
```
Metric                    High Quality   Acceptable   Minimum
─────────────────────────────────────────────────────────────────
answer_relevancy          >= 0.90        >= 0.70      >= 0.50
faithfulness              >= 0.90        >= 0.80      >= 0.70
contextual_precision      >= 0.85        >= 0.70      >= 0.60
contextual_recall         >= 0.85        >= 0.70      >= 0.60
hallucination             <= 0.05        <= 0.15      <= 0.25
toxicity                  <= 0.05        <= 0.15      <= 0.30
bias                      <= 0.10        <= 0.20      <= 0.30
summarization             >= 0.85        >= 0.70      >= 0.60
g_eval                    >= 0.80        >= 0.70      >= 0.60
```

**Filter by category:**

```bash
ai-test metrics --category safety
ai-test metrics --category rag
ai-test metrics --category agent
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key (for evaluation) | ✅ If using OpenAI models |
| `DEEPEVAL_API_KEY` | Confident AI platform key | ❌ Optional |
| `AI_TEST_MODEL` | Default model for evaluations | ❌ Falls back to `gpt-4` |
| `AI_TEST_THRESHOLD` | Default pass threshold | ❌ Falls back to `0.5` |
| `AI_TEST_REPORTER` | Default reporter type | ❌ Falls back to `console` |
| `SLACK_WEBHOOK_URL` | Slack notifications URL | ❌ Only for Slack reporter |

## Exit Codes Summary

| Code | Meaning |
|------|---------|
| `0` | Success — all tests passed / gate cleared |
| `1` | Failure — tests failed or gate blocked |
| `2` | Error — invalid arguments, missing files, or misconfiguration |

## Chaining Commands

You can chain `ai-test` commands in CI/CD pipelines:

```bash
# Run tests, then check quality gate
ai-test run --suite rag && ai-test gate --profile rag_production

# Run benchmark, save results, send Slack notification
ai-test benchmark --name rag --output results.json && \
  ai-test run --suite rag --reporter slack

# Full CI pipeline
ai-test run --suite rag --coverage \
  && ai-test benchmark --name rag \
  && ai-test gate --profile strict
```

## Debugging

**Enable pytest tracebacks:**

```bash
ai-test run --suite rag --tb=long
```

**Run without DeepEval LLM calls (for CI dry-runs):**

```bash
# Mock mode not yet available — check docs for updates
```

**Verbose internal logging:**

```bash
export AI_TEST_DEBUG=1
ai-test run --suite rag -vvv
```
