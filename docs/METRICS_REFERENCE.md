# AI Testing Framework - Metrics Reference

This document provides a comprehensive reference for all evaluation metrics available in the ai-testing-framework.

## Table of Contents

- [Core Metrics](#core-metrics)
- [RAG-Specific Metrics](#rag-specific-metrics)
- [Safety and Quality Metrics](#safety-and-quality-metrics)
- [Agent Evaluation Metrics](#agent-evaluation-metrics)
- [Custom and Advanced Metrics](#custom-and-advanced-metrics)
- [Usage Guidelines](#usage-guidelines)

## Core Metrics

### Answer Relevancy

**Description:** Measures how relevant the response is to the input query or prompt.

**Score Range:** 0.0 - 1.0
- 0.0: Completely irrelevant response
- 1.0: Highly relevant and on-topic response

**Recommended Thresholds:**
- High quality: ≥ 0.9
- Acceptable: ≥ 0.7
- Minimum: ≥ 0.5

**When to Use:**
- All response evaluation scenarios
- Question-answering systems
- Conversational AI
- Search result relevance

**When NOT to Use:**
- Creative writing tasks
- When relevance is not the primary concern

**Implementation:** Semantic similarity between input and output using embeddings.

---

### Faithfulness

**Description:** Evaluates whether the response is factually consistent with provided context or known facts.

**Score Range:** 0.0 - 1.0
- 0.0: Contains major factual contradictions
- 1.0: Fully consistent with provided information

**Recommended Thresholds:**
- High quality: ≥ 0.9
- Acceptable: ≥ 0.8
- Minimum: ≥ 0.7

**When to Use:**
- RAG systems with retrieval context
- Factual question answering
- Citation-dependent responses
- Knowledge-based systems

**When NOT to Use:**
- Creative or hypothetical content
- Opinion-based responses

**Implementation:** Verifies claims against provided context using LLM evaluation.

---

## RAG-Specific Metrics

### Contextual Precision

**Description:** Evaluates if all retrieved context pieces are relevant and contribute to the answer.

**Score Range:** 0.0 - 1.0
- 0.0: Most retrieved context is irrelevant
- 1.0: All retrieved context is relevant and useful

**Recommended Thresholds:**
- High quality: ≥ 0.85
- Acceptable: ≥ 0.7
- Minimum: ≥ 0.6

**When to Use:**
- RAG systems
- Information retrieval systems
- Context-dependent generation

**When NOT to Use:**
- Non-retrieval based generation
- When context quality isn't being evaluated

---

### Contextual Recall

**Description:** Measures if all relevant information from context is used in the response.

**Score Range:** 0.0 - 1.0
- 0.0: Important context information is ignored
- 1.0: All relevant context is utilized

**Recommended Thresholds:**
- High quality: ≥ 0.9
- Acceptable: ≥ 0.75
- Minimum: ≥ 0.6

**When to Use:**
- RAG systems
- Comprehensive answer generation
- When completeness matters

**When NOT to Use:**
- Concise response requirements
- When brevity is prioritized

---

### Contextual Relevancy

**Description:** Assesses if the response is relevant given the conversation or task context.

**Score Range:** 0.0 - 1.0
- 0.0: Response ignores context completely
- 1.0: Response perfectly matches context requirements

**Recommended Thresholds:**
- High quality: ≥ 0.9
- Acceptable: ≥ 0.75
- Minimum: ≥ 0.6

**When to Use:**
- Conversational AI
- Multi-turn dialogues
- Context-dependent tasks

**When NOT to Use:**
- Single-turn, context-free tasks
- When context doesn't influence quality

---

## Safety and Quality Metrics

### Toxicity

**Description:** Detects harmful, offensive, or inappropriate content in responses.

**Score Range:** 0.0 - 1.0
- 0.0: Completely safe content
- 1.0: Highly toxic or harmful content

**Recommended Thresholds:**
- Safe: ≤ 0.1
- Acceptable: ≤ 0.2
- High risk: ≥ 0.5

**When to Use:**
- Content moderation
- User-generated content
- Public-facing AI responses
- Safety-critical applications

**When NOT to Use:**
- When offensive content is expected/allowed
- Creative or edgy content generation

---

### Bias Detection

**Description:** Identifies biased language, stereotypes, or unfair representations.

**Score Range:** 0.0 - 1.0
- 0.0: No detectable bias
- 1.0: Highly biased content

**Recommended Thresholds:**
- Fair: ≤ 0.1
- Concerning: ≤ 0.3
- Highly biased: ≥ 0.6

**When to Use:**
- Content generation
- Recommendation systems
- Hiring/recruitment AI
- Public policy AI

**When NOT to Use:**
- When bias analysis isn't relevant
- Highly domain-specific content

---

### Hallucination Detection

**Description:** Identifies fabricated information or claims not supported by evidence.

**Score Range:** 0.0 - 1.0
- 0.0: Fully factual and supported
- 1.0: Contains major fabrications

**Recommended Thresholds:**
- Factual: ≤ 0.1
- Some fabrication: ≤ 0.3
- Highly hallucinated: ≥ 0.7

**When to Use:**
- Factual content generation
- Knowledge-based QA
- Citation-dependent responses

**When NOT to Use:**
- Creative writing
- Hypothetical scenarios
- Speculative content

---

## Agent Evaluation Metrics

### Tool Correctness

**Description:** Evaluates if AI agent selects and uses tools appropriately.

**Score Range:** 0.0 - 1.0
- 0.0: Incorrect tool selection and usage
- 1.0: Perfect tool selection and execution

**Recommended Thresholds:**
- Correct: ≥ 0.9
- Acceptable: ≥ 0.7
- Needs improvement: ≥ 0.5

**When to Use:**
- AI agent evaluation
- Tool-using systems
- Function calling assessment

**When NOT to Use:**
- Non-agent systems
- Systems without tool use

---

### Step Efficiency

**Description:** Measures if AI agent completes tasks with optimal number of steps.

**Score Range:** 0.0 - 1.0
- 0.0: Highly inefficient (excessive steps)
- 1.0: Optimal step count

**Recommended Thresholds:**
- Optimal: ≥ 0.9
- Acceptable: ≥ 0.7
- Inefficient: ≥ 0.5

**When to Use:**
- AI agent evaluation
- Multi-step task completion
- Workflow optimization

**When NOT to Use:**
- Single-step tasks
- When efficiency isn't critical

---

### Goal Accuracy

**Description:** Assesses if AI agent achieves intended objectives.

**Score Range:** 0.0 - 1.0
- 0.0: Complete failure to achieve goals
- 1.0: Perfect goal achievement

**Recommended Thresholds:**
- Successful: ≥ 0.95
- Partial success: ≥ 0.7
- Failed: ≥ 0.3

**When to Use:**
- AI agent evaluation
- Goal-oriented systems
- Task completion assessment

**When NOT to Use:**
- When goals are subjective
- Exploratory or open-ended tasks

---

### Task Completion

**Description:** Measures if assigned tasks are fully completed successfully.

**Score Range:** 0.0 - 1.0
- 0.0: Task not completed
- 1.0: Fully completed successfully

**Recommended Thresholds:**
- Completed: ≥ 0.95
- Mostly completed: ≥ 0.8
- Incomplete: ≥ 0.4

**When to Use:**
- Task-oriented AI
- Workflow completion
- Assignment fulfillment

**When NOT to Use:**
- Open-ended creative tasks
- Exploratory activities

---

## Custom and Advanced Metrics

### G-Eval

**Description:** Custom evaluation using LLM-as-judge with specific criteria and rubrics.

**Score Range:** 0.0 - 1.0
- Customizable based on evaluation criteria

**Recommended Thresholds:**
- Excellent: ≥ 0.9
- Good: ≥ 0.8
- Acceptable: ≥ 0.7

**When to Use:**
- Custom quality criteria
- Complex evaluation requirements
- Domain-specific standards

**When NOT to Use:**
- When standard metrics suffice
- High-volume evaluation needs

**Cost:** High (requires LLM calls)

---

### Summarization Score

**Description:** Evaluates quality of text summarization including coverage, conciseness, and coherence.

**Score Range:** 0.0 - 1.0
- 0.0: Poor summarization quality
- 1.0: Excellent summarization

**Recommended Thresholds:**
- High quality: ≥ 0.85
- Acceptable: ≥ 0.7
- Minimum: ≥ 0.6

**When to Use:**
- Text summarization tasks
- Document condensation
- Key point extraction

**When NOT to Use:**
- When full detail is required
- Non-summarization tasks

---

## Usage Guidelines

### Choosing Metrics

1. **Start with Core Metrics:** Begin with answer relevancy and faithfulness for most use cases.

2. **Add Safety Metrics:** Always include toxicity and bias detection for user-facing applications.

3. **Domain-Specific Selection:**
   - RAG systems: Add contextual precision, recall, and relevancy
   - Agents: Include tool correctness, step efficiency, and goal accuracy
   - Summarization: Use summarization score
   - Safety-critical: Prioritize toxicity, bias, and hallucination detection

4. **Performance Considerations:**
   - Low cost: answer_relevancy, toxicity, step_efficiency
   - Medium cost: faithfulness, contextual metrics
   - High cost: G-Eval, hallucination detection

### Setting Thresholds

1. **Development Phase:** Use lower thresholds (0.6-0.7) to allow iteration.

2. **Staging/Testing:** Medium thresholds (0.7-0.8) for quality gates.

3. **Production:** High thresholds (0.8-0.9) for deployed systems.

4. **Critical Applications:** Very high thresholds (0.9+) with multiple safety metrics.

### Common Pitfalls

1. **Threshold Too High:** May reject good responses during development.

2. **Threshold Too Low:** May allow poor quality responses in production.

3. **Over-Reliance on Single Metric:** Use multiple complementary metrics.

4. **Ignoring Context:** Different domains may require different standards.

5. **Cost vs. Benefit:** Balance evaluation thoroughness with computational cost.

### Best Practices

1. **Baseline Establishment:** Run metrics on known good/bad examples to establish baselines.

2. **Iterative Refinement:** Adjust thresholds based on actual performance data.

3. **Monitoring:** Continuously monitor metric performance and adjust as needed.

4. **Documentation:** Clearly document which metrics and thresholds are used for each use case.

5. **A/B Testing:** Use metrics to compare different model versions or prompts.