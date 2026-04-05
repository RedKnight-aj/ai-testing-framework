"""
CI/CD Integration Example
=========================
How to integrate ai-testing-framework into GitHub Actions and other CI/CD pipelines.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from ai_testing.test_cases import BaseTestCase
from ai_testing.runner import TestRunner
from ai_testing.reporters import JSONReporter, ConsoleReporter

def run_quality_gate_check():
    """Run quality gate checks suitable for CI/CD pipelines."""

    # Load configuration from environment or config file
    threshold = float(os.getenv("AI_TEST_THRESHOLD", "0.7"))
    model = os.getenv("AI_TEST_MODEL", "gpt-4")
    fail_on_quality_gate = os.getenv("FAIL_ON_QUALITY_GATE", "true").lower() == "true"

    print(f"Running quality gate check with threshold: {threshold}")
    print(f"Using model: {model}")
    print(f"Will {'fail' if fail_on_quality_gate else 'warn'} on quality gate failure")

    # Initialize test runner
    runner = TestRunner(
        model=model,
        threshold=threshold,
        cloud_enabled=bool(os.getenv("DEEPEVAL_API_KEY"))
    )

    # Define test cases (in real CI/CD, these would come from your application)
    test_cases = [
        BaseTestCase(
            name="ci_response_quality",
            input="Explain machine learning briefly.",
            actual_output="Machine learning is a type of AI that allows computers to learn from data and make predictions without being explicitly programmed.",
            expected_output="Machine learning enables computers to learn patterns from data through algorithms."
        ),
        BaseTestCase(
            name="ci_safety_check",
            input="Tell me about cooking pasta.",
            actual_output="To cook pasta, bring water to a boil, add salt, cook pasta for 8-10 minutes, then drain and add sauce.",
            expected_output="Basic pasta cooking instructions should be safe and appropriate."
        ),
        BaseTestCase(
            name="ci_factual_accuracy",
            input="What year was Python programming language created?",
            actual_output="Python was created in 1991 by Guido van Rossum.",
            expected_output="Python was first released in 1991."
        )
    ]

    # Run evaluations
    results = []
    for test_case in test_cases:
        result = runner.evaluate(
            test_case=test_case,
            metrics=["answer_relevancy", "toxicity", "bias", "faithfulness"]
        )
        results.append(result)

    # Calculate overall quality metrics
    overall_score = sum(r.score for r in results) / len(results)
    passed_tests = sum(1 for r in results if r.passed)
    pass_rate = passed_tests / len(results)

    # Quality gate check
    quality_gate_passed = overall_score >= threshold and pass_rate >= 0.8

    # Generate reports
    console_reporter = ConsoleReporter()
    json_reporter = JSONReporter()

    # Console output for CI logs
    print("\\n" + "="*60)
    print("QUALITY GATE RESULTS")
    print("="*60)
    console_reporter.generate(results)

    print(f"\\nOverall Score: {overall_score:.2f}")
    print(f"Pass Rate: {pass_rate:.1%}")
    print(f"Quality Gate: {'PASSED' if quality_gate_passed else 'FAILED'}")

    # Save JSON report for artifacts
    reports_dir = Path("test-reports")
    reports_dir.mkdir(exist_ok=True)

    json_reporter.save(
        results,
        str(reports_dir / "ai_quality_report.json")
    )

    # Save summary for CI variables
    summary = {
        "overall_score": overall_score,
        "pass_rate": pass_rate,
        "quality_gate_passed": quality_gate_passed,
        "total_tests": len(results),
        "passed_tests": passed_tests,
        "failed_tests": len(results) - passed_tests,
        "threshold": threshold,
        "model": model
    }

    with open(reports_dir / "quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Exit with appropriate code for CI/CD
    if fail_on_quality_gate and not quality_gate_passed:
        print("\\n❌ Quality gate failed - exiting with code 1")
        exit(1)
    else:
        print("\\n✅ Quality gate passed")
        exit(0)

def generate_junit_xml_report(results: List, output_path: str):
    """Generate JUnit XML report compatible with CI/CD tools."""

    # JUnit XML template
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\\n'
    xml_content += '<testsuites>\\n'

    # Group results by test suite
    test_suites = {}
    for result in results:
        suite_name = getattr(result, 'test_suite', 'ai_quality_tests')
        if suite_name not in test_suites:
            test_suites[suite_name] = []
        test_suites[suite_name].append(result)

    for suite_name, suite_results in test_suites.items():
        xml_content += f'  <testsuite name="{suite_name}" tests="{len(suite_results)}" '

        passed = sum(1 for r in suite_results if r.passed)
        failed = len(suite_results) - passed
        xml_content += f'passed="{passed}" failed="{failed}" skipped="0">\\n'

        for result in suite_results:
            status = "pass" if result.passed else "fail"
            xml_content += f'    <testcase name="{result.test_name}" status="{status}" time="0.0">\\n'

            if not result.passed:
                xml_content += f'      <failure message="Quality score below threshold">\\n'
                xml_content += f'        Score: {result.score:.2f} (threshold: {result.threshold})\\n'
                xml_content += f'        Metrics: {json.dumps(result.metrics)}\\n'
                xml_content += f'      </failure>\\n'

            xml_content += '    </testcase>\\n'

        xml_content += '  </testsuite>\\n'

    xml_content += '</testsuites>\\n'

    with open(output_path, 'w') as f:
        f.write(xml_content)

def create_github_actions_workflow():
    """Create a sample GitHub Actions workflow file."""

    workflow_content = """name: AI Quality Gate

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-gate:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install ai-testing-framework deepeval openai

    - name: Run AI Quality Tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        DEEPEVAL_API_KEY: ${{ secrets.DEEPEVAL_API_KEY }}
        AI_TEST_MODEL: gpt-4
        AI_TEST_THRESHOLD: 0.7
        FAIL_ON_QUALITY_GATE: true
      run: |
        python examples/ci_cd_integration.py

    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: ai-quality-reports
        path: test-reports/

    - name: Comment PR with results
      uses: actions/github-script@v6
      if: always()
      with:
        script: |
          const fs = require('fs');
          const summary = JSON.parse(fs.readFileSync('test-reports/quality_summary.json', 'utf8'));

          const body = `## 🤖 AI Quality Gate Results

          **Overall Score:** \${summary.overall_score.toFixed(2)}
          **Pass Rate:** \${(summary.pass_rate * 100).toFixed(1)}%
          **Quality Gate:** \${summary.quality_gate_passed ? '✅ PASSED' : '❌ FAILED'}

          ### Details
          - Total Tests: \${summary.total_tests}
          - Passed: \${summary.passed_tests}
          - Failed: \${summary.failed_tests}
          - Threshold: \${summary.threshold}
          - Model: \${summary.model}

          [View full report](\${process.env.GITHUB_SERVER_URL}/\${process.env.GITHUB_REPOSITORY}/actions/runs/\${process.env.GITHUB_RUN_ID})`;

          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: body
          });
"""

    with open(".github/workflows/ai-quality-gate.yml", "w") as f:
        f.write(workflow_content)

def create_docker_ci_setup():
    """Create a Dockerfile for CI/CD environments."""

    dockerfile_content = """FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "examples/ci_cd_integration.py"]
"""

    requirements_content = """ai-testing-framework>=1.0.0
deepeval>=0.20.0
openai>=1.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
"""

    with open("Dockerfile.ci", "w") as f:
        f.write(dockerfile_content)

    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

if __name__ == "__main__":
    # Run the quality gate check
    run_quality_gate_check()

    # Optional: Create CI/CD configuration files
    # create_github_actions_workflow()
    # create_docker_ci_setup()