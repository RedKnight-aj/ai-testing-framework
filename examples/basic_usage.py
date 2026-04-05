"""
Basic Usage Example
===================
Simple evaluation with inline code using the ai-testing-framework.
"""

from ai_testing.test_cases import BaseTestCase
from ai_testing.runner import TestRunner

def main():
    # Create a simple test case
    test_case = BaseTestCase(
        name="basic_example",
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )

    # Initialize the test runner
    runner = TestRunner(
        model="gpt-4",  # or your preferred model
        threshold=0.5
    )

    # Run evaluation
    result = runner.evaluate(
        test_case=test_case,
        metrics=["answer_relevancy", "faithfulness"]
    )

    # Print results
    print("Evaluation Results:")
    print(f"Test: {result.test_name}")
    print(f"Score: {result.score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"Metrics: {result.metrics}")

    # Expected output:
    # Evaluation Results:
    # Test: basic_example
    # Score: 0.95
    # Passed: True
    # Metrics: {'answer_relevancy': 0.95, 'faithfulness': 0.95}

if __name__ == "__main__":
    main()