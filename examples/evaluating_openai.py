"""
Evaluating OpenAI GPT Models
============================
Example of evaluating OpenAI GPT models with the ai-testing-framework.
"""

import os
from openai import OpenAI
from ai_testing.test_cases import BaseTestCase
from ai_testing.runner import TestRunner

def evaluate_openai_response():
    # Set up OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define test prompt
    test_prompt = "Explain quantum computing in simple terms."

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": test_prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    actual_output = response.choices[0].message.content

    # Create test case
    test_case = BaseTestCase(
        name="openai_gpt4_evaluation",
        input=test_prompt,
        actual_output=actual_output,
        expected_output="Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, potentially solving complex problems much faster than classical computers."
    )

    # Initialize framework runner
    runner = TestRunner(
        model="gpt-4",  # Using GPT-4 for evaluation
        threshold=0.7
    )

    # Evaluate the response
    result = runner.evaluate(
        test_case=test_case,
        metrics=["answer_relevancy", "faithfulness", "g_eval"]
    )

    print("OpenAI GPT-4 Evaluation Results:")
    print(f"Prompt: {test_prompt}")
    print(f"Response: {actual_output[:100]}...")
    print(f"Overall Score: {result.score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"Answer Relevancy: {result.metrics.get('answer_relevancy', 'N/A'):.2f}")
    print(f"Faithfulness: {result.metrics.get('faithfulness', 'N/A'):.2f}")
    print(f"G-Eval Score: {result.metrics.get('g_eval', 'N/A'):.2f}")

    return result

def batch_evaluate_models():
    """Compare multiple OpenAI models on the same prompt."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    test_prompt = "What are the benefits of renewable energy?"
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]

    results = {}

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=150,
                temperature=0.3
            )

            actual_output = response.choices[0].message.content

            test_case = BaseTestCase(
                name=f"{model}_evaluation",
                input=test_prompt,
                actual_output=actual_output,
                expected_output="Renewable energy provides clean power, reduces carbon emissions, creates jobs, enhances energy security, and offers long-term cost savings."
            )

            runner = TestRunner(model="gpt-4", threshold=0.7)
            result = runner.evaluate(test_case, ["answer_relevancy", "faithfulness"])

            results[model] = {
                "score": result.score,
                "passed": result.passed,
                "metrics": result.metrics
            }

        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            results[model] = {"error": str(e)}

    # Print comparison
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model, result in results.items():
        if "error" not in result:
            print(f"{model}:")
            print(".2f")
            print(f"  Passed: {result['passed']}")
            print()

    return results

if __name__ == "__main__":
    # Single model evaluation
    evaluate_openai_response()

    # Multi-model comparison
    batch_evaluate_models()