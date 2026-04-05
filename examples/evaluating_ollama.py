"""
Evaluating Local Ollama Models
===============================
Example of evaluating local Ollama models with the ai-testing-framework.
"""

import requests
import json
from ai_testing.test_cases import BaseTestCase
from ai_testing.runner import TestRunner

def evaluate_ollama_model():
    """Evaluate a local Ollama model."""

    # Ollama API endpoint (default local installation)
    ollama_url = "http://localhost:11434/api/generate"

    test_prompt = "Explain the concept of recursion in programming."

    # Request to Ollama
    payload = {
        "model": "llama2",  # or your preferred model
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        }
    }

    try:
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()

        result = response.json()
        actual_output = result.get("response", "")

        # Create test case
        test_case = BaseTestCase(
            name="ollama_llama2_evaluation",
            input=test_prompt,
            actual_output=actual_output,
            expected_output="Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems."
        )

        # Initialize framework runner
        runner = TestRunner(
            model="gpt-4",  # Use GPT-4 for evaluation
            threshold=0.7
        )

        # Evaluate the response
        result = runner.evaluate(
            test_case=test_case,
            metrics=["answer_relevancy", "faithfulness"]
        )

        print("Ollama Model Evaluation Results:")
        print(f"Model: llama2")
        print(f"Prompt: {test_prompt}")
        print(f"Response length: {len(actual_output)} characters")
        print(f"Overall Score: {result.score:.2f}")
        print(f"Passed: {result.passed}")
        print(f"Answer Relevancy: {result.metrics.get('answer_relevancy', 'N/A'):.2f}")
        print(f"Faithfulness: {result.metrics.get('faithfulness', 'N/A'):.2f}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return None

def compare_ollama_models():
    """Compare multiple Ollama models."""

    models_to_test = ["llama2", "codellama", "mistral"]
    test_prompt = "Write a Python function to reverse a string."

    results = {}

    for model in models_to_test:
        try:
            payload = {
                "model": model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150
                }
            }

            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            actual_output = result.get("response", "")

            # Expected correct Python function
            expected_code = "def reverse_string(s):\\n    return s[::-1]"

            test_case = BaseTestCase(
                name=f"ollama_{model}_code_generation",
                input=test_prompt,
                actual_output=actual_output,
                expected_output=expected_code
            )

            runner = TestRunner(model="gpt-4", threshold=0.7)
            eval_result = runner.evaluate(test_case, ["answer_relevancy", "g_eval"])

            results[model] = {
                "response": actual_output[:100] + "..." if len(actual_output) > 100 else actual_output,
                "score": eval_result.score,
                "passed": eval_result.passed,
                "metrics": eval_result.metrics
            }

        except Exception as e:
            results[model] = {"error": str(e)}

    # Print comparison
    print("\\nOllama Model Comparison:")
    print("-" * 50)
    for model, result in results.items():
        print(f"{model}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(".2f")
            print(f"  Passed: {result['passed']}")
            print(f"  Response: {result['response']}")
        print()

    return results

def benchmark_ollama_performance():
    """Benchmark Ollama model performance across different tasks."""

    benchmark_prompts = [
        {
            "name": "factual_qa",
            "prompt": "What is the capital of Australia?",
            "expected": "Canberra"
        },
        {
            "name": "creative_writing",
            "prompt": "Write a haiku about artificial intelligence.",
            "expected": "A short poem following 5-7-5 syllable structure about AI"
        },
        {
            "name": "code_explanation",
            "prompt": "Explain what this Python code does: x = [i**2 for i in range(10)]",
            "expected": "Creates a list of squares of numbers from 0 to 9"
        }
    ]

    model = "llama2"
    results = {}

    for benchmark in benchmark_prompts:
        payload = {
            "model": model,
            "prompt": benchmark["prompt"],
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 100
            }
        }

        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            actual_output = result.get("response", "")

            test_case = BaseTestCase(
                name=f"ollama_{model}_{benchmark['name']}",
                input=benchmark["prompt"],
                actual_output=actual_output,
                expected_output=benchmark["expected"]
            )

            runner = TestRunner(model="gpt-4", threshold=0.6)
            eval_result = runner.evaluate(test_case, ["answer_relevancy"])

            results[benchmark["name"]] = {
                "score": eval_result.score,
                "passed": eval_result.passed,
                "response": actual_output[:150] + "..." if len(actual_output) > 150 else actual_output
            }

        except Exception as e:
            results[benchmark["name"]] = {"error": str(e)}

    print(f"\\nOllama {model} Benchmark Results:")
    print("-" * 50)
    for task, result in results.items():
        print(f"{task.upper()}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(".2f")
            print(f"  Passed: {result['passed']}")
            print(f"  Sample: {result['response']}")
        print()

    return results

if __name__ == "__main__":
    # Single model evaluation
    evaluate_ollama_model()

    # Model comparison
    compare_ollama_models()

    # Performance benchmarking
    benchmark_ollama_performance()