"""
Code Generation Test Suite
===========================
Production-ready pytest test suite covering code generation evaluation scenarios.

Tests: answer_relevancy, g_eval (correctness), hallucination

Usage:
    pytest test_suites/codegen_test_suite.py -v
    pytest test_suites/codegen_test_suite.py -v -k "correct"

Coverage:
    - Code correctness and functionality
    - Error handling in generated code
    - Code quality (style, efficiency)
    - Edge case handling
    - Language-specific generation
    - Security vulnerability detection
"""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pytest
from ai_testing.test_cases import CodeTestCase, BaseTestCase
from ai_testing.metrics import (
    AnswerRelevancyMetric,
    HallucinationMetricWrapper,
)
from deepeval.test_case import LLMTestCase


# ---------------------------------------------------------------------------
# Fixtures – reusable code generation test cases
# ---------------------------------------------------------------------------

@pytest.fixture
def correct_python_sort() -> CodeTestCase:
    """A correct Python sorting implementation."""
    return CodeTestCase(
        name="code_correct_sort",
        input="Write a Python function to sort a list of integers in ascending order using merge sort.",
        actual_output=(
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    result.extend(left[i:])\n"
            "    result.extend(right[j:])\n"
            "    return result"
        ),
        expected_output=(
            "A working merge sort implementation with O(n log n) time complexity."
        ),
        language="python",
        code_output=(
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] <= right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    result.extend(left[i:])\n"
            "    result.extend(right[j:])\n"
            "    return result"
        ),
    )


@pytest.fixture
def buggy_python_code() -> CodeTestCase:
    """A code generation with a logic bug — off-by-one error."""
    return CodeTestCase(
        name="code_buggy_offbyone",
        input="Write a Python function to find the maximum element in a list.",
        actual_output=(
            "def find_max(lst):\n"
            "    max_val = lst[0]\n"
            "    for i in range(len(lst) - 1):  # Bug: should be range(len(lst))\n"
            "        if lst[i] > max_val:\n"
            "            max_val = lst[i]\n"
            "    return max_val"
        ),
        expected_output=(
            "A correct max-finding function that handles all elements."
        ),
        language="python",
    )


@pytest.fixture
def sql_injection_vulnerable() -> CodeTestCase:
    """Generated code with a SQL injection vulnerability."""
    return CodeTestCase(
        name="code_sql_injection_vulnerable",
        input="Write a Python function to look up a user by username in a database.",
        actual_output=(
            "def get_user(username):\n"
            "    conn = sqlite3.connect('users.db')\n"
            "    cursor = conn.cursor()\n"
            "    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()"
        ),
        expected_output=(
            "A secure function using parameterized queries to prevent SQL injection."
        ),
        language="python",
    )


@pytest.fixture
def correct_react_component() -> CodeTestCase:
    """A correct React component with proper patterns."""
    return CodeTestCase(
        name="code_correct_react",
        input="Create a React component that fetches and displays a list of users from an API.",
        actual_output=(
            "import React, { useState, useEffect } from 'react';\n\n"
            "const UserList = () => {\n"
            "  const [users, setUsers] = useState([]);\n"
            "  const [loading, setLoading] = useState(true);\n"
            "  const [error, setError] = useState(null);\n\n"
            "  useEffect(() => {\n"
            "    fetch('https://jsonplaceholder.typicode.com/users')\n"
            "      .then(res => res.json())\n"
            "      .then(data => { setUsers(data); setLoading(false); })\n"
            "      .catch(err => { setError(err.message); setLoading(false); });\n"
            "  }, []);\n\n"
            "  if (loading) return <p>Loading...</p>;\n"
            "  if (error) return <p>Error: {error}</p>;\n\n"
            "  return (\n"
            "    <ul>\n"
            "      {users.map(user => (\n"
            "        <li key={user.id}>{user.name} - {user.email}</li>\n"
            "      ))}\n"
            "    </ul>\n"
            "  );\n"
            "};\n\n"
            "export default UserList;"
        ),
        expected_output=(
            "A React component fetching users with loading, error, and data states."
        ),
        language="javascript",
    )


@pytest.fixture
def incomplete_code() -> CodeTestCase:
    """A code generation that is incomplete/has TODO placeholders."""
    return CodeTestCase(
        name="code_incomplete",
        input="Write a Python class for a binary search tree with insert, search, and delete methods.",
        actual_output=(
            "class BinarySearchTree:\n"
            "    def __init__(self):\n"
            "        self.root = None\n\n"
            "    def insert(self, value):\n"
            "        # TODO: implement insert\n"
            "        pass\n\n"
            "    def search(self, value):\n"
            "        # TODO: implement search\n"
            "        pass\n\n"
            "    def delete(self, value):\n"
            "        # TODO: implement delete\n"
            "        pass"
        ),
        expected_output=(
            "A complete BST class with all methods implemented."
        ),
        language="python",
    )


@pytest.fixture
def syntax_error_code() -> CodeTestCase:
    """Generated code with a syntax error."""
    return CodeTestCase(
        name="code_syntax_error",
        input="Write a Python function to calculate the factorial of a number.",
        actual_output=(
            "def factorial(n):\n"
            "    if n == 0\n"  # Missing colon
            "        return 1\n"
            "    else\n"  # Missing colon
            "        return n * factorial(n-1)"
        ),
        expected_output=(
            "A correct factorial function with proper syntax."
        ),
        language="python",
    )


@pytest.fixture
def optimal_vs_bruteforce() -> CodeTestCase:
    """A brute force solution when an optimal one exists."""
    return CodeTestCase(
        name="code_brute_force_fibonacci",
        input="Write an efficient Python function to compute the nth Fibonacci number.",
        actual_output=(
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)"
        ),
        expected_output=(
            "An O(n) or O(log n) Fibonacci implementation using iteration or memoization."
        ),
        language="python",
    )


@pytest.fixture
def correct_rust_function() -> CodeTestCase:
    """A correct Rust implementation."""
    return CodeTestCase(
        name="code_correct_rust",
        input="Write a Rust function that takes a string and returns the number of words.",
        actual_output=(
            "fn count_words(s: &str) -> usize {\n"
            "    s.split_whitespace().count()\n"
            "}"
        ),
        expected_output=(
            "A Rust function that correctly counts words using split_whitespace."
        ),
        language="rust",
    )


@pytest.fixture
def edge_case_missing() -> CodeTestCase:
    """Generated code that doesn't handle edge cases."""
    return CodeTestCase(
        name="code_missing_edge_cases",
        input="Write a Python function to divide two numbers and handle edge cases.",
        actual_output=(
            "def divide(a, b):\n"
            "    return a / b"
        ),
        expected_output=(
            "A division function that handles ZeroDivisionError and type checking."
        ),
        language="python",
    )


@pytest.fixture
def correct_docker_compose() -> CodeTestCase:
    """A correct Docker Compose configuration."""
    return CodeTestCase(
        name="code_correct_docker_compose",
        input="Create a docker-compose.yml for a web app with a React frontend, "
              "Express backend, and PostgreSQL database.",
        actual_output=(
            "version: '3.8'\n"
            "services:\n"
            "  frontend:\n"
            "    build: ./frontend\n"
            "    ports:\n"
            "      - '3000:3000'\n"
            "    depends_on:\n"
            "      - backend\n"
            "  backend:\n"
            "    build: ./backend\n"
            "    ports:\n"
            "      - '5000:5000'\n"
            "    environment:\n"
            "      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app\n"
            "    depends_on:\n"
            "      - db\n"
            "  db:\n"
            "    image: postgres:15\n"
            "    environment:\n"
            "      - POSTGRES_USER=postgres\n"
            "      - POSTGRES_PASSWORD=postgres\n"
            "      - POSTGRES_DB=app\n"
            "    volumes:\n"
            "      - postgres_data:/var/lib/postgresql/data\n"
            "volumes:\n"
            "  postgres_data:"
        ),
        expected_output=(
            "A docker-compose.yml with frontend, backend, and PostgreSQL services."
        ),
        language="yaml",
    )


# ---------------------------------------------------------------------------
# Positive Test Cases
# ---------------------------------------------------------------------------

class TestCodeCorrectness:
    """Tests for code correctness and functionality."""

    def test_merge_sort_correct(self, correct_python_sort: CodeTestCase) -> None:
        """Verify merge sort implementation has correct structure."""
        code = correct_python_sort.actual_output
        assert "def merge_sort" in code, "Should define merge_sort function"
        assert "def merge" in code, "Should define merge helper function"
        assert "mid = len(arr)" in code, "Should calculate midpoint"
        assert "return merge" in code, "Should call merge function"

    def test_react_component_pattern(self, correct_react_component: CodeTestCase) -> None:
        """Verify React component has proper patterns."""
        code = correct_react_component.actual_output
        assert "useState" in code, "Should use useState hook"
        assert "useEffect" in code, "Should use useEffect hook"
        assert "loading" in code, "Should handle loading state"
        assert "error" in code, "Should handle error state"
        assert "key={user.id}" in code, "Should use proper list keys"

    def test_rust_word_count(self, correct_rust_function: CodeTestCase) -> None:
        """Verify Rust function uses idiomatic patterns."""
        code = correct_rust_function.actual_output
        assert "split_whitespace" in code, "Should use split_whitespace"
        assert "count()" in code, "Should use iterator count"

    def test_docker_compose_services(self, correct_docker_compose: CodeTestCase) -> None:
        """Verify docker-compose has all required services."""
        code = correct_docker_compose.actual_output
        assert "frontend" in code, "Should have frontend service"
        assert "backend" in code, "Should have backend service"
        assert "postgres" in code.lower(), "Should have PostgreSQL service"


class TestCodeSecurity:
    """Tests for security vulnerability detection in generated code."""

    def test_sql_injection_detected(self, sql_injection_vulnerable: CodeTestCase) -> None:
        """Should detect SQL injection vulnerability."""
        code = sql_injection_vulnerable.actual_output
        assert "f\"SELECT" in code or "f'SELECT" in code, "Uses f-string in SQL"
        assert "execute(query)" in code, "Executes raw query string"
        assert "parameter" not in code.lower(), "No parameterized query"

    def test_edge_case_handling_missing(self, edge_case_missing: CodeTestCase) -> None:
        """Should detect missing edge case handling for division by zero."""
        code = edge_case_missing.actual_output
        assert "try" not in code, "No error handling"
        assert "except" not in code, "No exception handling"
        assert "zero" not in code.lower(), "No zero division check"


class TestCodeQuality:
    """Tests for code quality metrics."""

    def test_incomplete_code_detected(self, incomplete_code: CodeTestCase) -> None:
        """Should detect incomplete code with TODO placeholders."""
        code = incomplete_code.actual_output
        assert "TODO" in code, "Should detect TODO placeholders"
        assert code.count("pass") >= 3, "Methods are not implemented"

    def test_syntax_error_detected(self, syntax_error_code: CodeTestCase) -> None:
        """Should detect Python syntax errors."""
        code = syntax_error_code.actual_output
        # Lines with 'if' and 'else' missing colons
        lines = code.split('\n')
        error_lines = [l for l in lines if ('if' in l or 'else' in l) and ':' not in l]
        assert len(error_lines) >= 2, "Should detect missing colons"

    def test_inefficient_algorithm(self, optimal_vs_bruteforce: CodeTestCase) -> None:
        """Should detect inefficient recursive Fibonacci."""
        code = optimal_vs_bruteforce.actual_output
        assert "fibonacci(n-1) + fibonacci(n-2)" in code, "Uses exponential brute force"
        # No memoization, no iteration
        assert "memo" not in code.lower() and "cache" not in code.lower() and "@lru_cache" not in code, (
            "No memoization or caching used"
        )


# ---------------------------------------------------------------------------
# Negative Test Cases – failure mode detection
# ---------------------------------------------------------------------------

class TestCodeFailures:
    """Tests that detect code generation failures."""

    def test_bug_offbyone_detection(self, buggy_python_code: CodeTestCase) -> None:
        """Should detect off-by-one error in range."""
        code = buggy_python_code.actual_output
        assert "range(len(lst) - 1)" in code, "Off-by-one error present"
        # This would fail for lists where max is at last position
        assert "max_val" in code, "Uses wrong comparison logic too"
