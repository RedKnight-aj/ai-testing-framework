from setuptools import setup, find_packages

setup(
    name="ai-testing-framework",
    version="1.0.0",
    description="Production-ready AI Testing Framework using DeepEval",
    author="RedKnight AI",
    author_email="redknight@ai.com",
    url="https://github.com/RedKnight-aj/ai-testing-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "deepeval>=0.21.0",
        "pytest>=7.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "ai-test=ai_testing.cli:main",
        ],
    },
)
