from setuptools import find_packages, setup

setup(
    name="adaptive_portfolio",
    version="1.0.0",
    description="Adaptive Portfolio Optimization based on Markov-Switching Volatility Models",
    author="NSU, Business Informatics, Course Project 2026",
    python_requires=">=3.11",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        line.strip()
        for line in open("requirements.txt", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ],
)
