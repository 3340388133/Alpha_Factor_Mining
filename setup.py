"""
Setup script for the Alpha Factor Mining project
"""

from setuptools import setup, find_packages

setup(
    name="alpha-factor-mining",
    version="0.1.0",
    description="Synergistic Alpha Factor Mining with RL, Crowding Simulator, and GAN",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "scipy>=1.7.0",
        "gym>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "pandas>=1.3.0",
        ]
    }
)
