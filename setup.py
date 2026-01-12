# setup.py
from setuptools import setup, find_packages

setup(
    name="newsflow-ai",
    version="2.5",
    description="Real-time News Intelligence with RAG",
    author="Team Corner Stone",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
        "pathway==0.27.1",
    ],
    python_requires=">=3.8",
)
