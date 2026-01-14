from setuptools import setup, find_packages

setup(
    name="flh",
    version="0.1.0",
    description="FLH - Fast Linear Hadamard quantization library",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "tqdm>=4.60.0",
        "datasets>=2.0.0",  # For WikiText2 and other datasets
    ],
    extras_require={
        "eval": [
            "lm-eval>=0.4.0",  # For additional benchmarks
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
