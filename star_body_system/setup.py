from setuptools import setup, find_packages

setup(
    name="star-body-system",
    version="0.1.0",
    description="Multi-layer hierarchical body representation for STAR model",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "plotly",
    ],
    python_requires=">=3.7",
)