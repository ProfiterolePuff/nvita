"""

To install this local package
    python -m pip install .
To upgrade this package
    python -m pip install --upgrade .

"""
from setuptools import setup, find_packages

setup(
    name="nvita",
    description="nVITA -- The adversarial attack algorithm for time series forecasting",
    packages=find_packages(),
    version="0.0.6",
    python_requires=">=3.9",
)