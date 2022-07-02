"""

To install this local package
    python -m pip install .
To upgrade this package
    python -m pip install --upgrade .

"""
from setuptools import setup, find_packages

setup(
    name='nvita',
    description='nVITA -- The time-series attack algorithm',
    packages=find_packages(),
    version='0.0.1',
    python_requires='>=3.8',
)