#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="boml",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        'torch>=0.4.1',
        'gpytorch',
        'seaborn',
        'numpy',
        'tqdm',
        'matplotlib',
        'click'
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
