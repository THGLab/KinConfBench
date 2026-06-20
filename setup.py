#!/usr/bin/env python
"""
Setup script for kinconfbench package.
"""

from setuptools import setup, find_packages

setup(
    name="kinconfbench",
    version="0.1.0",
    description="Kinase conformational state benchmarking for co-folding predictions",
    author="Kinconfbench Contributors",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "biopython",
        "pyyaml",
        "requests",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
