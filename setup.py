# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

# with open("LICENSE") as f:
#     license = f.read()

setup(
    name="llmcomp",
    version="0.1.0",
    description="Tools to compare representations of LLMs",
    long_description=readme,
    author="Max Klabunde",
    author_email="max.klabunde@uni-passau.de",
    url="https://github.com/mklabunde/llmcomp",
    # license=license,
    packages=find_packages(exclude=("tests", "docs")),
)
