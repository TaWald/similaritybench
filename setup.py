import os

from setuptools import find_packages
from setuptools import setup


def resolve_requirements(file):
    req = []
    with open(file) as f:
        req = f.read().splitlines()
        for r in req:
            if r.startswith("-r"):
                req += resolve_requirements(os.path.join(os.path.dirname(file), r.split(" ")[1]))
            else:
                req.append(r)
    return req


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


requirements: list[str] = resolve_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
readme = read_file(os.path.join(os.path.dirname(__file__), "README.md"))

setup(
    name="RepresentationComparison",
    version="v0.3",
    packages=find_packages(),
    include_package_data=True,
    test_suite="unittest",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    tests_require=["coverage"],
    python_requires=">=3.10",
    author="Division of Medical Image Computing, German Cancer Research Center",
    maintainer_email="tassilo.wald@dkfz-heidelberg.de",
    entry_points={"console_scripts": []},
)
