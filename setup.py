from pathlib import Path

from setuptools import find_packages, setup

requirements = [
    line.strip()
    for line in Path("requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="netzeroai",
    version="0.1.0",
    description="AI-based framework for Net Zero readiness assessment",
    packages=find_packages(),
    install_requires=requirements,
)
