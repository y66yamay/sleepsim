"""Package setup for sleepsim."""

from setuptools import setup, find_packages

setup(
    name="sleepsim",
    version="0.1.0",
    description="Synthetic PSG data generator for sleep digital twin model testing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4"],
    },
)
