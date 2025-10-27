from setuptools import setup, find_packages


VERSION = "0.0.4"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PetTag",
    version=VERSION,
    author="Sean Farrell",
    author_email="sean.farrell2@durham.ac.uk",
    description="PetTag is a Python package designed for coding Veterianry clinical notes ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanfarr788/PetTag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "accelerate",
        "backports.tarfile",
        "colorlog",
        "datasets",
        "transformers",
        "importlib-metadata",
        "jaraco.collections",
        "pandas",
        "protobuf",
        "pysocks",
        "sentencepiece",
        "tomli",
        "torch",  # Added for petharbor[advance]
    ],
    extras_require={
        "advance": ["torch", "transformers", "accelerate"],
        "lite": [
            "datasets",
        ],
    },
)
