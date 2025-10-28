from setuptools import setup, find_packages
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--version",
    type=str,
    default="0.0.7",
    help="Version number for the package.",
)
args = parser.parse_args()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# read in the pyproject.toml, update the version to match VERSION
with open("pyproject.toml", "r", encoding="utf-8") as f:
    pyproject_lines = f.readlines()
with open("pyproject.toml", "w", encoding="utf-8") as f:
    for line in pyproject_lines:
        if line.startswith("version = "):
            f.write(f'version = "{args.version}"\n')
        else:
            f.write(line)

setup(
    name="PetTag",
    version=args.version,
    author="Sean Farrell",
    author_email="sean.farrell2@durham.ac.uk",
    description="PetTag is a Python package designed for automated disease coding of veterinary clinical texts using either a pre-trained model.",
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
        "datasets>=4.0.0",
        "sentence-transformers>=5.0.0",
    ],
)
