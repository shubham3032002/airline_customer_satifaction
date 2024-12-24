from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f.readlines() if line.strip() and not line.startswith("-e")
    ]

setup(
    name="airline",
    version="0.1",
    author="Shubham",
    author_email="shubhammokal30@gmail.com",
    description="MLOPS PROJECT",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
