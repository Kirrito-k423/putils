from setuptools import setup, find_packages

setup(
    name="putils",
    version="0.1.0",
    description="Development AI model utilities",
    author="Kirrito-k423",
    url="https://github.com/Kirrito-k423/putils",
    packages=find_packages(),
    install_requires=[
        "portalocker",
        "debugpy"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
