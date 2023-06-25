import os

import pkg_resources
from setuptools import setup, find_packages


setup(
  name="memos",
  version="0.0.1",
  packages=find_packages(exclude=["tests*"]),
  install_requires=[
    "cache-decorator",
    "fuzzysearch",
    "GitPython",
    "langchain",
    "numpy", # missing dependency from cache-decorator
    "openai",
    "pydub",
    "pyyaml",
    "python-slugify",
    "tiktoken",
  ],
  entry_points={
    "console_scripts": ["memos=memos.cli:main"],
  },
  extras_require={
    "test": ["pytest"],
    "dev": ["pylint", "black"],
  },
)
