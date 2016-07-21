#!/usr/bin/env python3

import sys, os, re
from setuptools import setup
from setuptools import find_packages


if sys.version_info[:3] < (3, 3):
    raise SystemExit("You need Python 3.3+")

with open(os.path.join(os.path.dirname(__file__),'migen','version.py'),'r') as fh:
    version = re.search(r'__version__\s*=\s*("([^"]*)")|(\'([^\']*)\')')
    version = version.group(2) or version.group(4)

setup(
    name="migen",
    version=version,
    description="Python toolbox for building complex digital hardware",
    long_description=open("README.md").read(),
    author="Sebastien Bourdeauducq",
    author_email="sb@m-labs.hk",
    url="https://m-labs.hk",
    download_url="https://github.com/m-labs/migen",
    packages=find_packages(),
    test_suite="migen.test",
    license="BSD",
    platforms=["Any"],
    keywords="HDL ASIC FPGA hardware design",
    classifiers=[
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Environment :: Console",
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
