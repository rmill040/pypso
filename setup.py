#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from os.path import abspath, dirname, join
from setuptools import find_packages, setup
import versioneer

# Package meta-data
NAME            = 'pypso'
DESCRIPTION     = 'Sample Python package for naive particle swarm optimization.'
URL             = 'https://github.com/rmill040/pypso'
EMAIL           = 'rmill040@gmail.com'
AUTHOR          = 'Robert Milletich'
REQUIRES_PYTHON = '>=3.6.0'

# Requirements for project
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

# Import the README and use it as the long-description
here = abspath(dirname(__file__))
try:
    with io.open(join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Define scripts
scripts = ["scripts/pypso_version"]

# Run setup
setup(
    name                          = NAME,
    version                       = versioneer.get_version(),
    cmdclass                      = versioneer.get_cmdclass(),
    description                   = DESCRIPTION,
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = AUTHOR,
    author_email                  = EMAIL,
    python_requires               = REQUIRES_PYTHON,
    url                           = URL,
    packages                      = find_packages(exclude=['tests']),
    scripts                       = scripts,
    package_data                  = {},
    install_requires              = list_reqs(),
    extras_require                = {},
    include_package_data          = True,
    license                       = 'MIT',
    classifiers                   = [
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
)