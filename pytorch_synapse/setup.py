#!/usr/bin/env python

import os
import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements_filename():
    if 'READTHEDOCS' in os.environ:
        return "REQUIREMENTS-RTD.txt"
    elif 'DOCKER' in os.environ:
        return "REQUIREMENTS-DOCKER.txt"
    else:
        return "REQUIREMENTS.txt"


install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), get_requirements_filename()))
]

setuptools.setup(
    name='pytorch_synapse',
    version='0.1.0',
    description='A PyTorch package for contrastive representation learning of 3D electron microscopy image chunks',
    long_description=readme(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research'
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering',
    ],
    url='http://github.com/broadinstitute/pytorch_synapse',
    author='Mehrtash Babadi, Alyssa Wilson',
    license='BSD (3-Clause)',
    packages=setuptools.find_packages(),
    package_data={'': ['config/*.yaml']},
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)