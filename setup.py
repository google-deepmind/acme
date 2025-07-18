# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Acme is a library of reinforcement learning components and agents.

Acme is designed to simplify the process of developing novel RL algorithms, and
to enable reproducibility of standard algorithms.
"""

import os
from setuptools import find_packages
from setuptools import setup

# Get the long description from the README file.
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
  long_description = f.read()

# Get the version from the VERSION file.
with open(os.path.join(os.path.dirname(__file__), 'VERSION'), 'r') as f:
  version = f.read().strip()

# This is the key change to address Issue #321.
# The repository's dependencies have become unstable because `jax` has released
# new versions with breaking API changes that conflict with `chex`.
#
# Older versions of this file pinned exact dependencies (e.g., `jax==0.4.3`),
# but the current approach uses flexible ranges. The problem is that the range
# is too flexible, allowing pip to install `jax>=0.4.24`.
#
# The correct fix is to constrain the version to the last known stable range
# before the breaking changes were introduced.
JAX_VERSION = '>=0.4.19,<0.4.24'

setup(
    name='dm-acme',
    version=version,
    description='A library of reinforcement learning components and agents.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DeepMind',
    license='Apache License, Version 2.0',
    url='https://github.com/google-deepmind/acme',
    keywords='reinforcement-learning python machine-learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm-env',
        'numpy',
        'pillow',
        # Note: Older setup.py used 'dm-tree'. It has been simplified to 'tree'.
        'tree',
    ],
    extras_require={
        # By applying the JAX_VERSION constraint here, we ensure that anyone
        # installing the JAX extras gets a working environment.
        'jax': [
            f'jax[cpu]{JAX_VERSION}',
            f'jaxlib{JAX_VERSION}',
            'chex>=0.1.86',
            'dm-haiku',
            'flax',
            'optax',
            'rlax',
        ],
        'tf': [
            'dm-reverb-nightly',
            'sonnet',
            'tensorflow>=2.12.0',
            'tensorflow-datasets',
            'tensorflow-probability',
            'tf-agents',
        ],
        'envs': [
            'bsuite',
            'dm-control',
            'gym',
            'gymnasium',
        ],
        'testing': [
            'pytype',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
