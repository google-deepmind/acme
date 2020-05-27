# python3
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

"""Install script for setuptools."""

from importlib import util as import_util
from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location('_metadata', 'acme/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

tf_requirements = [
    'tf-nightly',
    'tfp-nightly',
    'dm-sonnet',
    'trfl',
]

jax_requirements = [
    'jax',
    'jaxlib',
    'dm-haiku @ git+git://github.com/deepmind/dm-haiku.git#egg=dm-haiku',
    'rlax @ git+git://github.com/deepmind/rlax.git#egg=rlax',
]

env_requirements = [
    'bsuite @ git+git://github.com/deepmind/bsuite.git#egg=bsuite',
    'dm-control',
    'gym',
    'gym[atari]',
]

# Use the first paragraph of our README as the long_description.
with open('README.md', 'r') as fh:
  long_description = fh.read().split('\n\n')[1]

# Add a link to github.
long_description += '\n\nFor more information see our '
long_description += '[github repository](https://github.com/deepmind/acme).'

setup(
    name='dm-acme',
    version=_metadata.__version__,
    description='A Python library for Reinforcement Learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'dm-tree',
        'numpy',
        'pillow',
    ],
    tests_require=[
        'pytype',
        'pytest-xdist',
    ],
    extras_require={
        'jax': jax_requirements,
        'tf': tf_requirements,
        'envs': env_requirements,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
