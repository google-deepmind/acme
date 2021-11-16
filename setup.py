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

import datetime
from importlib import util as import_util
import sys

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location('_metadata', 'acme/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

# TODO(b/184148890): Add a release flag


tensorflow = [
    'dm-reverb==0.4.0',
    'tensorflow-datasets==4.4.0',
    'tensorflow-estimator==2.6.0',
    'tensorflow==2.6.0',
    'tfp-nightly==0.14.0.dev20210818',
    'tensorflow_probability'
]

core_requirements = [
    'absl-py==0.12.0',
    'dm-env==1.5',
    'dm-tree==0.1.6',
    'numpy',
    'pillow==8.3.2',
]

jax_requirements = [
    'jax==0.2.19',
    'jaxlib==0.1.70',
    'dm-haiku==0.0.4',
    'flax==0.3.5',
    'optax==0.0.9',
    'rlax==0.0.4',
    'keras==2.6.0',
    'typing-extensions',
] + tensorflow

tf_requirements = [
    'bsuite==0.3.5',
    'dm-sonnet==2.0.0',
    'trfl==1.2.0',
] + tensorflow

launchpad_requirements = [
    'dm-launchpad==0.3.2',
]

testing_requirements = [
    'pytype==2021.8.11',
    'pytest-xdist==2.3.0',
]

envs_requirements = [
    'atari-py==0.2.9',
    'bsuite==0.3.5',
    'dm-control==0.0.364896371',
    'gym',
    'gym[atari]',
    'tensorflow_datasets',
]


def generate_requirements_file(path):
  """Generates requirements.txt file with the Acme's dependencies.

  It is used by Launchpad GCP runtime to generate Acme requirements to be
  installed inside the docker image. Acme itself is not installed from pypi,
  but instead sources are copied over to reflect any local changes made to
  the codebase.

  Args:
    path: path to the requirements.txt file to generate.
  """
  with open(path, 'w') as f:
    for package in set(core_requirements + jax_requirements + tf_requirements +
                       launchpad_requirements + envs_requirements):
      f.write(f'{package}\n')


long_description = """Acme is a library of reinforcement learning (RL) agents
and agent building blocks. Acme strives to expose simple, efficient,
and readable agents, that serve both as reference implementations of popular
algorithms and as strong baselines, while still providing enough flexibility
to do novel research. The design of Acme also attempts to provide multiple
points of entry to the RL problem at differing levels of complexity.

For more information see [github repository](https://github.com/deepmind/acme)."""

# Get the version from metadata.
version = _metadata.__version__

# If we're releasing a nightly/dev version append to the version string.
if '--nightly' in sys.argv:
  sys.argv.remove('--nightly')
  version += '.dev' + datetime.datetime.now().strftime('%Y%m%d')

setup(
    name='dm-acme',
    version=version,
    description='A Python library for Reinforcement Learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=core_requirements,
    extras_require={
        'jax': jax_requirements,
        'tf': tf_requirements,
        'launchpad': launchpad_requirements,
        'testing': testing_requirements,
        'envs': envs_requirements,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
