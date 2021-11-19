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
import os
import sys

from setuptools import find_packages
from setuptools import setup
import setuptools.command.build_py
import setuptools.command.develop

spec = import_util.spec_from_file_location('_metadata', 'acme/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

# TODO(b/184148890): Add a release flag


# Any particular version of reverb needs to be pinned against a particular
# version of TF due to how it is built. While the versions below should be the
# most recent stable versions of each library we'll be explicit just make make
# sure this constraint is upheld.
tensorflow = [
    'dm-reverb==0.6.1',
    'keras==2.7.0',
    'tensorflow-datasets==4.4.0',
    'tensorflow-estimator==2.7.0',
    'tensorflow==2.7.0',
    'tensorflow_probability==0.15.0',
]

core_requirements = [
    'absl-py',
    'dm-env',
    'dm-tree',
    'numpy',
    'pillow',
    'typing-extensions',
]

jax_requirements = [
    'jax',
    'jaxlib',
    'dm-haiku',
    'flax',
    'optax',
    'rlax',
] + tensorflow

tf_requirements = [
    'dm-sonnet',
    'trfl',
] + tensorflow

# This is the version of launchpad that works with the version of reverb/TF
# specified above.
launchpad_requirements = [
    'dm-launchpad==0.3.2',
]

testing_requirements = [
    'pytype==2021.8.11',  # TODO(b/206926677): update to new version.
    'pytest-xdist',
]

envs_requirements = [
    'atari-py',
    'bsuite',
    'dm-control==0.0.364896371',  # TODO(b/206925413): update to new version.
    'gym',
    'gym[atari]',
    'tensorflow_datasets',
]


def generate_requirements_file(path=None):
  """Generates requirements.txt file with the Acme's dependencies.

  It is used by Launchpad GCP runtime to generate Acme requirements to be
  installed inside the docker image. Acme itself is not installed from pypi,
  but instead sources are copied over to reflect any local changes made to
  the codebase.

  Args:
    path: path to the requirements.txt file to generate.
  """
  if not path:
    path = os.path.join(os.path.dirname(__file__), 'acme/requirements.txt')
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


class BuildPy(setuptools.command.build_py.build_py):

  def run(self):
    generate_requirements_file()
    setuptools.command.build_py.build_py.run(self)


class Develop(setuptools.command.develop.develop):

  def run(self):
    generate_requirements_file()
    setuptools.command.develop.develop.run(self)

cmdclass = {
    'build_py': BuildPy,
    'develop': Develop,
}

setup(
    name='dm-acme',
    version=version,
    cmdclass=cmdclass,
    description='A Python library for Reinforcement Learning.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    package_data={'': ['requirements.txt']},
    include_package_data=True,
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
