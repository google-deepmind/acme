#!/bin/bash
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


# Bash settings: fail on any error and display all commands being run.
set -e
set -x

# Python must be 3.6 or higher.
python --version

# Set up a virtual environment.
python -m venv acme_testing
source acme_testing/bin/activate

# Install dependencies.
pip install --upgrade pip setuptools
pip --version
pip install .
pip install .[jax]
pip install .[tf]
pip install .[reverb]
pip install .[envs]
pip install .[testing]
pip install .[launchpad]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run static type-checking.
pytype -k -j "${N_CPU}" `find . -maxdepth 1 -mindepth 1 -type d` -x 'examples/open_spiel examples/offline acme_testing'

# Run all tests.
pytest --ignore-glob="*/agent_test.py" --ignore-glob="*/agent_distributed_test.py" --durations=10 -n "${N_CPU}" acme

# Clean-up.
deactivate
rm -rf acme_testing/
