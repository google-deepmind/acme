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

# Python must be 3.7 or higher.
python --version

# Set up a virtual environment.
python -m venv acme_testing
source acme_testing/bin/activate

# Install dependencies.
pip install --upgrade pip setuptools wheel xmanager
pip install .[jax,tf,launchpad,testing,envs]


N_CPU=$(grep -c ^processor /proc/cpuinfo)
EXAMPLES=$(find examples/ -mindepth 1 -type d -not -path examples/offline -not -path examples/open_spiel)

# Run static type-checking.
for TESTDIR in acme ${EXAMPLES}; do
  pytype -k -j "${N_CPU}" "${TESTDIR}"
done

# Run all tests.
pytest --ignore-glob="*/agent_test.py" --ignore-glob="*/agent_distributed_test.py" --durations=10 -n "${N_CPU}" acme

# Run sample of examples.
# For each of them make sure StepsLimiter reached the limit step count.
# TODO(sinopalnikov): uncomment when we fix the failure:
# http://sponge2/c1d157fd-f885-4156-88e4-7a96abfac7e7
# cd examples/gym
# time python lp_ppo_jax.py --lp_termination_notice_secs=1 > /tmp/log.txt 2>&1 || cat /tmp/log.txt
# cat /tmp/log.txt | grep -E 'StepsLimiter: Max steps of [0-9]+ was reached, terminating'
# time python lp_sac_jax.py --lp_termination_notice_secs=1 > /tmp/log.txt 2>&1 || cat /tmp/log.txt
# cat /tmp/log.txt | grep -E 'StepsLimiter: Max steps of [0-9]+ was reached, terminating'

# Clean-up.
deactivate
ls -l
rm -rf acme_testing/
