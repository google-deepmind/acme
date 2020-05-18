#!/bin/bash

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
pip install .[envs]
pip install gym[atari]

# Reverb isn't quite ready yet.
pip install -i https://test.pypi.org/simple/ \
    --pre \
    --extra-index-url https://pypi.org/simple/ \
    dm-reverb-nightly==0.0.2.dev20200515

# Run all tests.
pip install pytest
pytest acme

# Clean-up.
deactivate
rm -rf acme_testing/
