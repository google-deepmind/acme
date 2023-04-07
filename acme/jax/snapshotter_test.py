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

"""Tests for snapshotter."""

import os
import time
from typing import Any, Sequence

import jax.numpy as jnp
from absl.testing import absltest

from acme import core
from acme.jax import snapshotter, types
from acme.testing import test_utils


def _model0(params, x1, x2):
    return params["w0"] * jnp.sin(x1) + params["w1"] * jnp.cos(x2)


def _model1(params, x):
    return params["p0"] * jnp.log(x)


class _DummyVariableSource(core.VariableSource):
    def __init__(self):
        self._params_model0 = {
            "w0": jnp.ones([2, 3], dtype=jnp.float32),
            "w1": 2 * jnp.ones([2, 3], dtype=jnp.float32),
        }

        self._params_model1 = {
            "p0": jnp.ones([3, 1], dtype=jnp.float32),
        }

    def get_variables(
        self, names: Sequence[str]
    ) -> Sequence[
        Any
    ]:  # pytype: disable=signature-mismatch  # overriding-return-type-checks
        variables = []
        for n in names:
            if n == "params_model0":
                variables.append(self._params_model0)
            elif n == "params_model1":
                variables.append(self._params_model1)
            else:
                raise ValueError("Unknow variable name: {n}")
        return variables


def _get_model0(variable_source: core.VariableSource) -> types.ModelToSnapshot:
    return types.ModelToSnapshot(
        model=_model0,
        params=variable_source.get_variables(["params_model0"])[0],
        dummy_kwargs={
            "x1": jnp.ones([2, 3], dtype=jnp.float32),
            "x2": jnp.ones([2, 3], dtype=jnp.float32),
        },
    )


def _get_model1(variable_source: core.VariableSource) -> types.ModelToSnapshot:
    return types.ModelToSnapshot(
        model=_model1,
        params=variable_source.get_variables(["params_model1"])[0],
        dummy_kwargs={"x": jnp.ones([3, 1], dtype=jnp.float32),},
    )


class SnapshotterTest(test_utils.TestCase):
    def setUp(self):
        super().setUp()
        self._test_models = {"model0": _get_model0, "model1": _get_model1}

    def _check_snapshot(self, directory: str, name: str):
        self.assertTrue(os.path.exists(os.path.join(directory, name, "model0")))
        self.assertTrue(os.path.exists(os.path.join(directory, name, "model1")))

    def test_snapshotter(self):
        """Checks that the Snapshotter class saves as expected."""
        directory = self.get_tempdir()

        models_snapshotter = snapshotter.JAXSnapshotter(
            variable_source=_DummyVariableSource(),
            models=self._test_models,
            path=directory,
            max_to_keep=2,
            add_uid=False,
        )
        models_snapshotter._save()

        # The snapshots are written in a folder of the form:
        # PATH/{time.strftime}/MODEL_NAME
        first_snapshots = os.listdir(directory)
        self.assertEqual(len(first_snapshots), 1)
        self._check_snapshot(directory, first_snapshots[0])
        # Make sure that the second snapshot is constructed.
        time.sleep(1.1)
        models_snapshotter._save()
        snapshots = os.listdir(directory)
        self.assertEqual(len(snapshots), 2)
        self._check_snapshot(directory, snapshots[0])
        self._check_snapshot(directory, snapshots[1])

        # Make sure that new snapshotter deletes the oldest snapshot upon _save().
        time.sleep(1.1)
        models_snapshotter2 = snapshotter.JAXSnapshotter(
            variable_source=_DummyVariableSource(),
            models=self._test_models,
            path=directory,
            max_to_keep=2,
            add_uid=False,
        )
        self.assertEqual(snapshots, os.listdir(directory))
        time.sleep(1.1)
        models_snapshotter2._save()
        snapshots = os.listdir(directory)
        self.assertNotIn(first_snapshots[0], snapshots)
        self.assertEqual(len(snapshots), 2)
        self._check_snapshot(directory, snapshots[0])
        self._check_snapshot(directory, snapshots[1])


if __name__ == "__main__":
    absltest.main()
