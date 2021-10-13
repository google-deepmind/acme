"""Tests for the noop starts wrapper."""

from absl.testing import absltest
from unittest import mock

from acme.wrappers import NoopStartsWrapper
from acme.testing import fakes
from dm_env import specs
import numpy as np


class NoopStartsTest(absltest.TestCase):

  def test_reset(self):
    noop_action = 0
    noop_max = 10

    base_env = fakes.DiscreteEnvironment(
      action_dtype=np.int64,
      obs_dtype=np.int64,
      reward_spec=specs.Array(dtype=np.float64, shape=())
    )
    mock_step_fn = mock.MagicMock()
    with mock.patch.object(base_env, "step", mock_step_fn):
      env = NoopStartsWrapper(
        base_env,
        noop_action=noop_action,
        noop_max=noop_max,
        seed=24,
      )
      env.reset()

      # Test environment step called with noop action as part of wrapper.reset
      mock_step_fn.assert_called_with(noop_action)
      self.assertTrue(mock_step_fn.call_count <= noop_max)
      self.assertEqual( mock_step_fn.call_args, ((noop_action,), {}))


if __name__ == '__main__':
  absltest.main()
