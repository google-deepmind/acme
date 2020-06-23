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

"""Networks useful for building recurrent agents.
"""

from typing import Optional, Sequence, Tuple
from acme import types
from acme.tf import savers
import sonnet as snt
import tensorflow as tf
import tree

RNNState = types.NestedTensor


class UnpackWrapper(snt.Module):
  """Gets a list of arguments and pass them as separate arguments.

  Example
  ```
    class Critic(snt.Module):
      def __call__(self, o, a):
        pass

    critic = Critic()
    UnpackWrapper(critic)((o, a))
  ```
  calls critic(o, a)
  """

  def __init__(self, module: snt.Module, name: str = 'UnpackWrapper'):
    super().__init__(name=name)
    self._module = module

  def __call__(self,
               inputs: Sequence[types.NestedTensor]) -> types.NestedTensor:
    # Unpack the inputs before passing to the underlying module.
    return self._module(*inputs)


class RNNUnpackWrapper(snt.RNNCore):
  """Gets a list of arguments and pass them as separate arguments.

  Example
  ```
    class Critic(snt.RNNCore):
      def __call__(self, o, a, prev_state):
        pass

    critic = Critic()
    RNNUnpackWrapper(critic)((o, a), prev_state)
  ```
  calls m(o, a, prev_state)
  """

  def __init__(self, module: snt.RNNCore, name: str = 'RNNUnpackWrapper'):
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: Sequence[types.NestedTensor],
               prev_state: RNNState) -> Tuple[types.NestedTensor, RNNState]:
    # Unpack the inputs before passing to the underlying module.
    return self._module(*inputs, prev_state)

  def initial_state(self, batch_size):
    return self._module.initial_state(batch_size)


class CriticDeepRNN(snt.DeepRNN):
  """Same as snt.DeepRNN, but takes three inputs (obs, act, prev_state).
  """

  def __init__(self, layers: Sequence[snt.Module]):
    # Make the first layer take a single input instead of a list of arguments.
    if isinstance(layers[0], snt.RNNCore):
      first_layer = RNNUnpackWrapper(layers[0])
    else:
      first_layer = UnpackWrapper(layers[0])
    super().__init__([first_layer] + list(layers[1:]))

    self._unwrapped_first_layer = layers[0]
    self.__input_signature = None

  def __call__(self, inputs: types.NestedTensor, action: tf.Tensor,
               prev_state: RNNState) -> Tuple[types.NestedTensor, RNNState]:
    # Pack the inputs into a tuple and then using inherited DeepRNN logic to
    # pass them through the layers.
    # This in turn will pass the packed inputs into the first layer
    # (UnpackWrapper) which will unpack them back.
    return super().__call__((inputs, action), prev_state)

  @property
  def _input_signature(self) -> Optional[tf.TensorSpec]:
    """Return input signature for Acme snapshotting.

    The Acme way of snapshotting works as follows: you first create your network
    variables via the utility function `acme.tf.utils.create_variables()`, which
    adds an `_input_signature` attribute to your module. This attribute is
    critical for proper snapshot saving and loading.

    If a module with such an attribute is wrapped into e.g. DeepRNN, Acme
    descends into the `_layers[0]` of that DeepRNN to find the input
    signature.

    This implementation allows CriticDeepRNN to work seamlessly like DeepRNN for
    the following two use-cases:

        1) Creating variables *before* wrapping:
          ```
          unwrapped_critic = Critic()
          acme.tf.utils.create_variables(unwrapped_critic, specs)
          critic = CriticDeepRNN([unwrapped_critic])
          ```

        2) Create variables *after* wrapping:
          ```
          unwrapped_critic = Critic()
          critic = CriticDeepRNN([unwrapped_critic])
          acme.tf.utils.create_variables(critic, specs)
          ```

    Returns:
      input_signature of the module or None of it is not known (i.e. the
      variables were not created by acme.tf.utils.create_variables nor for this
      module nor for any of its descendants).
    """

    if self.__input_signature is not None:
      # To make case (2) (see above) work, we need to allow create_variables to
      # assign an _input_signature attribute to this module, which is why we
      # create additional __input_signature attribute with a setter (see below).
      return self.__input_signature

    # To make case (1) work, we descend into self._unwrapped_first_layer
    # and try to get its input signature (if it exists) by calling
    # savers.get_input_signature.

    # Ideally, savers.get_input_signature should automatically descend into
    # DeepRNN. But in this case it breaks on CriticDeepRNN because
    # CriticDeepRNN._layers[0] is an UnpackWrapper around the underlying module
    # and not the module itself.
    input_signature = savers._get_input_signature(self._unwrapped_first_layer)  # pylint: disable=protected-access
    if input_signature is None:
      return None
    # Since adding recurrent modules via CriticDeepRNN changes the recurrent
    # state, we need to update its spec here.
    state = self.initial_state(1)
    input_signature[-1] = tree.map_structure(
        lambda t: tf.TensorSpec((None,) + t.shape[1:], t.dtype), state)
    self.__input_signature = input_signature
    return input_signature

  @_input_signature.setter
  def _input_signature(self, new_spec: tf.TensorSpec):
    self.__input_signature = new_spec
