# Components

## Environments

Acme is designed to work with environments which implement the [dm_env
environment interface][dm_env]. This provides a common API for interacting with
an environment in order to take actions and receive observations.  This
environment API also provides a standard way for environments to specify the
input and output spaces relevant to that environment via methods like
`environment.action_spec()`. Note that Acme also exposes these spec types
directly via `acme.specs`. However, it is also common for Acme agents to require
a _full environment spec_ which can be obtained by making use of
`acme.make_environment_spec(environment)`.

[dm_env]: https://github.com/deepmind/dm_env

Acme also exposes, under `acme.wrappers`, a number of classes which wrap and/or
expose a `dm_env` environment. All such wrappers are of the form:

```python
environment = Wrapper(raw_environment, ...)
```

where additional parameters may be passed to the wrapper to control its behavior
(see individual implementations for more details). Wrappers exposed directly
include

- `SinglePrecisionWrapper`: converts any double-precision `float` and `int`
  components returned by the environment to single-precision.
- `AtariWrapper`: converts a standard ALE Atari environment using a stack of
  wrappers corresponding to the modifications used in the "[Human Level Control
  Through Deep Reinforcement Learning][nature-atari]" publication.

Acme also includes the `acme.wrappers.gym_wrapper` module which can be used to
interact with [OpenAI Gym][gym] environments. This includes a general
`GymWrapper` class as well as `AtariGymWrapper` which exposes a lives count
observation which can optionally be exposed by the `AtariWrapper`.

[nature-atari]: https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning
[gym]: https://gym.openai.com/

## Networks

An important building block for any agent implementation consists of the
parameterized functions or networks which are used to construct policies, value
functions, etc. Agents implemented in Acme are built to be as agnostic as
possible to the environment on which they will be applied. As a result they
typically require network(s) which are used to directly interact with the
environment either by consuming observations, producing actions, or both. These
are typically passed directly into the agent at initialization, e.g.

```python
policy_network = ...
critic_network = ...
agent = MyActorCriticAgent(policy_network, critic_network, ...)
```

### Tensorflow and Sonnet

For TensorFlow agents, networks in Acme are typically implemented using the
[Sonnet][sonnet] neural network library. These network objects take the form of
a `Callable` object which takes a collection of (nested) `tf.Tensor` objects as
input and outputs a collection of (nested) `tf.Tensor` or `tfp.Distribution`
objects. In what follows we use the following aliases.

```python
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
```

While custom Sonnet modules can be implemented and used directly, Acme also
provides a number of useful network primitives which are tailored to RL tasks;
these can be imported from `acme.tf.networks`, see [networks] for more details.
These primitives can be combined using `snt.Sequential`, or `snt.DeepRNN` when
stacking network modules with state.

When stacking modules it is often, though not always, helpful to distinguish
between what are often called _torso_, _head_, and multiplexer networks. Note
that this categorization is purely pedagogical but has nevertheless proven
useful when discussing network architectures.

Torsos are usually the first to transform inputs (observations, actions, or a
combination) and produce what is commonly known in the deep learning literature
as an embedding vector. These modules can be stacked so that an embedding is
transformed multiple times before it is fed into a head.

Let us consider for instance a simple network we use in the Impala agent when
training it on Atari games:

```python
impala_network = snt.DeepRNN([
    # Torsos.
    networks.AtariTorso(),  # Default Atari ConvNet offered as convenience.
    snt.LSTM(256),  # Custom LSTM core.
    snt.Linear(512), # Custom perceptron layer before head.
    tf.nn.relu,  # Seemlessly stack Sonnet modules and TF ops as usual.
    # Head producing 18 action logits and a value estimate for the input
    # observation.
    networks.PolicyValueHead(num_actions=18),
])
```

Heads are networks that consume the embedding vector to produce a desired output
(e.g. action logits or distribution, value estimates, etc). These modules can
also be stacked, which is useful particularly when dealing with stochastic
policies. For example, consider the following stochastic policy used in the MPO
agent, trained on the control suite:

```python
policy_layer_sizes: Sequence[int] = (256, 256, 256)

stochastic_policy_network = snt.Sequential([
    # MLP torso with initial layer normalization; activate the final layer since
    # it feeds into another module.
    networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
    # Head producing a tfd.Distribution: in this case `num_dimensions`
    # independent normal distributions.
    networks.MultivariateNormalDiagHead(num_dimensions),
  ])
```

This stochastic policy is used internally by the MPO algorithm to compute log
probabilities and Kullback-Leibler (KL) divergences. We can also stack an
additional head that will select the mean of the stochastic policy as a greedy
action:

```python
greedy_policy_network = snt.Sequential([
    networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
    networks.MultivariateNormalDiagHead(num_dimensions),
    networks.StochasticModeHead(),
  ])
```

When designing our actor-critic agents for continuous control tasks, we found
one simple module particularly useful: the `CriticMultiplexer`. This callable
Sonnet module takes two inputs, an observation and an action, and concatenates
them along all but the batch dimension, after possibly transforming them if
either (both) `[observation|action]_network` is (are) passed. For example, the
following is the C51 (see [Bellemare et al., 2017]) distributional critic
network adapted for our D4PG experiments:

```python
critic_layer_sizes: Sequence[int] = (512, 512, 256)

distributional_critic_network = snt.Sequential([
    # Flattens and concatenates inputs; see `tf2_utils.batch_concat` for more.
    networks.CriticMultiplexer(),
    networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
    # Distributional head corresponding to the C51 network.
    networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51),
])
```

Finally, our actor-critic control agents also allow the specification of an
observation network that is shared by the policy and critic. This network embeds
the observations once and uses the transformed input in both the policy and
critic as needed, which saves computation particularly when the transformation
is expensive. This is the case for example when learning from pixels where the
observation network can be a large ResNet. In such cases, the shared visual
network can be specified to any of DDPG, D4PG, MPO, DMPO by simply defining and
passing the following:

```python
shared_resnet = networks.ResNetTorso()  # Default (deep) Impala network.

agent = dmpo.DMPO(
    # Networks defined above.
    policy_network=stochastic_policy_network,
    critic_network=distributional_critic_network,
    # New ResNet visual module, shared by both policy and critic.
    observation_network=shared_resnet,
    # ...
)
```

In this case, the `policy_` and `critic_network` act as heads on top of the
shared visual torso.

[networks]: ../acme/tf/networks/
[sonnet]: https://github.com/deepmind/sonnet/

## Internal components

Acme also includes a number of components and concepts that are typically
internal to an agent's implementation. These components can, in general, be
ignored if you are only interested in using an Acme agent. However they prove
useful when implementing a novel agent or modifying an existing agent.

### Losses

These are some commonly-used loss functions. Note that in general we defer to
[TRFL][trfl] where possible, except in cases for which it does not support
TensorFlow 2.

[trfl]: https://github.com/deepmind/trfl

RL-specific losses implemented include:

-   a [distributional TD loss](distributional) for categorical distributions;
    see [Bellemare et al., 2017].
-   the Deterministic Policy Gradient [(DPG) loss](dpg); see
    [Silver et al., 2014].
-   the Maximum a posteriori Policy Optimization [(MPO) loss](mpo); see
    [Abdolmaleki et al., 2018].

Also implemented (and useful within the losses mentioned above) are:

- the [Huber loss](huber) for robust regression.

[distributional]: distributional.py
[dpg]: dpg.py
[mpo]: mpo.py
[huber]: huber.py

[Abdolmaleki et al., 2018]: https://arxiv.org/abs/1806.06920
[Bellemare et al., 2017]: https://arxiv.org/abs/1707.06887
[Silver et al., 2014]: http://proceedings.mlr.press/v32/silver14

### Adders

An `Adder` packs together data to send to the replay buffer, and potentially
does some reductions/transformations to this data in the process.

All Acme `Adder`s can be interacted through their `add()`, `add_first()`, and
`reset()` methods.

The `add()` method takes actions, timesteps, and potentially some extras and
adds the `action`, `observation`, `reward`, `discount`, `extra` fields to the
buffer.

The `add_first()` method takes the first timestep of an episode and adds it to
the buffer, automatically padding the empty `action` `reward` `discount`, and
`extra` fields that don't exist at the first timestep of an episode.

The `reset` method clears the buffer.

Example usage of an adder:

```python
# Reset the environment and add the first observation.
timestep = env.reset()
adder.add_first(timestep)

while not timestep.last():
  # Generate an action from the policy and step the environment.
  action = my_policy(timestep)
  timestep = env.step(action)

  # Add the action and the resulting timestep.
  adder.add(action, next_timestep=timestep)
```

### ReverbAdders

Acme uses [Reverb](http://github.com/deepmind/reverb) for creating data structures like 
*replay buffers* to store RL experiences.

For convenience, Acme provides several `ReverbAdders` for adding actor
experiences to a Reverb table. The `ReverbAdder`s provided include:

*   `NStepTransitionAdder` takes single steps from an environment/agent loop,
    automatically concatenates them into N-step transitions, and adds the
    transitions to Reverb for future retrieval. The steps are buffered and then
    concatenated into N-step transitions, which are stored in and returned from
    replay.

    Where N is 1, the transitions are of the form:

    ```
    (s_t, a_t, r_t, d_t, s_{t+1}, e_t)
    ```

    For N greater than 1, transitions are of the form:

    ```
    (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+n}, e_t),
    ```

    Transitions can be stored as sequences or episodes.

*   `EpisodeAdder` which adds entire episodes as trajectories of the form:

    ```python
    (s_0, a_0, r_0, d_0, e_0,
     s_1, a_1, r_1, d_1, e_1,
              .
              .
              .
     s_T, a_T, r_T, 0., e_T)
    ```

*   `SequenceAdder` which adds sequences of fixed `sequence_length` n of the
    form:

    ```python
      (s_0, a_0, r_0, d_0, e_0,
       s_1, a_1, r_1, d_1, e_1,
                .
                .
                .
       s_n, a_n, r_n, d_n, e_n)
    ```

    sequences can be overlapping (if the `period` parameter < `sequence_length`
    n) or non-overlapping (if `period <= sequence_length`)

### [Loggers](../acme/utils/loggers/)

Acme contains several loggers for writing out data to common places,
based on the absract `Logger` class, all with `write()` methods.<br><br>
NOTE: By default, loggers will immediately output all data passed through `write()` unless given a nonzero value for the `time_delta` argument when constructing a logger representing the number of seconds between logger outputs. <br>

#### [Terminal Logger](../acme/utils/loggers/terminal.py)

Logs data directly to the terminal.<br><br>
Example:<br>

```python
terminal_logger = loggers.TerminalLogger(label='TRAINING',time_delta=5)
terminal_logger.write({'step': 0, 'reward': 0.0})

>> TRAINING: step: 0, reward: 0.0
```

#### [CSV Logger](../acme/utils/loggers/csv.py)

Logs to specified CSV file.<Br><br>
Example:<br>

```python
csv_logger = loggers.CSVLogger(logdir='logged_data', label='my_csv_file')
csv_logger.write({'step': 0, 'reward': 0.0})
```

### [Tensorflow savers](../acme/tf/savers.py)

To save trained TensorFlow models, we can *checkpoint* or *snapshot*
them. <br>

Both *checkpointing* and *snapshotting* are ways to save and restore model state
  for later use. The difference comes when restoring the checkpoint. <br>

With checkpoints, you have to first re-build the exact graph, then restore the
  checkpoint. They are useful to have while running experiments, in case the
  experiment gets interrupted/preempted and has to be restored to continue the
  \experiment run without losing the experiment state.<br>

Snapshots re-build the graph internally, so all you have to do is restore the
  snapshot.<br>

Acme’s Checkpointer class provides functionality to *both* checkpoint (with the
  `objects_to_save` argument) and snapshot (with the `objects_to_snapshot`
  argument) different parts of the model state as desired. <br>

```python
 model = snt.Linear(10)
 checkpointer = utils.tf2_utils.Checkpointer(
     objects_to_save={'model': model},
     objects_to_snapshot={'model': model})
 for _ in range(100):
   # ...
   checkpointer.save()
```

