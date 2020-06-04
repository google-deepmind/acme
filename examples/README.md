# Examples

This directory includes a number of working examples of Acme agents. These
examples are not meant to be comprehensive and instead show a number of common
use cases under which Acme agents can be applied.

## Quickstart

We include a quick start [tutorial] to get you started. This takes you through
the construction of the D4PG agent as an example, but the general structure
should apply to any of the agents and environment combinations below.

## Control Suite examples

These include a number of agents running on the [dm_control] suite of continuous
control tasks:

-   [`run_d4pg`](control_suite/run_d4pg.py): a deterministic policy gradient (D4PG) agent which includes a
    determinstic policy and a distributional critic.
-   [`run_dmpo`](control_suite/run_dmpo.py): a maximum-a-posterior policy optimization agent which combines
    both a distributional critic and a stochastic policy.

Note that these examples require a [MuJoCo license](https://www.roboti.us/license.html) in order to run. See our
[tutorial] for more details or see the [dm_control] repository for further
information.

[tutorial]: tutorial.ipynb

## Behaviour Suite examples

We also include simple baselines across the [bsuite] set of tasks:

-   [`run_dqn`](bsuite/run_dqn.py): an off-policy DQN examples; and
-   [`run_impala`](bsuite/run_impala.py): an (on-policy) Impala agent;

Finally we also include an example using demonstrations:

-   [`run_dqfd`](bsuite/run_dqfd.py): the DQfD agent running on hard-exploration tasks within bsuite
    (e.g. deep sea) using demonstration data; and
-   [`run_bc`](bsuite/run_bc.py): a behaviour cloning agent.

as well as a model-based example:

-   [`run_mcts`](bsuite/run_mcts.py): the MCTS agent running on the task suite using either a
    simulator of the environment or a learned model.

[bsuite]: https://github.com/deepmind/bsuite
[dm_control]: https://github.com/deepmind/dm_control
