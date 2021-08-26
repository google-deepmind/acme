# FAQ

## Environments

-   **Does Acme support my environment?** All _agents_ in Acme are designed to
    work with environments which implement the
    [dm_env environment interface][dm_env]. This interface, however, has been
    designed to match general concepts widely in use across the RL research
    community. As a result, it should be quite straight-forward to write a
    wrapper for other environments in order to make them conform to this
    interface. See e.g. the `acme.wrappers.gym_wrapper` module which can be used
    to interact with [OpenAI Gym][gym] environments.

    Note: Follow the instructions [here][atari] to install ROMs for Atari
    environments.

    Similarly, _learners_ in Acme are designed to consume dataset iterators
    (generally `tf.Dataset` instances) which consume either transition tuples or
    sequences of state, action, reward, etc. tuples. If your data does not match
    these formats it should be relatively straightforward to write an adaptor!
    See individual agents for more information on their expected input.

[dm_env]: https://github.com/deepmind/dm_env
[gym]: https://gym.openai.com/
[atari]: https://github.com/openai/atari-py#roms

## TensorFlow agents

-   **How do I debug my TF2 learner?** Debugging TensorFlow code has never been
    easier! All our learners’ `_step()` functions are decorated with a
    `@tf.function` which can easily be commented out to run them in eager mode.
    In this mode, one can easily run through the code (say, via `pdb`) line by
    line and examine outputs. Most of the time, if your code works in eager
    mode, it will work in graph mode (with the `@tf.function` decorator) but
    there are rare exceptions when using exotic ops with unsupported dtypes.
    Finally, don’t forget to add the decorator back in or you’ll find your
    learner to be a little sluggish!

## Misc.

-   **How should I spell Acme?** Acme is a proper noun, not an acronym, and
    hence should be spelled "Acme" without caps.
