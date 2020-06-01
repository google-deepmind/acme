# FAQ

## Misc.

-   **How should I spell Acme?** Acme is a proper noun, not an acronym, and
    hence should be spelled "Acme" without caps.

-   **How do I debug my TF2 learner?** Debugging TensorFlow code has never been
    easier! All our learners’ `_step()` functions are decorated with a
    `@tf.function` which can easily be commented out to run them in eager mode.
    In this mode, one can easily run through the code (say, via `pdb`) line by
    line and examine outputs. Most of the time, if your code works in eager
    mode, it will work in graph mode (with the `@tf.function` decorator) but
    there are rare exceptions when using exotic ops with unsupported dtypes.
    Finally, don’t forget to add the decorator back in or you’ll find your
    learner to be a little sluggish!
