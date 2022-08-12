# Decentralized Multiagent Learning

This folder contains an implementation of decentralized multiagent learning.
The current implementation supports homogeneous sub-agents (i.e., all agents
running identical sub-algorithms).

The underlying multiagent environment should produce observations and rewards
that are each a dict, with keys corresponding to string IDs for the agents that
map to their respective local observation and rewards. Rewards can be
heterogeneous (e.g., for non-cooperative environments).

The environment step() should consume dict-style actions, with key:value pairs
corresponding to agent:action, as above.

Discounts are assumed shared between agents (i.e., should be a single scalar).
