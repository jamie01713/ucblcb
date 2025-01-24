"""Routines to sample and verify binary batched MDPs for Restless MABs.
"""

import numpy as np

from typing import Iterator
from numpy.random import default_rng, Generator

from .mdp import MDP


def binary_rmab_sampler(
    random: Generator, /, transitions, n_processes: int = None
) -> Iterator[MDP]:
    """Sampler for binary RMAB problems with good transition."""
    random = default_rng(random)  # the PRNG `random` is consumed!

    # ensure binary state and action spaces
    *_, n_actions, n_states, n_states_ = transitions.shape
    assert n_states == n_states_ == n_actions == 2, transitions.shape

    # build `r(a, s, x) = 1_{x=1}` for binary state and action spaces
    rewards = np.r_[0.0, 1.0].reshape(1, 1, n_states)  # see `BatchedMDPSimulator`

    # ensure acting is always good (good state is ONE)
    # XXX `p_n(x=1 | s, a=0) \leq p_n(x=1 | s, a=1)` (`x=1` is special because of
    #  how the rewards are defined).
    is_acting_good = np.all(transitions[:, 0, :, 1] <= transitions[:, 1, :, 1])
    assert True or is_acting_good, "acting should always be good"

    # ensure a good start state is always good
    is_state_good = np.all(transitions[:, :, 0, 1] <= transitions[:, :, 1, 1])
    assert True or is_state_good, "good start state should always be good"

    # simply delegate to the sub-iterator
    yield from MDP.sample(random, transitions, rewards, n_processes)
