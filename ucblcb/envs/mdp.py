"""Batch of independent MDPSs with identical state and action spaces.
"""

import numpy as np

from numpy import ndarray

from functools import partial
from numpy.random import default_rng, Generator

from typing import Iterator
from .base import State, Action, Observation, Reward

from .base import Env


class MDP(Env):
    """A batched MDP."""

    # the random state (consumed!)
    random_: Generator

    # (N, S, A, X) markov transition kernels
    kernels: ndarray[float]

    # (N, S, A, X) MDP's reward (see `.validate`)
    rewards: ndarray[float]

    # (N,) the current hidden state of each pop member's mdp instance
    state_: State

    def __init__(self, random: Generator, /, kernels, rewards) -> None:
        self.kernels, self.rewards = self.validate(kernels, rewards)
        self.random_ = default_rng(random)  # PRNG is consumed!

    def reset(self) -> tuple[Observation, dict]:
        # sample the initial state at random
        n_population, n_states, _, _ = self.kernels.shape
        self.state_ = self.random_.choice(n_states, size=n_population, p=None)

        # the observation is the state (complete information MDP)
        # XXX `random_` is updated inplace!
        return self.state_.copy(), {}

    def step(self, /, actions: Action) -> tuple[Observation, Reward, bool, dict]:
        # get the member to mdp designation
        index = np.arange(len(self.kernels))

        # draw the next state and take the reward
        probas = self.kernels[index, self.state_, actions]
        state_ = self.random_.multinomial(1, probas, size=None).argmax(-1)
        reward_ = self.rewards[index, self.state_, actions, state_]

        # return the next state and the reward feedback
        # XXX shouldn't it be a vector of bool?
        self.state_ = state_
        return state_.copy(), reward_.astype(float), False, {}

    # runtime props
    n_population = property(lambda self: self.kernels.shape[0])
    n_states = property(lambda self: self.kernels.shape[1])  # == .shape[3]
    n_actions = property(lambda self: self.kernels.shape[2])

    def __repr__(self) -> str:
        return f"MDP({self.n_population}, S={self.n_states}, A={self.n_actions})"

    @staticmethod
    def validate(kernels: ndarray, rewards: ndarray) -> tuple[ndarray, ndarray]:
        # make sure the kernels and rewards broadcast and have shape (N, S, A, X)
        # XXX rewards's shape uniquely determines the expected reward function
        # * `(S, A, X)` -- `r(s, a, x)`
        # * `(1, A, 1)` -- `r(a)`
        # * `(1, 1, X)` -- `r(x)`
        # basically, unit dims correspond to "independence" from the respective
        #  piece of the `(state, action, next_state)` transition.
        shape = np.broadcast_shapes(kernels.shape, (1, 1, 1, 1))
        kers, rews = np.broadcast_arrays(np.broadcast_to(kernels, shape), rewards)

        # check consistency
        *_, n_states, n_actions, n_states_ = kers.shape
        assert n_states == n_states_, kers.shape
        assert np.allclose(kernels.sum(-1), 1.0), "markov kernel is not a probability"
        assert np.all((0 <= kernels) & (kernels <= 1)), "probability must be in [0, 1]"

        return kers / np.sum(kers, axis=-1, keepdims=True), rews

    @classmethod
    def sample(
        cls, random: Generator, /, kernels, rewards, *, n_processes: int = None
    ) -> Iterator["MDP"]:
        """Create instances of batched MDP by sampling from provided kernel-reward pairs.
        """

        random = default_rng(random)  # the prng is consumed!

        # check kernel-reward pair consistency
        kernels, rewards = cls.validate(kernels, rewards)

        # adjust the number of processes
        n_population = len(kernels)
        if n_processes is None:
            n_processes = n_population
        n_processes = min(n_population, n_processes)

        # sub-sample a cohort, or shuffle within population
        get_sample = random.permutation
        if n_population > n_processes:
            get_sample = partial(random.choice, size=n_processes, replace=False)

        # perpetually draw samples of MDPs from the same population
        while True:
            # sub-sample the MDPs
            indices = get_sample(n_population)
            kers = np.take(kernels, indices, axis=0)
            rews = np.take(rewards, indices, axis=0)

            # build an env with a forked PRNG
            yield MDP(*random.spawn(1), kers, rews)
