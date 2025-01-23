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
    """A vectorized finite-state markov decision process environment."""

    # the random state CONSUMED by the environment!
    random_: Generator

    # (N, S, A, X) Markov transition probability `p_n(S'=x | S=s, A=a)`
    kernels: ndarray[float]

    # (N, S, A, X) MDP's reward function `r_n(s, a, x)` (see `.validate`)
    rewards: ndarray[float]

    # (N,) the joint state is made up of state of each mdp instance in the environment
    state_: State

    def __init__(self, random: Generator, /, kernels, rewards) -> None:
        self.kernels, self.rewards = self.validate(kernels, rewards)
        self.random_ = default_rng(random)  # the PRNG `random` is consumed!

    def reset(self) -> tuple[Observation, dict]:
        # sample the initial state at random
        n_population, n_states, _, _ = self.kernels.shape
        self.state_ = self.random_.choice(n_states, size=n_population, p=None)

        # the observation is the state (complete information MDP)
        # XXX `random_` is consumed and updated inplace!
        return self.state_.copy(), {}

    def step(self, /, actions: Action) -> tuple[Observation, Reward, bool, dict]:
        # get the member-to-mdp designation
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
        r"""Make sure the kernels and rewards broadcast and have shape (N, S, A, X).

        Notes
        -----
        Rewards's shape uniquely determines the reward function:

        * `(S, A, X)` is :math:`r(s, a, x)`, i.e. reward for entering `x` from
          `s` by doing `a`. Avoids baking :math:`p(x | s, a)` into the expected
          reward :math:`R(s, a) = E_{p(x | s, a)} r(s, a, x)` prematurely.

        * `(1, A, 1)` is `r(a)` -- depends on action only! (i.e. movement
          cost in shortest path control problems)

        * `(1, 1, X)` is `r(x)` -- reward for entering or being at `x`
        * `(S, 1, 1)` is `r(s)` -- reward for leaving `s`

        Basically, unit dims correspond to "independence" from the respective piece
        of the `(state, action, next_state)` transition.
        """

        # broadcast
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
        """Create instances of batched MDP by sampling from provided kernel-reward pairs."""

        random = default_rng(random)  # the PRNG `random` is consumed!

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
