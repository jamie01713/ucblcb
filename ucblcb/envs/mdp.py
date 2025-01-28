"""Batch of independent MDPSs with identical state and action spaces.
"""

import numpy as np

from numpy import ndarray

from functools import partial
from numpy.random import default_rng, Generator, SeedSequence

from typing import Iterator, Iterable
from .base import State, Action, Observation, Reward, Done

from .base import Env


class MDP(Env):
    """A vectorized finite-state markov decision process environment."""

    # the random state CONSUMED by the environment!
    random_: Generator

    # (N, A, S, X) Markov transition probability `p_n(S'=x | S=s, A=a)`
    kernels: ndarray[float]

    # (N, A, S, X) MDP's reward function `r_n(a, s, x)` (see `.validate`)
    rewards: ndarray[float]

    # (N,) the joint state is made up of state of each mdp instance in the environment
    state_: State

    def __init__(
        self, random: Generator, /, kernels, rewards, *, noise: float = 0.0
    ) -> None:
        self.kernels, self.rewards = self.validate(kernels, rewards)
        self.random_ = default_rng(random)  # the PRNG `random` is consumed!

        # allow different per-arm noise level
        self.noise = np.broadcast_to(np.maximum(0.0, noise), self.kernels.shape[:-3])

    def reset(self) -> tuple[Observation, dict]:
        # sample the initial state at random
        n_population, _, n_states, _ = self.kernels.shape
        self.state_ = self.random_.choice(n_states, size=n_population, p=None)

        # the observation is the state (complete information MDP)
        # XXX `random_` is consumed and updated inplace!
        return self.state_.copy(), {}

    def step(self, /, actions: Action) -> tuple[Observation, Reward, Done, dict]:
        # get the member-to-mdp designation
        index = np.arange(len(self.kernels))

        # draw the next state and take the reward
        # XXX `.multinomial` always draws from the distribution over the last axis
        probas = self.kernels[index, actions, self.state_, :]
        state_ = self.random_.multinomial(1, probas, size=None).argmax(-1)
        reward_ = self.rewards[index, actions, self.state_, state_].astype(float)

        # Add N(0, \simga^2) noise to `r_{t+1}`
        # XXX sample the noise anyway in order to make PRNG consumption predictable
        reward_ += self.random_.normal(size=reward_.shape) * self.noise

        # return the next state and the reward feedback
        # XXX shouldn't it be a vector of bool?
        self.state_ = state_
        return state_.copy(), reward_, False, {}

    # runtime props
    n_population = property(lambda self: self.kernels.shape[0])
    n_actions = property(lambda self: self.kernels.shape[1])
    n_states = property(lambda self: self.kernels.shape[2])  # == .shape[3]

    def __repr__(self) -> str:
        return f"MDP({self.n_population}, S={self.n_states}, A={self.n_actions})"

    @staticmethod
    def validate(kernels: ndarray, rewards: ndarray) -> tuple[ndarray, ndarray]:
        r"""Make sure the kernels and rewards broadcast and have shape (N, S, A, X).

        Notes
        -----
        Rewards's shape uniquely determines the reward function:

        * `(A, S, X)` is :math:`r(a, s, x)`, i.e. reward for entering `x` from
          `s` by doing `a`. Avoids baking :math:`p(x | s, a)` into the expected
          reward :math:`R(s, a) = E_{p(x | s, a)} r(a, s, x)` prematurely.

        * `(A, 1, 1)` is `r(a)` -- depends on action only! (i.e. movement
          cost in shortest path control problems)

        * `(1, 1, X)` is `r(x)` -- reward for entering or being at `x`
        * `(1, S, 1)` is `r(s)` -- reward for leaving `s`

        Basically, unit dims correspond to "independence" from the respective piece
        of the `(state, action, next_state)` transition.
        """

        # broadcast
        shape = np.broadcast_shapes(kernels.shape, (1, 1, 1, 1))
        kers, rews = np.broadcast_arrays(np.broadcast_to(kernels, shape), rewards)

        # check consistency
        *_, n_actions, n_states, n_states_ = kers.shape
        assert n_states == n_states_, kers.shape
        assert np.allclose(kernels.sum(-1), 1.0), "markov kernel is not a probability"
        assert np.all((0 <= kernels) & (kernels <= 1)), "probability must be in [0, 1]"

        return kers / np.sum(kers, axis=-1, keepdims=True), rews

    @classmethod
    def sampler(
        cls,
        seedseqs: Iterable[SeedSequence],
        /,
        kernels,
        rewards,
        *,
        n_processes: int = None,
        shuffle: bool = True,
        **kwargs,
    ) -> Iterator["MDP"]:
        """Create instances of batched MDP by sampling from the kernel-reward pairs."""

        # adjust the number of processes
        n_population = len(kernels)
        if n_processes is None:
            n_processes = n_population
        n_processes = min(n_population, n_processes)

        # sub-sample a cohort w/o replacement, or shuffle within population
        if n_population > n_processes:
            draw = partial(Generator.choice, size=n_processes, replace=False)

        elif n_population == n_processes:
            draw = Generator.permutation if shuffle else (lambda _, m: np.arange(m))

        # oversample if the number of requested arms (processes) is larger then
        #  the population
        else:
            draw = partial(Generator.choice, size=n_processes, replace=True)

        # check kernel-reward pair consistency
        kernels, rewards = cls.validate(kernels, rewards)

        # loop over the seeds in the seed sequence and produce and environment
        for random in map(default_rng, seedseqs):
            # the PRNG `random` is consumed for sampling the environment's processes
            #  and then gets stolen by the env iteself for its own consumption
            jx = draw(random, n_population)

            # give up this RPGN to the MD for consumption
            yield MDP(
                random,
                np.take(kernels, jx, axis=0),
                np.take(rewards, jx, axis=0),
                **kwargs,
            )

    @classmethod
    def sample(
        cls,
        random: SeedSequence | Generator,
        /,
        kernels,
        rewards,
        *,
        n_processes: int = None,
        **kwargs,
    ) -> Iterator["MDP"]:
        """Sample MDP env instances from the given population."""

        random = default_rng(random)  # the PRNG `random` is consumed!

        # perpetually draw samples of MDPs from the same population
        stream = iter(lambda: random.spawn(1)[0], None)
        yield from cls.sampler(
            stream, kernels, rewards, n_processes=n_processes, **kwargs
        )


def random_mdp(
    random: Generator, /, n_states: int, n_actions: int, size: tuple[int, ...] = None
) -> tuple[ndarray, ndarray]:

    random = default_rng(random)
    size = () if size is None else size

    # get a pool of markov transition kernels `k_{asx} = p(x \mid s, a)`
    kernels = random.uniform(size=(*size, n_actions, n_states, n_states))
    kernels /= np.sum(kernels, axis=-1, keepdims=True)

    # random rewards `r_{asx} = r_x`, i.e. a feedback for entering a state
    return kernels, random.normal(size=(*size, 1, 1, n_states))
