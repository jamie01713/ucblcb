import numpy as np

from numpy.random import default_rng, Generator
from typing_extensions import Self

from ..envs.base import Observation, Action, Reward


class BasePolicy:
    """The base policy interface. Works as a random action picker."""

    n_actions: int

    def __init__(self, n_actions: int, /) -> None:
        self.n_actions = n_actions

    def update(
        self,
        /,
        obs: Observation,
        act: Action,
        rew: Reward,
        new: Observation,
        *,
        random: Generator = None,
    ) -> Self:
        """Update the policy on the provided batch of transitions.

        Parameters
        ----------
        obs: pytree, with leaves of shape `(batch, ...)`
            Observed state `x_t`.

        act: pytree, with leaves of shape `(batch, ...)`
            Action `a_t` taken at `x_t`.

        rew: pytree, with leaves of shape `(batch, ...)`
            Reward feedback `r_{t+1}` for playing `a_t` at `x_t`.

        new: pytree, with leaves of shape `(batch, ...)`
            Next observed state `x_{t+1}` due to playing `a_t` at `x_t`.

        random: Generator, optional
            Consumable PRNG instance.
        """

        return self

    def decide(self, random: Generator, /, obs: Observation) -> Action:
        """Decide the action to play at the provided observed state.

        Parameters
        ----------
        random: Generator
            Consumable PRNG instance.

        obs: pytree, with leaves of shape `()`
            Current observed state `x_t`.
        """
        assert isinstance(random, Generator)

        return random.choice(self.n_actions, size=len(obs))


class RandomSubsetPolicy(BasePolicy):
    """A random subset policy for binary action spaces."""

    n_actions: int
    budget: int

    def __init__(
        self, n_actions: int, /, budget: int
    ) -> None:
        assert n_actions == 2
        self.n_actions = n_actions

        assert budget >= 0
        self.budget = budget

    def decide(self, random: Generator, /, obs: Observation) -> Action:
        """Draw a subset of processes at random to play."""
        assert isinstance(random, Generator)

        # we don't care if we are under budget
        indices = random.permutation(len(obs))[:self.budget]

        # get the binary interaction mask (integers)
        actions = np.zeros(len(obs), int)
        np.put_along_axis(actions, indices, 1, axis=-1)

        return actions
