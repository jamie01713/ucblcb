from numpy.random import default_rng, Generator
from typing_extensions import Self

from ..envs import Observation, Action, Reward


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

        return default_rng(random).choice(self.n_actions, size=len(obs))
