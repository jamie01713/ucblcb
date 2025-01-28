"""Base interface for gym-like environments.
"""

from numpy import ndarray
from numpy.random import Generator

from typing import TypeVar


Observation = State = TypeVar("State", bound=ndarray[int])
Action = TypeVar("Action", bound=ndarray[int])
Reward = TypeVar("Reward", bound=float) | ndarray[float]
Done = TypeVar("Done", bound=bool) | ndarray[int] | ndarray[bool]


class Env:
    """Base environment interface.

    Attributes
    ----------
    random_: Generator
        The consumable PRNG this environment draws randomness from whenever it pleases.
    """

    random_: Generator

    def reset(self) -> tuple[Observation, dict]:
        raise NotImplementedError

        return 0, {}

    def step(self, /, actions: Action) -> tuple[Observation, Reward, Done, dict]:
        raise NotImplementedError

        return 0, 0.0, True, {}
