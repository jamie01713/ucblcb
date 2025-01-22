import numpy as np

from numpy import ndarray
from numpy.random import default_rng, Generator

from typing import Any
from typing_extensions import Self

from ..envs.base import Observation, Action, Reward


def check_consistent_shapes(*arys, axes: int | tuple[int, ...] = 2):
    """Make sure the specified axes conincide."""

    axes = (axes,) if isinstance(axes, int) else axes

    shapes = tuple(np.shape(x) for x in arys if x is not None)
    if not shapes:
        return

    shape = shapes[0]
    if all(shape[a] == x[a] for x in shapes for a in axes):
        return

    raise ValueError(shapes)


def check_n_arms(policy, /, n_arms: int, *, initialize: bool = True) -> bool:
    """Set or check the number of managed arms from the observed state."""

    # the current number of managed arms `n_arms_in_`
    n_arms_in_ = getattr(policy, "n_arms_in_", None)

    # initialize and indicate the first call
    if n_arms_in_ is None:
        if initialize:
            # setattr(policy, "n_arms_in_", n_arms)
            policy.n_arms_in_ = n_arms

        return True

    # check __exact__ consistency with an earlier initialized action space
    if n_arms == n_arms_in_:
        return False

    raise ValueError(
        f"`{n_arms=}` is not the same as on the last call `{n_arms_in_=}`."
    )


def check_observation_space(policy, /, obs: Any, *, reset=False) -> None:
    """Set the `shape_in_` attribute, or check the data against it."""

    # get the number of features in 2d+ data
    _, _, *features = np.shape(obs)

    # shortcut if we are resetting
    shape = tuple(features)
    if reset:
        # setattr(policy, "shape_in_", shape)
        policy.shape_in_ = shape
        return

    # we may get legitimately called on an uninitialized `policy`
    shape_in_ = getattr(policy, "shape_in_", None)
    if shape_in_ is not None and shape != shape_in_:
        raise ValueError(
            f"observation has {shape} features, but we are "
            f"expecting {shape_in_} features as input."
        )


def random_subset(
    random: Generator, /, n_arms: int, budget: int, *, size: int | tuple[int, ...] = ()
) -> ndarray[int]:
    """Pick random subset from `n_arm` elements of specifed size `budget`."""

    size = size if isinstance(size, tuple) else (size,)

    # produce a random samaple of budget-sized subsets of arms
    # XXX we don't care if we are under budget
    arms = np.broadcast_to(np.arange(n_arms), (*size, n_arms))
    indices = random.permuted(arms, axis=-1)

    # get the binary interaction mask (integers)
    # XXX we don't care if we are under budget
    subsets = np.zeros(arms.shape, int)
    np.put_along_axis(subsets, indices[..., :budget], 1, axis=-1)

    return subsets


class BasePolicy:
    """The base policy interface."""

    # total number of interaction steps over the policy's lifetime
    n_max_steps: int | None = None

    # maximal number of arms we can interact at each step (others get action `zero`)
    budget: int

    # attributes
    shape_in_: tuple[int, ...]
    n_arms_in_: int  # the number of arms to pick subsets from

    n_pulls_a_: ndarray[int]
    avg_rew_a_: ndarray[float]

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        /,
        *,
        random: Generator = None,
    ) -> None:
        assert isinstance(budget, int) and budget >= 0
        self.budget = budget

        assert n_max_steps is None or n_max_steps > 0
        self.n_max_steps = n_max_steps

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
        random = default_rng(random)
        assert np.ndim(act) == np.ndim(rew) == 2

        # check the observation space, validate or initialize the number of arms
        #  and holistically validate the data
        # XXX we reset on the first call, and set `n_arms_in_`
        check_consistent_shapes(obs, act, rew, new, axes=(0, 1))

        # make sure the `.n_arms_in_` are initialized: act is `(n_samples, n_arms,)`
        _, n_arms = np.shape(act)
        is_first_call = check_n_arms(self, n_arms, initialize=False) or not hasattr(
            self, "n_arms_in_"
        )
        # XXX we should probably make sure the valeus in `act` are integers between
        #  `0` and `n_actons-1`, where `n_actons` is the size of the action space.

        # see if the observation and rewards are consistent with what used in
        #  the previous call
        check_observation_space(self, obs, reset=is_first_call)
        if is_first_call:
            # by design, this should never fail at this point
            check_n_arms(self, n_arms, initialize=True)

        # call setup if we were not initialized earlier
        if is_first_call:
            self.setup_impl(obs, act, rew, new, random=random)

        return self.update_impl(obs, act, rew, new, random=random)

    def decide(self, random: Generator, /, obs: Observation) -> Action:
        """Decide the action to play to each arm at the provided observed state.

        Parameters
        ----------
        random: Generator
            Consumable PRNG instance.

        obs: pytree, with leaves of shape `(n_samples, n_arms, ...)`
            Pytree of the current observed state `x_t` from the multi-process
            environment (where we can pull multiple arm within the budget).
        """
        assert isinstance(random, Generator)

        # make sure that the observations are provided and consistent
        check_observation_space(self, obs)

        # get the number of arms in multi-arm 2d+ data
        _, n_arms, *_ = np.shape(obs)

        # until we have been updated, we resort to random assignments
        never_updated = check_n_arms(self, n_arms, initialize=False)
        if never_updated:
            return self.uninitialized_decide_impl(random, obs)

        # call the `decide` implementation
        return self.decide_impl(random, obs)

    def setup_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        # the base pocy tracks the number of times each arm has been interacted
        #  with and the average per-arm reward it yielded
        self.n_pulls_a_ = np.zeros(self.n_arms_in_, int)
        self.avg_rew_a_ = np.zeros(self.n_arms_in_, float)

        return self

    def update_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        # we make sure to keep track of pull counts and the reward for each pull
        # XXX this does `\mu_{n+m} - \mu_n = \frac{m}{n+m} (\bar{x}_m - \mu_n)`
        #     with per-arm m, n, and \bar{x}_m
        counts_ = np.sum(act, axis=0)  # XXX `act` is `(B, N)`
        update_ = np.sum(rew, axis=0, where=act > 0) - counts_ * self.avg_rew_a_

        # first, we make sure to keep track of pull counts
        self.n_pulls_a_ += counts_

        # then, update the average per arm reward estimate
        # XXX if we received no new samples for some `j` then `update_[j] = 0`
        #  and this zero is propagated through `.divide`, which is what we want
        np.divide(update_, self.n_pulls_a_, where=self.n_pulls_a_ > 0, out=update_)
        self.avg_rew_a_ += update_

        # XXX do we assume partially observed rewards, i.e. only for the arms
        #  for which `act != 0`?
        return self

    def uninitialized_decide_impl(self, random: Generator, /, obs):
        """Draw a subset of processes at random to play."""
        # default policy to restort to until the first `.update`.

        # get the number of arms in multi-arm 2d+ data
        n_samples, n_arms, *_ = np.shape(obs)
        return random_subset(random, n_arms, self.budget, size=n_samples)

    def decide_impl(self, random: Generator, /, obs):
        raise NotImplementedError


class RandomSubsetPolicy(BasePolicy):
    """A random subset policy for multi-arm subset action spaces."""

    decide_impl = BasePolicy.uninitialized_decide_impl
