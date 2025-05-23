"""Transition collection iterator and streaming rollout with on-policy update."""

import numpy as np
from numpy.random import default_rng, Generator, SeedSequence

from jax import tree_util as tu
from functools import partial

from typing import Iterator, Iterable, TypeVar
from collections.abc import Callable

from ..envs.base import Env, Observation, Action, Reward, Done
from ..policies.base import BasePolicy


T = TypeVar("T")


def default_collate(pytrees: list[T]) -> T:
    """Collate pytrees with array-like leaves preserving tree structure"""

    return tu.tree_map(lambda *x: np.stack(x), *pytrees)


def batched(iterable: Iterable, n: int) -> Iterable:
    """Batch transitions into tuples of length n. The last batch may be shorter."""

    buffer = []
    # keep collecting items into a buffer until we get n of them or exhaust it
    for item in iterable:
        buffer.append(item)
        if len(buffer) < n:
            continue

        # yield the complete batch
        yield tuple(buffer)
        buffer = []

    # do not forget the last partial batch
    if buffer:
        yield tuple(buffer)


# (x_t, a_t, r_{t+1}, x_{t+1}, F_{t+1})
Transtition = tuple[Observation, Action, Reward, Observation, Done]


def core_rollout(
    env: Env,
    decide: Callable[[Observation], Action],
    /,
    n_steps: int | None,
    auto: bool,
) -> Iterator[Transtition]:
    """Interact for n_steps, until termination, or indefinitely with auto-reset."""

    # make sure to reset the environment on the VERY first step
    step, obs_, done = 0, None, True

    # terminate if exceeded the steps cap
    while n_steps is None or step < n_steps:
        # local time tick `t-1 -> t` (`x` becomes `s`, and, optionally, env is reset)
        obs, _ = env.reset() if done else (obs_, {})

        # x_t --decide-->> a_t: `act` is a non-batched action in the env `a_t`
        act = decide(obs)  # XXX `decide`, like `env`, is stateful!

        # (x_t, a_t) --env-->> (r_{t+1}, x_{t+1}, F_{t+1}), where `F_{t+1}`
        #   indicates if the env's episode got naturally terminated
        obs_, rew_, done, _ = env.step(act)  # XXX info dict is not used
        step += 1

        # replace the binary `done` data with a the tri-state flag:
        #  <0: rollout is truncated after x_{t+1}
        #  =0: trajectory continues after x_{t+1}
        #  >0: episode has terminated at x_{t+1}
        done = -1 if step == n_steps else +1 if done else 0

        # return the x_t, a_t, r_{t+1}, x_{t+1}, F_{t+1} transition
        # XXX reward due to `t-1 -> t` transition is not used, because at state
        #  `x_t` we took action `a_t` and got `r_{t+1}` as feedback (env's local
        #  step ticked from `t` to `t+1`)!
        yield obs, act, rew_, obs_, done

        # break if the env terminated and we are not auto-resetting
        if done and not auto:
            return


def base_rollout(
    env: Env,
    /,
    decide: Callable[[Observation], Action],
    update: Callable[[Observation, Action, Reward, Observation, Done], None],
    *,
    n_steps: int,
    auto: bool = False,
    n_steps_per_update: int = 1,
    collate: Callable[[list[T]], T] = default_collate,
) -> Iterator[Transtition]:
    """Stream transitions from the online rollout with on-policy updates."""

    assert n_steps is None or isinstance(n_steps, int), n_steps

    # launch the transition collection loop: it plays the policy in the env and
    #  streams transitions into the consuming for-loop. The policy may not be
    #  static, and in fact is updated online in the batched for-loop below.
    # XXX `core_rollout` is an ITERATOR FUNCTION, runing in lockstep with the for-loop
    liveloop = core_rollout(env, decide, n_steps=n_steps, auto=auto)

    # interact with the env to generate experience in batches
    # XXX batch[j] is `(x^j_t, a^j_t, r^j_{t+1}, x^j_{t+1}, F^j_{t+1})`
    for batch in batched(liveloop, n_steps_per_update):
        # `update` expects a leading batch dimension, so we unpack and stack
        obs, act, rew_, obs_, done = collate(batch)

        # update with the observed ON-POLICY `sarx` transition data
        update(obs, act, rew_, obs_, done)

        # send the batched transition used for the update `(n_steps_per_update, ...)`
        # XXX consecutive transitions of the same trajectory (unless auto-reset)
        yield obs, act, rew_, obs_, done


def rollout(
    env: Env,
    pol: BasePolicy,
    /,
    random: Generator | SeedSequence,
    *,
    n_steps: int,
    auto: bool = False,
    n_steps_per_update: int = 1,
) -> Iterator[Transtition]:
    """Do on-policy update over the online rollout, streaming consecutive transitions."""

    assert n_steps is None or isinstance(n_steps, int), n_steps
    assert isinstance(pol, BasePolicy), type(pol)

    # the generator `random` is consumed by the policy's `.update` and `.decide`
    #  redefined below
    random = default_rng(random)

    # wrap the single-step observation into a unit-sized batch and decide the action
    def pol_decide(obs: Observation) -> Action:
        # side-effects: `random` (`pol` should not be updated in-place by .decide)
        singleton = tu.tree_map(lambda x: np.expand_dims(x, 0), obs)
        return pol.decide(random, singleton)[0]

    # update `pol` with the observed transitions
    # XXX `pol` is updated INPLACE, and `random` is consumed!
    pol_update = partial(pol.update, random=random)

    # stream from the online updated policy rollout
    yield from base_rollout(
        env,
        pol_decide,
        pol_update,
        n_steps=n_steps,
        auto=auto,
        n_steps_per_update=n_steps_per_update,
        collate=default_collate,
    )
