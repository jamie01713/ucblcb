"""Rollout and episodic update loop."""

import numpy as np

from numpy import ndarray
from numpy.random import default_rng, Generator

from typing import Iterator
from collections.abc import Callable

from ..envs.base import Env, Observation, Action, Reward, Done
from ..policies.base import BasePolicy


# (x_t, a_t, r_{t+1}, x_{t+1}, f_{t+1})
Transtition = tuple[Observation, Action, Reward, Observation, Done]


def rollout(
    env: Env, pol: Callable[[Observation], Action], /, *, auto: bool,
) -> Iterator[Transtition]:
    """Play the given policy in the env indefinitely, auto resetting if terminated."""

    # make sure to reset the environment on the VERY first step
    obs_, done = None, True
    while True:
        # local time tick `t-1 -> t` (`x` becomes `s`, and, optionally, env is reset)
        obs, _ = env.reset() if done else (obs_, {})

        # x_t --pol-->> a_t: `act` is a non-batched action in the env `a_t`
        act = pol(obs)  # XXX `pol`, like `env`, is stateful!

        # (x_t, a_t) --env-->> (r_{t+1}, x_{t+1}, f_{t+1}), where `f_{t+1}`
        #   indicates if the env's episode got naturally terminated
        obs_, rew_, done, _ = env.step(act)  # XXX info dict is not used

        # return the x_t, a_t, r_{t+1}, x_{t+1}, f_{t+1} transition
        # XXX reward due to `t-1 -> t` transition is not used, because at state
        #  `x_t` we took action `a_t` and got `r_{t+1}` as feedback (env's local
        #  step ticked from `t` to `t+1`)!
        yield obs, act, rew_, obs_, (+1 if done else 0)

        # break if the env's episode terminated and we are not auto-resetting
        if done and not auto:
            return


def play(
    env: Env,
    pol: Callable[[Observation], Action],
    /,
    n_steps: int | None,
    *,
    auto: bool,
) -> Iterator[Transtition]:
    """Play the policy in the env for n_steps or until termination, unless auto."""

    # launch the transition collection loop, which generates and streams transitions
    #  under the given policy (which may not be static):
    #  (x_t, a_t -> r_{t+1}, x_{t+1}, F_{t+1})
    loop = rollout(env, pol, auto=auto)

    # short-circuit if there is no cap on the number of steps
    if not isinstance(n_steps, int):
        yield from loop
        return

    elif n_steps <= 0:
        return

    for step, (*sarx, done) in enumerate(loop, 1):
        # replace the binary `done` data with a the tri-state flag:
        #  <0: rollout is truncated after x_{t+1}
        #  =0: trajectory continues after x_{t+1}
        #  >0: episode has terminated at x_{t+1}
        yield *sarx, (-1 if step == n_steps else done)

        # terminate if exceeded the steps cap
        if step >= n_steps:
            return


def episode(
    random: Generator,
    /,
    env: Env,
    pol: BasePolicy,
    n_steps: int,
    *,
    n_batch_size_per_update: int = 1,
) -> ndarray:
    """Online update over one episode for n_steps or until the env terminates."""

    assert isinstance(pol, BasePolicy), type(pol)
    assert isinstance(n_steps, int), n_steps

    # the generator `random` is consumed by the policy's `.update` and `.decide`
    #  redefined below
    random = default_rng(random)

    # wrap the single-step observation into a unit-sized batch and decide the action
    def pol_decide(obs: Observation) -> Action:
        # side-effects: `random`
        return pol.decide(random, np.expand_dims(obs, 0))[0]

    # update the policy on the collected transition batch
    def pol_update(batch: tuple[Transtition]) -> Transtition:
        # side-effects: `pol`, `random`
        assert batch

        # `pol.update` expects a leading batch dimension, so we unpack
        #   the data and create one of size `len(batch)`
        obs, act, rew_, obs_, fin_ = map(np.stack, zip(*batch))

        # update the policy with the observed `sarx` transition
        # XXX `pol` is updated INPLACE, and the PRNG is consumed!
        pol.update(obs, act, rew_, obs_, fin_, random=random)

        # keep track of the total reward from the transitions
        return obs, act, rew_, obs_, fin_

    # reward trace and a transition buffer for batched updates (size >= 1)
    trace, buffer = [], []

    # interact with a new mdp env for `n_steps` to generate new experience
    # XXX `rollout` is an ITERATOR FUNCTION, whose body is run concurrently lockstep
    #  with the body of this `for-loop-and-a-half`.
    for sarxf in play(env, pol_decide, n_steps, auto=False):
        buffer.append(sarxf)  # save `(x_t, a_t, r_{t+1}, x_{t+1}, f_{t+1})`
        if len(buffer) < n_batch_size_per_update:
            continue

        # update the policy with the observed batch of `sarx` transitions
        _, _, rew_, _, _ = pol_update(tuple(buffer))

        # keep track of the total reward from the transitions
        trace.append(np.sum(rew_, axis=-1))  # XXX respect the batch dims!

        buffer = []

    # do not forget the last incomplete batch
    if buffer:
        _, _, rew_, _, _ = pol_update(tuple(buffer))
        trace.append(np.sum(rew_, axis=-1))

    # concatenate all the steps (over the batch dims)
    return np.concatenate(trace)
