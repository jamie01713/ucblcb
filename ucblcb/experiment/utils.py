"""Rollout loop, episodic update loop, and experimnet seeding.
"""

import re
import warnings

from importlib import import_module

import numpy as np
from functools import partial

from numpy import ndarray
from numpy.random import default_rng, Generator, SeedSequence

from functools import wraps

from typing import Iterator
from collections.abc import Callable

from ..envs.base import Env, Observation, Action
from ..policies.base import BasePolicy


def rollout(
    env: Env,  # non-batched env!
    /,
    policy: Callable[[Observation], Action],
    n_steps: int | None = None,
) -> Iterator[tuple[int, tuple, bool]]:
    """Play the given policy in the env for n_steps or until termination."""
    # set the initial rewards to NaN, because `.reset` sort of implies that
    #  the next `.step` is the VERY first interactions with the MDP. Hence,
    #  there is simply no reward to be had because of not prior interaction
    rew_, done, step = np.full(env.n_population, np.nan), False, 0  # r_0

    # reset the env and get its initial observation
    obs_, _ = env.reset()  # x_0
    while (n_steps is None or step < n_steps) and not done:
        # local time tick `t-1 -> t` (`x` becomes `s`)
        obs, rew = obs_, rew_  # noqa: F841
        # XXX `obs`, `act`, `rew`, `obs_`,    and `rew_` are
        #     `x_t`, `a_t`, `r_t`, `x_{t+1}`, and `r_{t+1}`, respectively.

        # `act` is `a_t` array of int of shape (P,)
        # x_t --policy-->> a_t
        # XXX `policy`, like env, can be stateful
        act = policy(np.expand_dims(obs, 0))[0]  # XXX single-element batch!

        # (x_t, a_t) --env-->> (r_{t+1}, x_{t+1})
        # XXX reward due to `t-1 -> t` transition is not used (`rew`), because
        #  at state `x_t` we took action `a_t` and got `r_{t+1}` as feedback
        #  (env's local step ticked from `t` to `t+1`)!
        obs_, rew_, done, _ = env.step(act)

        # return the x_t, a_t, r_{t+1}, x_{t+1} transition
        step += 1
        yield step, (obs, act, rew_, obs_), done


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

    random = default_rng(random)  # the PR generator `random` is consumed!

    def pol_update(batch):
        # side-effects from outer scope: `pol`, `random`
        assert batch

        # `pol.update` expects a leading batch dimension, so we unpack
        #   the data and create one of size `len(batch)`
        obs, act, rew_, obs_ = map(np.stack, zip(*batch))

        # update the policy with the observed `sarx` transition
        # XXX `pol` is updated INPLACE, and the PRNG is consumed!
        pol.update(obs, act, rew_, obs_, random=random)

        # keep track of the total reward from the transitions
        return obs, act, rew_, obs_

    # supply the policy with its own PRNG
    pol_decide = partial(pol.decide, *random.spawn(1))

    # interact with a new mdp env for `n_steps` to generate new experience
    trace, buffer = [], []
    for step, sarx, done in rollout(env, pol_decide, n_steps):
        # save `sarx = (x_t, a_t, r_{t+1}, x_{t+1})`
        buffer.append(sarx)
        if len(buffer) < n_batch_size_per_update:
            continue

        # update the policy with the observed batch of `sarx` transitions
        _, _, rew_, _ = pol_update(tuple(buffer))

        # keep track of the total reward from the transitions
        trace.append(np.sum(rew_, axis=-1))  # respect the batch dim!

        buffer = []

    # do not forget the last incomplete batch
    if buffer:
        _, _, rew_, _ = pol_update(tuple(buffer))
        trace.append(np.sum(rew_, axis=-1))

    return np.concatenate(trace)


def sq_spawn(
    sq: SeedSequence | Generator, /, *shapes: tuple[int, ...], axis: int = 0
) -> ndarray[SeedSequence | Generator]:
    """Prepare seeds for the experiment."""

    # produce shaped seed sequence spawns
    arys = []
    for shape in shapes:
        # get an zero-data shape and spawn from the seed sequence
        # XXX see `np.broadcast_shapes(...)`
        x = np.empty(shape, dtype=np.dtype([]))
        sprout = np.reshape(sq.spawn(x.size), x.shape)

        arys.append(sprout)

    # broadcast and stack them over the specified dim
    return np.stack(np.broadcast_arrays(*arys), axis)


def sneak_peek(
    pol: BasePolicy, /, method: str = "sneak_peek", *, honest: bool = True
) -> Callable[[Env], None]:
    """Brake open the black box and let the policy rummage in it for unfair advantage.

    Notes
    -----
    The title is self-explanatory.

    Any policy that implements the method and/or makes use of it on an environment
    that it is played in should be ashamed of itself, condemned by its peers, and
    shunned by everybody.
    """

    peeker = getattr(pol, method, None)
    if not callable(peeker):
        return lambda _: None

    if not honest:
        return peeker

    @wraps(peeker)
    def peeker_with_alert(env: Env) -> None:
        message = f"policy {pol}({id(pol)}) took a peek into the black box"
        warnings.warn(message, RuntimeWarning)
        return peeker(env)

    return peeker_with_alert


def snapshot_git(path=None, *, diff: bool = True) -> dict:
    """Save some minimal version control info so that past results
    are easier to recover if something gets irreversibly forgotten.
    """

    try:
        from git import Repo, InvalidGitRepositoryError

        # open the repo
        repo = Repo(path, search_parent_directories=True)

        # get the head commit sha
        head = str(repo.head.commit)

        # get diff againts head
        diff = repo.git.diff(None) if diff else ""

    except (InvalidGitRepositoryError, ImportError):
        head, diff = "", ""

    return {"head": head, "diff": diff}


def from_qualname(spec: str | type) -> type:
    """Parse the specified qualname, import it and return the type."""
    if isinstance(spec, type):
        return spec

    # remove the text-wrapper resulting from `str(type(obj))`
    match = re.fullmatch(r"^<(?:class)\s+'(.*)'>$", spec)
    qualname = spec if match is None else match.group(1)

    # import from built-ins if no module was detected
    module, dot, name = qualname.rpartition(".")
    return getattr(import_module(module if dot else "builtins"), name)
