import tqdm
from time import monotonic, strftime

import numpy as np

from functools import partial

from numpy.random import SeedSequence

from collections.abc import Callable

from ..envs import MDP
from ..policies.base import BasePolicy
from ..envs.base import Env
from .play import rollout
from .utils import sq_spawn, sneak_peek, snapshot_git


def pseudocode() -> None:
    r"""Pseudocode of the experiment, implemented in `run()` of this module.

    Notes
    -----
    We externalize the state of the policy and environment in order to show what
    updated what and when explicitly.

    Let :math:`h_t` be the internal state of the policy (`pol`), and :math:`x_t` --
    be the state of the environment (`env`). The stepping is governed by a step
    in the environment :math:`(x_t, a_t) \to (x_{t+1}, r_{t+1}, f_{t+1})` and
    the feedback :math:`r_{t+1}` is used to assign credit to past actions
    :math:`(a_s)_{s \leq t}`. The flag :math:`f_{t+1}` indicates if the episode
    has natually terminated.
    """

    # `experiment` is a collection of environments to be played with independent
    #  instance of the same policy, and each environment is played for multiple
    #  eposodes
    experiment: list[Env] = []
    n_episodes: int = ...
    policy_factory: type = ...  # policy factory

    # experiment loop (with pre-seeded envs in episodes)
    for env in experiment:
        h, k = [...], 0

        # let the policy rummage through the env for an oracle and init the policy's
        #  "externalized" state
        pol = policy_factory()
        h[0] = pol.init_with_peek(env)

        # multi-episodic rollout loop
        for _ in range(n_episodes):
            # per-episode loop
            x, r, a, f, t = [...], [...], [...], [...], 0
            x[0], done = env.reset(), False  # env "externalized" state
            # XXX `r[0]` is undefined!
            while not done:
                # pol: (h_k, x_t) -->> a_t
                # XXX random decision if `pol` has NEVER been updated!
                a[t] = pol.decide(h[k], x[t])

                # env: (x_t, a_t) -->> (x_{t+1}, r_{t+1}, f_{t+1})
                x[t + 1], r[t + 1], f[t + 1] = env.step(x[t], a[t])

                # report (x[t], a[t], r[t + 1], x[t + 1], f[t + 1]) to an analyzer
                pass

                # pol: (h_k, x_t, a_t, r_{t+1}, x_{t+1}, f_{t+1}) -->> h_{k+1}
                h[k + 1] = pol.update(h[k], x[t], a[t], r[t + 1], x[t + 1], f[t + 1])

                # env's within-episode step counter
                t += 1

                # policy's clock spans acorss multiple episodes
                k += 1


def run(
    entropy: int | None,
    Policy: Callable[[], BasePolicy],
    /,
    kernels,
    rewards,
    *,
    n_processes: int,
    n_budget: int,
    n_experiments: int,
    n_episodes_per_experiment: int,
    n_steps_per_episode: int,
    noise: float,
) -> tuple[BasePolicy, dict]:
    n_population, n_actions, n_states, _ = kernels.shape
    assert 0 < n_budget <= n_processes

    # prepare the policy spawner
    # XXX we assume the state and action spaces are known
    n_steps_per_experiment = n_episodes_per_experiment * n_steps_per_episode
    Creator = partial(Policy, n_steps_per_experiment, n_budget, n_states)

    # prepare the seed sequence for the experiment: the same seed for episodic env
    #   generator, but different for the policy's interactions within an episodes
    main = entropy if isinstance(entropy, SeedSequence) else SeedSequence(entropy)

    # prepare the seed sequences for the experiment: one sequence for the env sampler
    #  (env init and dynamics)m and another one for the policy (init and behaviour)
    sqs_env, sqs_pol = sq_spawn(main, (2, n_experiments))

    # get a deterministically chaotic sampler from a pool of the potential MDPs
    envs = MDP.sampler(sqs_env, kernels, rewards, n_processes=n_processes, noise=noise)

    # setup a rollout with policy updated inplace online after every step
    seeded_online_episode = partial(
        # interact for at most `n_steps_per_episode` or until the episode terminates
        #  since auto is False
        rollout, n_steps=n_steps_per_episode, auto=False, n_steps_per_update=1
    )

    # per experiment loop
    history = []
    for env, sq_pol in zip(envs, tqdm.tqdm(sqs_pol, ncols=60)):
        # print(sq_pol, default_rng(sq_pol).uniform())  # XXX is chaos deterministic?

        # seeds for the policy: one for policy's init to consume, and the rest for
        #  the policy's randomness during its multi-episode rollout in the env
        sq_init, *sq_pol_episodes = sq_pol.spawn(1 + n_episodes_per_experiment)

        # create a new policy from the provided seed
        pol = Creator(random=sq_init)
        gain_gt_advantage = sneak_peek(pol)

        # use a shortcut to access the ground truth about the env
        gain_gt_advantage(env)

        # explicit multi-episode policy rollout: the same policy `pol` (i.e. init and
        #  ground truth advantage) runs and updates online for the total if
        # `n_episodes_per_experiment * n_steps_per_episode` steps through `env`.
        episode_rewards, walltimes = [], [monotonic()]
        for sq_pol in sq_pol_episodes:
            trace = []
            # run a new episode, keeping track of the reward over all arms
            # XXX `rew_` is `(n_steps_per_update, n_arms)`
            for _, _, rew_, _, _ in seeded_online_episode(env, pol, sq_pol):
                trace.append(np.sum(rew_, axis=-1))

            # the trace of rewards gained during the episode
            episode_rewards.append(np.concatenate(trace))
            walltimes.append(monotonic())

        # track whatever the episode runner yielded and per-episode time measurements
        history.append((np.stack(episode_rewards), np.ediff1d(walltimes)))

    # unpack, stack, and then return a dictionary
    episode_rewards, walltimes = map(np.stack, zip(*history))
    return pol, dict(
        # save the name of the policy played during the last replication
        policy_name=repr(pol),
        # the timestamp and git
        __dttm__=strftime("%d%m%Y%H%M%S"),
        __git__=snapshot_git(),
        # entropy used by to seed this experiment
        entropy=main.entropy,
        # `episode_rewards[n, e, j]` the reward for (j+1)-th interaction in
        #  episode `e` of experiment `n`
        episode_rewards=episode_rewards,
        # `walltimes[n, e, j]` the walltime in seconds took by the (j+1)-th
        #  interaction in episode `e` of experiment `n`
        walltimes=walltimes,
        # meta information
        n_processes=n_processes,
        n_budget=n_budget,
        n_states=n_states,
        n_actions=n_actions,
        discount=float("nan"),
        n_experiments=n_experiments,
        n_episodes_per_experiment=n_episodes_per_experiment,
        n_steps_per_episode=n_steps_per_episode,
        noise=noise,
    )


def make_name(
    xp: str = "xp1",
    /,
    *,
    policy_name: str,
    entropy: int,
    n_processes: int,
    n_budget: int,
    n_states: int,
    n_actions: int,
    n_experiments: int,
    n_episodes_per_experiment: int,
    n_steps_per_episode: int,
    noise: float,
    __dttm__: str,
    **ignore,
) -> str:
    """Make a name for this experiment."""

    suffix = "__".join(
        [
            f"n{n_processes}",
            f"b{n_budget}",
            f"s{n_states}",
            f"a{n_actions}",
            f"H{n_steps_per_episode}",
            f"L{n_episodes_per_experiment}",
            f"E{n_experiments}",
            f"{noise!s}",
        ]
    )
    return f"{xp}__{suffix}__{policy_name}__{__dttm__}__{entropy:032X}"
