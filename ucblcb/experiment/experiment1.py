import tqdm
from time import monotonic, strftime

import numpy as np

from functools import partial

from numpy.random import SeedSequence

from collections.abc import Callable

from ..envs import MDP
from ..policies.base import BasePolicy
from ..envs.base import Env
from .utils import sq_spawn, sneak_peek, snapshot_git, episode as play


def pseudocode() -> None:
    r"""Pseudocode of the experiment, implemented in `run()` of this module.

    Notes
    -----
    We externalize the state of the policy and environment in order to show what
    updated what and when explicitly.

    Let :math:`h_t` be the internal state of the policy (`pol`), and :math:`x_t` --
    be the state of the environment (`env`). The stepping is governed by a step
    in the environment :math:`(x_t, a_t) \to (x_{t+1}, r_{t+1})` and the feedback
    :math:`r_{t+1}` is used to assign credit to past actions :math:`(a_s)_{s \leq t}`.
    """

    # `experiment` is a collection of series to be played with DIFFERENT policies,
    #  and `series` is a list of environments to play with the same policy.
    experiment: list[tuple[list[Env], Callable[[], BasePolicy]]] = []

    # experiment loop
    for episodes, Player in experiment:
        pol = Player()

        # series of episodes loop
        h, k = [...], 0
        h[0] = pol.reset()
        for env in episodes:
            # let the policy rummage through the env for an oracle
            pol.peek(env)

            # per-episode loop
            x, r, a, t = [...], [...], [...], 0
            x[0], done = env.reset(), False
            # XXX unlike the original impl., we keep `r[0]` undefined!
            while not done:
                # pol: (h_k, x_t) -->> a_t
                a[t] = pol.decide(h[t], x[t])

                # env: (x_t, a_t) -->> (x_{t+1}, r_{t+1})
                x[t + 1], r[t + 1], done = env.step(x[t], a[t])

                # pol: (h_k, x_t, a_t, r_{t+1}, x_{t+1}) -->> h_{k+1}
                h[k + 1] = pol.update(h[k], x[t], a[t], r[t + 1], x[t + 1])

                # env's within-episode steps
                t += 1

                # policy's lifetime spanning acorss multiple episodes
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
) -> None:
    n_population, n_states, n_actions, _ = kernels.shape
    assert 0 < n_budget <= n_processes

    # prepare the policy spawner
    # XXX we assume the state and action spaces are known
    n_steps_per_experiment = n_episodes_per_experiment * n_steps_per_episode
    Creator = partial(Policy, n_steps_per_experiment, n_budget, n_states)

    # prepare the seed sequence for the experiment: the same seed for episodic env
    #   generator, but different for the policy's interactions within an episodes
    main = SeedSequence(entropy)
    sq_experiment = sq_spawn(
        main,
        # policy init
        (n_experiments, 1),
        # environment generator init
        (1, 1),
        # intra-episode deterministic chaos
        (n_experiments, n_episodes_per_experiment),
        axis=1,
    )

    # per experiment loop
    history = []
    for sq_pol_init, sq_env_init, sq_episodes in tqdm.tqdm(sq_experiment, ncols=60):
        # create a new policy
        pol = Creator(random=sq_pol_init[0])
        policy_gains_unfair_advantage = sneak_peek(pol)

        # reuse the same seed sequence deterministic chaos in env
        episodes = MDP.sample(
            sq_env_init[0], kernels, rewards, n_processes=n_processes
        )

        # multi-episode policy: fully reset envs between episodes, but not the policy
        #  unless it takes an unfair sneak peek through the env
        # XXX We draw a new env (at random the the pool) at the start of each episode,
        #  but the policy is not reset. why? who knows?
        episode_rewards, walltimes = [], [monotonic()]
        for env, seed_seq in zip(episodes, sq_episodes):
            policy_gains_unfair_advantage(env)

            # the trace of rewards gained during the episode
            episode_rewards.append(play(seed_seq, env, pol, n_steps_per_episode))
            walltimes.append(monotonic())

        # track whatever the episode runner yielded and per-episode time measurements
        history.append(
            (np.stack(episode_rewards), np.ediff1d(walltimes))
        )

    # unpack, stack, and then return a dictionary
    episode_rewards, walltimes = map(np.stack, zip(*history))
    return dict(
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
    )
