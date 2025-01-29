import tqdm
from time import monotonic, strftime

import numpy as np
from numpy import ndarray

from functools import partial

from typing import Iterator, Iterable
from numpy.random import SeedSequence, Generator

from collections.abc import Callable

from ..envs import MDP
from ..policies.base import BasePolicy
from ..envs.base import Env
from .play import rollout, collate
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
    envs: list[Env] = []
    n_replications: int = ...  # number of independent reruns for the same env
    policy_factory: type = ...  # policy factory

    # experiment loop (with pre-seeded envs in episodes)
    for env in envs:
        for _ in range(n_replications):
            h, x, r, a, f, t = [...], [...], [...], [...], [...], 0

            # let the policy rummage through the env for an oracle and init the policy's
            #  "externalized" state
            pol = policy_factory()
            h[0] = pol.init_with_peek(env)

            # per-episode loop
            x[0], done = env.reset(), False  # env "externalized" state
            # XXX `r[0]` is undefined!
            while not done:
                # pol: (h_k, x_t) -->> a_t
                # XXX random decision if `pol` has NEVER been updated!
                a[t] = pol.decide(h[t], x[t])

                # env: (x_t, a_t) -->> (x_{t+1}, r_{t+1}, f_{t+1})
                x[t + 1], r[t + 1], f[t + 1] = env.step(x[t], a[t])

                # report (x[t], a[t], r[t + 1], x[t + 1], f[t + 1]) to someone
                pass

                # pol: (h_t, x_t, a_t, r_{t+1}, x_{t+1}, f_{t+1}) -->> h_{t+1}
                h[t + 1] = pol.update(h[t], x[t], a[t], r[t + 1], x[t + 1], f[t + 1])

                # step counter
                t += 1


def run_one_replication(
    env: Env,
    PolicyFactory: Callable[[Generator], BasePolicy],
    /,
    sq: SeedSequence,
    *,
    n_steps: int,
) -> tuple[BasePolicy, dict[str, ndarray]]:
    # one seed consumed by init, another by randomness used during rollout
    seed_init, seed_play = sq.spawn(2)

    # create a new policy from the provided seed
    pol = PolicyFactory(random=seed_init)

    # use a shortcut to access the ground truth about the env
    gain_gt_advantage = sneak_peek(pol)
    gain_gt_advantage(env)  # XXX `pol.reset(env)`?

    # setup a rollout runner with policy updated online in-place after every step
    runner = partial(
        # interact exactly for `n_steps`, resetting env on termination
        rollout,
        env,
        pol,
        seed_play,
        n_steps=n_steps,
        auto=True,
        n_steps_per_update=1,
    )

    # policy rollout: the same policy `pol` (i.e. init and ground truth advantage)
    #  runs and updates online for `n_episodes_per_experiment * n_steps_per_episode`
    #  steps through auto-resetting `env`.
    rewards, walltimes = [], [monotonic()]

    # rollout for the number of steps, keeping track of the reward over all arms
    # XXX `rew_` is `(n_steps_per_update, n_arms)`
    for _, _, rew_, _, _ in runner():
        rewards.append(np.sum(rew_, axis=-1))
        walltimes.append(monotonic())

    output = {
        # `rewards[j]` the reward for (j+1)-th interaction in the rollout
        "rewards": np.concatenate(rewards),
        # `walltimes[j]` the wall-time in seconds taken by the (j+1)-th interaction
        #  and policy update
        "walltimes": np.ediff1d(walltimes),
    }
    return pol, output


def run_one_experiment(
    env: Env,
    PolicyFactory: Callable[[Generator], BasePolicy],
    /,
    sq: SeedSequence,
    *,
    n_replications: int,
    n_steps_per_replication: int,
) -> tuple[BasePolicy, dict]:
    """replay this env instance with the independent instances of the policy"""

    # loop over seeds for the policy in this experiment (init and behaviour)
    replications = []
    for seed in sq.spawn(n_replications):
        pol, output = run_one_replication(
            env,
            PolicyFactory,
            seed,
            n_steps=n_steps_per_replication,
        )
        replications.append(output)

    # collate the replications
    return pol, collate(replications)


def run_all_experiments(
    EnvSampler: Callable[[Iterable[SeedSequence]], Iterator[Env]],
    PolicyFactory: Callable[[Generator], BasePolicy],
    /,
    sq: SeedSequence,
    *,
    n_experiments: int,
    n_replications_per_experiment: int,
    n_steps_per_replication: int,
) -> tuple[BasePolicy, dict]:

    # prepare the seed sequences for the experiment: one sequence for the env sampler
    #  (env init and dynamics)m and another one for the policy (init and behaviour)
    sqs_env, sqs_pol = sq_spawn(sq, (2, n_experiments))

    # per experiment loop
    experiments = []
    for env, sq_pol in zip(EnvSampler(sqs_env), tqdm.tqdm(sqs_pol, ncols=60)):
        # print(sq_pol, default_rng(sq_pol).uniform())  # XXX is chaos deterministic?

        pol, output = run_one_experiment(
            env,
            PolicyFactory,
            sq_pol,
            n_replications=n_replications_per_experiment,
            n_steps_per_replication=n_steps_per_replication,
        )

        # collate the replications
        experiments.append(output)

    # collate the experiments
    return pol, collate(experiments)


def run(
    kernels: ndarray,
    rewards: ndarray,
    Policy: Callable[[...], BasePolicy],
    /,
    entropy: int | None,
    *,
    n_arms: int,
    n_budget: int,
    n_experiments: int,
    n_replications_per_experiment: int,
    n_steps_per_replication: int,
    noise: float,
) -> tuple[BasePolicy, dict]:
    n_population, n_actions, n_states, _ = kernels.shape
    assert 0 < n_budget <= n_arms

    # prepare the seed sequence for the experiment: the same seed for episodic env
    #   generator, but different for the policy's interactions within an episodes
    main = entropy if isinstance(entropy, SeedSequence) else SeedSequence(entropy)

    # get a deterministically chaotic generator from a pool of the potential MDPs
    EnvGenerator = partial(
        MDP.sampler, kernels, rewards, n_processes=n_arms, noise=noise
    )

    # prepare the policy spawner (we assume the state and action spaces are known)
    PolicyFactory = partial(Policy, n_steps_per_replication, n_budget, n_states)

    # per experiment loop
    pol, output = run_all_experiments(
        EnvGenerator,
        PolicyFactory,
        entropy,
        n_experiments=n_experiments,
        n_replications_per_experiment=n_replications_per_experiment,
        n_steps_per_replication=n_steps_per_replication,
    )

    return pol, dict(
        # the timestamp and git
        __dttm__=strftime("%d%m%Y%H%M%S"),
        __git__=snapshot_git(),
        # entropy used by to seed this experiment
        entropy=main.entropy,
        # meta information
        config=dict(
            # processes
            n_arms=n_arms,
            n_states=n_states,
            n_actions=n_actions,
            noise=noise,
            # policy
            n_budget=n_budget,
            # experiments
            n_experiments=n_experiments,
            n_replications_per_experiment=n_replications_per_experiment,
            n_steps_per_replication=n_steps_per_replication,
        ),
        # save the name of the policy played during the last replication
        policy_name=repr(pol),
        # results
        results=output,
    )
