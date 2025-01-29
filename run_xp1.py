"""Run experiment ype 1 on the given policy.
"""

import os
import pickle
import json
import warnings

from copy import deepcopy

from numpy.random import SeedSequence
from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt

from ucblcb.policies.base import BasePolicy

from ucblcb.envs.rmab import random_valid_binary_mdp, binary_rmab_from_nasx_npz
from ucblcb.experiment import experiment1 as xp1

from ucblcb.experiment.utils import from_qualname
from ucblcb.experiment.plotting import (
    plot_average_cumulative_reward,
    plot_average_reward,
)

from itertools import product


# populate the dictionary of algorithms and parameters for them
specs = {
    "ucblcb.policies.base.RandomSubsetPolicy": {},
    # product of confidence interval incremental reward estimates with greedy policy
    "ucblcb.policies.lcbggt.LGGT": {
        "threshold": [0.1, 0.5],  # assumes reward in `[0, 1]`
    },
    # whittle-index q-learning
    "ucblcb.policies.wiql.WIQL": {
        "gamma": [0.99],  # discount (was set to one in the original impl)
        "alpha": [None],  # lr schedule
    },
    # optimal policy
    "ucblcb.policies.whittle.Whittle": {
        "gamma": [0.99],  # discount (the closer to one, the slower the VI!)
    },
    # # UCWhittle-Extreme
    # "ucblcb.policies.ucw.UCWhittleExtreme": {
    #     "gamma": [0.99],  # discount (the closer to one, the slower the VI!)
    # },
    # # UCWhittle-UCB
    # "ucblcb.policies.ucw.UCWhittleUCB": {
    #     "gamma": [0.99],  # discount (the closer to one, the slower the VI!)
    # },
}


def generate_policies(specs: dict):
    """Generate the policies to try out."""

    # run the selected algorithms
    for qn, grid in specs.items():
        cls = from_qualname(qn)
        assert issubclass(cls, BasePolicy)
        assert all(isinstance(x, (list, tuple)) for x in grid.values())

        # enumerate all hparam in the priduct
        for values in product(*grid.values()):
            # return a policy builder
            yield partial(cls, **dict(zip(grid, values)))


def main(
    path: str,
    prefix: str = "",
    *,
    # the entropy for seeding the environments
    entropy: int = 243799254704924441050048792905230269161,  # a value from numpy docs
    source: str = "",
    # the total size of the mdp pool
    n_population: int = 1000,
    # the number of MDPs to be managed by the ramb
    n_arms: int = 25,
    # the allotted budget
    n_budget: int = 7,
    # the number of experiment replications
    n_experiments: int = 100,
    # the number of episodes each policy instance plays
    n_episodes_per_experiment: int = 33,
    # the number of instraction steps in each episode
    n_steps_per_episode: int = 500,
    # properties of binary MDPs' transitions and rewards
    no_good_to_act: bool = True,
    no_good_origin: bool = True,
    noise: float = 0.0,
    # policies to run the experiment on (see `specs`)
    override: dict = None,
    **ignore,
):
    if ignore:
        warnings.warn(repr(ignore), RuntimeWarning)

    assert prefix or not isinstance(override, dict)

    # the name of the experiment
    xp1all = "xp1all" + ("_" if prefix else "") + prefix

    # handle policy spec overrides
    if override is not None and not isinstance(override, dict):
        warnings.warn(repr(override), RuntimeWarning)
    override = override if isinstance(override, dict) else specs

    # create the main seed sequence
    main = SeedSequence(entropy)

    # ensure a path for the results
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # decide where to sample MDPs from
    if source:
        if not os.path.isfile(source):
            raise RuntimeError(source)

        sample_pool = partial(binary_rmab_from_nasx_npz, source)
        data = "npz"

    else:
        sample_pool = partial(
            random_valid_binary_mdp,
            size=(n_population,),
            good_to_act=not no_good_to_act,
            good_origin=not no_good_origin,
        )
        data = "random"

    # construct the filename tag
    tag = "__".join(
        [
            f"P{n_population}",
            f"n{n_arms}",
            f"b{n_budget}",
            f"H{n_steps_per_episode}",
            f"L{n_episodes_per_experiment}",
            f"E{n_experiments}",
            f"{noise!s}",
            "-ga" if no_good_to_act else "+ga",
            "-go" if no_good_origin else "+go",
            data,
            f"{main.entropy:032X}",
        ]
    )

    # run the experiment is not data is available
    data_pkl = os.path.join(path, f"{xp1all}_data__{tag}.pkl")
    if not os.path.isfile(data_pkl):
        # one seed for the MDP population, another for the experiment
        sq_pop, sq_exp = main.spawn(2)

        # get the pool of Markov processes
        kernels, rewards = sample_pool(sq_pop)

        # run the implemented policies
        results = []
        for Policy in generate_policies(override):
            results.append(
                xp1.run(
                    # each experiment uses the exact same seed sequence
                    #  instead of sequentially updated one with `.spawn`
                    deepcopy(sq_exp),
                    Policy,
                    kernels,
                    rewards,
                    n_processes=n_arms,
                    n_budget=n_budget,
                    n_experiments=n_experiments,
                    n_episodes_per_experiment=n_episodes_per_experiment,
                    n_steps_per_episode=n_steps_per_episode,
                    noise=noise,
                )
            )

        # save the experiment data
        with open(data_pkl, "wb") as pkl:
            pickle.dump(results, pkl)

    # load the experiment data
    with open(data_pkl, "rb") as pkl:
        results = pickle.load(pkl)

    # save the pdf for the average cumulative reward
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(7, 4))
    with mpl.rc_context({"legend.fontsize": "x-small"}):
        plot_average_cumulative_reward(results, ax=ax, C=1.0)
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp1all}_fig1__{tag}.pdf"))

    # save the pdf for the smoothed average reward
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(7, 4))
    with mpl.rc_context({"legend.fontsize": "x-small"}):
        plot_average_reward(results, ax=ax, C=1.0)
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp1all}_fig2__{tag}.pdf"))

    return results


if __name__ == "__main__":
    # why not use pydantic and manage experiments via json?
    import argparse
    from ucblcb.config import make

    # we allow some limited config through the command line args
    parser = make(
        argparse.ArgumentParser(
            description="Run experiment one over all policies.", add_help=True
        ),
        "path",
        "prefix",
        # state, action, and arm space sizes
        "n_arms",
        # the budget of arms
        "n_budget",
        # exepriment parameters and replications
        "n_steps_per_episode",
        "n_episodes_per_experiment",
        "n_experiments",
        # diversity and properties of the MDPs
        "n_population",
        "no_good_to_act",
        "no_good_origin",
        "noise",
        # the source of the mdp pool
        "source",
        # override policy specs
        "override",
        # seed
        "entropy",
    )

    # get the namespace with declared cli args, and a list of remaining argument strings
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_known_args
    args, _ = parser.parse_known_args()
    print(repr(args))

    results = main(**vars(args))  # noqa: F401
