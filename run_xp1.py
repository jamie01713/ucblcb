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
    "ucblcb.policies.ucblcb.UcbLcb": {
        "threshold": [0.1, 0.5, 0.9],  # assumes reward in `[0, 1]`
    },
    # whittle-index q-learning
    "ucblcb.policies.wiql.WIQL": {
        "gamma": [0.99],  # discount (was set to one in the original impl)
        "alpha": [0.5],  # lr schedule
    },
    # optimal policy
    "ucblcb.policies.whittle.Whittle": {
        "gamma": [0.99],  # discount (the closer to one, the slower the VI!)
    },
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
    # properties of binary MDPs' transitions
    no_good_to_act: bool = True,
    no_good_origin: bool = True,
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
        plot_average_cumulative_reward(results, ax=ax)
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp1all}_fig1__{tag}.pdf"))

    # save the pdf for the smoothed average reward
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(7, 4))
    with mpl.rc_context({"legend.fontsize": "x-small"}):
        plot_average_reward(results, ax=ax)
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp1all}_fig2__{tag}.pdf"))

    return results


if __name__ == "__main__":
    # why not use pydantic and manage experiments via json?
    import argparse

    # we allow some limited config through the command line args
    parser = argparse.ArgumentParser(
        description="Run experiment one over all policies.", add_help=True
    )
    parser.register("type", "hexadecimal", lambda x: int(x, 16) if x else None)
    parser.register("type", "json", json.loads)

    # the policy and its hyperparameters
    parser.add_argument(
        "--path",
        type=str,
        default="./",
        help="The folder to store the resulting pickle and the figures",
    )
    parser.add_argument(
        "--prefix",
        required=False,
        type=str,
        default="",
        help="extra prefix to add to the filenames",
    )

    # state, action, and arm space sizes
    parser.add_argument(
        "--n_arms",
        "-N",
        type=int,
        default=100,
        help="The number or arms in the binary MDP environment",
    )

    # the budget of arms
    parser.add_argument(
        "--n_budget",
        "-B",
        type=int,
        default=20,
        help="The number of arms the policy is allowed to pull at each step",
    )

    # exepriment parameters and replications
    parser.add_argument(
        "--n_steps_per_episode",
        "-H",
        type=int,
        default=20,
        help="The maximal number of steps in one episodes rollout",
    )
    parser.add_argument(
        "--n_episodes_per_experiment",
        "-T",
        type=int,
        default=500,
        help="The number of episodes to play in one experiment replication",
    )
    parser.add_argument(
        "--n_experiments",
        "-E",
        type=int,
        default=30,
        help="The total number of independent replications to run",
    )

    # diversity and properties of the MDPs
    parser.add_argument(
        "--n_population",
        "-P",
        type=int,
        default=100,
        help="The size of pool of MDP arms from which environment are sampled",
    )
    parser.add_argument(
        "--no-good-to-act",
        required=False,
        action="store_true",
        help="enforce good-to-act peroperty of MDPs",
    )
    parser.add_argument(
        "--no-good-origin",
        required=False,
        action="store_true",
        help="enforce good-origin peroperty of MDPs",
    )

    # the source of the mdp pool
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="An npz-file to load transitions and rewards from "
             "(leave empty for random pool).",
    )

    # override policy specs
    parser.add_argument("--override", required=False, type="json", default="null")

    # seed
    parser.add_argument(
        "--entropy",
        required=False,
        type="hexadecimal",
        # https://www.random.org/cgi-bin/randbyte?nbytes=16&format=h
        default="75CEF71D882634C09033BC8108F9E0C0",
        help="128-bit seed for the experiment (leave empty to use system entropy)",
    )

    # get the namespace with declared cli args, and a list of remaining argument strings
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_known_args
    args, _ = parser.parse_known_args()
    print(repr(args))

    results = main(**vars(args))  # noqa: F401
