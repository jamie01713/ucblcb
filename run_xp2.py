"""Run experiment ype 1 on the given policy.
"""

import os
import pickle
import warnings

from copy import deepcopy

from numpy.random import SeedSequence
from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt

from ucblcb.policies.base import BasePolicy

from ucblcb.envs.rmab import random_valid_binary_mdp, binary_rmab_from_nasx_npz
from ucblcb.experiment import experiment2 as xp2

from ucblcb.experiment.utils import from_qualname

from itertools import product
from ucblcb.experiment.utils import ewmmean, expandingmean


# populate the dictionary of algorithms and parameters for them
specs = {
    # random subset and the optimal policy
    "ucblcb.policies.base.RandomSubsetPolicy": {},
    "ucblcb.policies.whittle.Whittle": {"gamma": [0.99]},  # high discount makes VI slow
    # product of confidence interval incremental reward estimates with greedy policy
    "ucblcb.policies.lcbggt.LGGT": {
        "threshold": [0.1, 0.5],  # assumes reward in `[0, 1]`
    },
    # whittle-index q-learning
    "ucblcb.policies.wiql.WIQL": {
        "gamma": [0.99],  # discount (was set to one in the original impl)
        "alpha": [None],  # lr schedule
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
    # properties of binary MDPs' transitions and rewards
    no_good_to_act: bool = True,
    no_good_origin: bool = True,
    noise: float = 0.0,
    # policies to run the experiment on (see `specs`)
    override: dict = None,
    # the number of experiment replications
    n_experiments: int = 11,
    # the number of independent replications of a policy per experiment
    n_replications_per_experiment: int = 11,
    # the total number of steps in each replication
    n_steps_per_replication: int = 500,
    # other
    **ignore,
):
    if ignore:
        warnings.warn(repr(ignore), RuntimeWarning)

    assert prefix or not isinstance(override, dict)

    # the name of the experiment
    xp2all = "xp2all" + ("_" if prefix else "") + prefix

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
            f"N{n_arms}",
            f"{noise!s}",
            f"B{n_budget}",
            f"E{n_experiments}",
            f"L{n_replications_per_experiment}",
            f"H{n_steps_per_replication}",
            "-ga" if no_good_to_act else "+ga",
            "-go" if no_good_origin else "+go",
            data,
            f"{main.entropy:032X}",
        ]
    )

    # run the experiment is not data is available
    data_pkl = os.path.join(path, f"{xp2all}_data__{tag}.pkl")
    if not os.path.isfile(data_pkl):
        # one seed for the MDP population, another for the experiment
        sq_pop, sq_exp = main.spawn(2)

        # get the pool of Markov processes
        kernels, rewards = sample_pool(sq_pop)

        # run the implemented policies
        results = []
        for Policy in generate_policies(override):
            results.append(
                xp2.run(
                    kernels,
                    rewards,
                    Policy,
                    # each experiment uses the exact same seed sequence
                    #  instead of sequentially updated one with `.spawn`
                    deepcopy(sq_exp),
                    n_arms=n_arms,
                    n_budget=n_budget,
                    n_experiments=n_experiments,
                    n_replications_per_experiment=n_replications_per_experiment,
                    n_steps_per_replication=n_steps_per_replication,
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
        colors, legend = mpl.cm.tab10.colors, []
        for (pol, output), col in zip(results, colors):
            result = output["results"]  # (E, R, T)

            # cumulative reward over the trajectories
            acr_ert = expandingmean(result["rewards"], axis=-1)

            # average over replications
            avg_acr_et = acr_ert.mean(axis=-2)

            # plot averaged dynamics for all experiments
            line, *_ = ax.plot(avg_acr_et.T, color=col, alpha=0.5)
            legend.append((line, output["policy_name"]))

        # create the legend
        ax.legend(*zip(*legend), loc="best")

        # name the axes and the figure
        cfg = output["config"]
        ax.set_title("E={n_experiments} N={n_arms} B={n_budget}".format_map(cfg))
        ax.set_xlabel("step out of {n_steps_per_replication} total".format_map(cfg))
        ax.set_ylabel(
            f"Average cumulative reward ({n_replications_per_experiment} reps.)"
        )
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp2all}_fig1__{tag}.pdf"))

    # save the pdf for the smoothed average reward
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(7, 4))
    with mpl.rc_context({"legend.fontsize": "x-small"}):
        colors, legend = mpl.cm.tab10.colors, []
        for (pol, output), col in zip(results, colors):
            result = output["results"]  # (E, R, T)

            # cumulative reward over the trajectories
            acr_ert = ewmmean(result["rewards"], alpha=0.95, axis=-1)

            # average over replications
            avg_acr_et = acr_ert.mean(axis=-2)

            # plot averaged dynamics for all experiments
            line, *_ = ax.plot(avg_acr_et.T, color=col, alpha=0.5)
            legend.append((line, output["policy_name"]))

        # create the legend
        ax.legend(*zip(*legend), loc="best")

        # name the axes and the figure
        cfg = output["config"]
        ax.set_title("E={n_experiments} N={n_arms} B={n_budget}".format_map(cfg))
        ax.set_xlabel("$t=1..T$ T={n_steps_per_replication}".format_map(cfg))
        ax.set_ylabel(
            f"Average cumulative reward ({n_replications_per_experiment} reps.)"
        )
        ax.set_ylim(17, 30)

    fig.savefig(os.path.join(path, f"{xp2all}_fig2__{tag}.pdf"))

    return results


if __name__ == "__main__":
    # why not use pydantic and manage experiments via json?
    import argparse
    from ucblcb.config import make

    # we allow some limited config through the command line args
    parser = make(
        argparse.ArgumentParser(
            description="Run experiment two over all policies.", add_help=True
        ),
        "path",
        "prefix",
        # state, action, and arm space sizes
        "n_arms",
        # the budget of arms
        "n_budget",
        # exepriment parameters and replications
        "n_experiments",
        "n_replications_per_experiment",
        "n_steps_per_replication",
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
