"""
python run_xp1.py ucblcb.policies.UcbLcb         \
    --params='{"threshold": 0.65}'               \
    --entropy=B76A074C23C703767710E1D756F73AE9   \
    --path='./results'
"""

import os
import pickle

import json
import numpy as np

from scipy.special import softmax
from numpy.random import default_rng
from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt

from ucblcb.experiment import experiment1 as xp1
from ucblcb.experiment.utils import from_qualname


def main(
    path: str,
    *,
    # the entropy for seeding the environments
    entropy: int = 243799254704924441050048792905230269161,  # a value from numpy docs
    # policy params
    cls: str = "ucblcb.policies.ucblcb.UcbLcb",
    params: str = '{"threshold": 0.65}',
    # mdp specs
    n_states: int = 5,
    n_actions: int = 3,
    temperature: float = 0.5,
    # the total size of the mdp pool
    n_population: int = 1000,
    # the number of MDPs to be managed by the ramb
    n_processes: int = 25,
    # the allotted budget
    n_budget: int = 7,
    # the number of experiment replications
    n_experiments: int = 100,
    # the number of episodes each policy instance plays
    n_episodes_per_experiment: int = 33,
    # the number of instraction steps in each episode
    n_steps_per_episode: int = 500,
):
    # ensure a path for the results
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # Reinforcement learning augmented asymptotically optimal index policy for
    #  finite-horizon restless bandits
    random_ = default_rng(None)

    # get the pool of markov processes
    kernels = softmax(
        random_.normal(
            size=(
                n_population,
                n_states,
                n_actions,
                n_states,
            )
        )
        / temperature,
        axis=-1,
    )
    rewards = random_.normal(size=(1, 1, 1, kernels.shape[3]))

    # assemble a policy builder
    Policy = partial(from_qualname(cls), **params)

    # run the experiment
    results = xp1.run(
        entropy,
        Policy,
        kernels,
        rewards,
        n_processes=n_processes,
        n_budget=n_budget,
        n_experiments=n_experiments,
        n_episodes_per_experiment=n_episodes_per_experiment,
        n_steps_per_episode=n_steps_per_episode,
    )

    # save the experiment data
    filename_tag = xp1.make_name(**results)
    with open(os.path.join(path, f"data__{filename_tag}.pkl"), "wb") as pkl:
        pickle.dump(results, pkl)

    # fetch the result
    traces = results["episode_rewards"]
    full_rewards = np.reshape(traces, (len(traces), -1))
    averages = full_rewards.cumsum(-1) / (1 + np.arange(full_rewards.shape[1]))

    # measure the cumulative reward due to policy randomness (all envs are the same)
    m, s = averages.mean(0), averages.std(0)

    # make pretty picture
    fig, ax = plt.subplots(1, 1, dpi=120, figsize=(5, 3))
    ax.set_title("Average cumulative multi-episodic reward")
    with mpl.rc_context(
        {
            "legend.fontsize": "x-small",
        }
    ):
        xs = 1 + np.arange(len(m))
        (line,) = ax.plot(xs, m, label=results["policy_name"], color="C0")
        ax.fill_between(
            xs,
            m - 1.96 * s,
            m + 1.96 * s,
            zorder=-10,
            alpha=0.15,
            color=line.get_color(),
        )
        ax.legend(loc="lower right")

    # save the pdf
    fig.savefig(os.path.join(path, f"fig1__{filename_tag}.pdf"))


if __name__ == "__main__":
    import argparse

    # we allow some limited config through the command line args
    parser = argparse.ArgumentParser(description="Run experiment one.", add_help=True)
    parser.register("type", "hexadecimal", lambda x: int(x, 16) if x else None)
    parser.register("type", "json", json.loads)

    # the policy and its hyperparameters
    parser.add_argument("cls", type=str, default="ucblcb.policies.ucblcb.UcbLcb")
    parser.add_argument(
        "--params", required=False, type="json", default='{"threshold": 0.2}'
    )
    parser.add_argument("--entropy", required=False, type="hexadecimal", default=None)

    parser.add_argument("--n_population", type=int, default=1000)
    parser.add_argument("--n_states", type=int, default=5)
    parser.add_argument("--n_actions", type=int, default=3)
    parser.add_argument("--n_processes", type=int, default=25)
    parser.add_argument("--n_budget", type=int, default=7)
    parser.add_argument("--n_experiments", type=int, default=100)
    parser.add_argument("--n_episodes_per_experiment", type=int, default=33)
    parser.add_argument("--n_steps_per_episode", type=int, default=500)

    parser.add_argument("--path", type=str, default="./")

    # get the namespace with declared cli args, and a list of remaining argument strings
    # https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_known_args
    args, _ = parser.parse_known_args()

    main(temperature=0.5, **vars(args))
