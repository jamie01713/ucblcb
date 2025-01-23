# import pickle
import json
import numpy as np

from scipy.special import softmax
from numpy.random import default_rng
from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt

from ucblcb.experiment import experiment1 as xp1
from ucblcb.experiment.utils import from_qualname


entropy: int | None = 243799254704924441050048792905230269161
config = dict(
    n_population=1000,
    n_states=5,
    n_actions=3,
    temperature=0.5,
    n_processes=25,
    n_budget=7,
    n_experiments=100,
    n_episodes_per_experiment=33,
    n_steps_per_episode=500,
    # policy params
    cls="ucblcb.policies.ucblcb.UcbLcb",
    params='{"threshold": 0.65}',
)

# Reinforcement learning augmented asymptotically optimal index policy for
#  finite-horizon restless bandits
random_ = default_rng(None)

# get the pool of markov processes
kernels = softmax(
    random_.normal(
        size=(
            config["n_population"],
            config["n_states"],
            config["n_actions"],
            config["n_states"],
        )
    )
    / config["temperature"],
    axis=-1,
)
rewards = random_.normal(size=(1, 1, 1, kernels.shape[3]))

# assemble a policy builder
Policy = partial(from_qualname(config["cls"]), **json.loads(config["params"]))

# run the experiment
results = xp1.run(
    entropy,
    Policy,
    kernels,
    rewards,
    n_processes=config["n_processes"],
    n_budget=config["n_budget"],
    n_experiments=config["n_experiments"],
    n_episodes_per_experiment=config["n_episodes_per_experiment"],
    n_steps_per_episode=config["n_steps_per_episode"],
)

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
tag = xp1.make_name(**results)
fig.savefig(f"fig1__{tag}.pdf")
