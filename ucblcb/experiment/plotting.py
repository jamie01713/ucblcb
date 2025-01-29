import numpy as np
from matplotlib import pyplot as plt

from .utils import ewmmean, expandingmean


def plot_one_average_cumulative_reward(
    res, color=None, /, C=1.96, *, ax=None
) -> plt.Axes:
    """Plot the average cumulative reward across multiple episodes."""

    ax = plt.gca() if ax is None else ax

    # get the rewards and collapse episode and step dims
    rewards = res["episode_rewards"]
    n_experiments, _, _ = rewards.shape
    x = np.reshape(rewards, (n_experiments, -1))

    # the average cumulative reward and its std-dev
    xt = 1 + np.arange(x.shape[-1])
    xm = expandingmean(x, axis=-1).mean(0)
    xs = np.sqrt(expandingmean(x * x, axis=-1).mean(0) - xm * xm)

    # plot the average and the 95% normal CI
    (line,) = ax.plot(xt, xm, label=res["policy_name"], color=color)
    if C > 0:
        ax.fill_between(
            xt,
            xm - C * xs,
            xm + C * xs,
            color=line.get_color(),
            zorder=-10,
            alpha=0.075,
        )

    return ax


def plot_average_cumulative_reward(results, *, C: float = 0.0, ax=None) -> plt.Axes:
    """Plot the `average cumulative multi-episodic reward` for a list of results."""
    ax = plt.gca() if ax is None else ax

    for pol, res in results:
        plot_one_average_cumulative_reward(res, C=C, ax=ax)

    # add all the aesthetics
    ax.set_title("N={n_processes} B={n_budget} E={n_experiments}".format_map(res))
    ax.set_xlabel(
        "step $t$ ({n_episodes_per_experiment} episodes "
        "x {n_steps_per_episode} steps)".format_map(res)
    )
    ax.set_ylabel("Average cumulative reward")
    ax.legend(loc="lower right")

    return ax


def plot_one_average_reward(
    res, color=None, /, alpha=0.99, C=1.96, *, ax=None
) -> plt.Axes:
    """Plot the average reward across multiple episodes."""

    ax = plt.gca() if ax is None else ax

    # get the rewards and collapse episode and step dims
    rewards = res["episode_rewards"]
    n_experiments, _, _ = rewards.shape
    x = np.reshape(rewards, (n_experiments, -1))

    # the smoothed average reward and its std-dev
    xt = 1 + np.arange(x.shape[-1])
    xm = ewmmean(x.mean(0), alpha=alpha, axis=-1)
    xs = ewmmean(x.std(0), alpha=alpha, axis=-1)

    # plot the average and the 95% normal CI
    (line,) = ax.plot(xt, xm, label=res["policy_name"], color=color)
    if C > 0:
        ax.fill_between(
            xt,
            xm - C * xs,
            xm + C * xs,
            color=line.get_color(),
            zorder=-10,
            alpha=0.075,
        )

    return ax


def plot_average_reward(
    results, *, alpha: float = 0.7, C: float = 0.0, ax=None
) -> plt.Axes:
    """Plot the `average smoothed multi-episodic reward` for a list of results."""
    # higher alpha means less smoothing
    assert 0 <= alpha < 1, alpha

    ax = plt.gca() if ax is None else ax
    for pol, res in results:
        plot_one_average_reward(res, alpha=alpha, C=C, ax=ax)

    # add all the aesthetics
    ax.set_title("N={n_processes} B={n_budget} E={n_experiments}".format_map(res))
    ax.set_xlabel(
        "step $t$ ({n_episodes_per_experiment} episodes "
        "x {n_steps_per_episode} steps)".format_map(res)
    )
    ax.set_ylabel(rf"Average reward (ewm $\alpha={alpha}$)")
    ax.legend(loc="lower right")

    return ax
