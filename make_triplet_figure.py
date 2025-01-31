import os
import pickle

from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt

from collections import defaultdict
from ucblcb.experiment.utils import ewmmean, expandingmean
from ucblcb import policies as lib


path = "./results_xp2_all-no-ucw/"
path = os.path.abspath(path)
triplet = [
    'xp2all_all_data__P50__M50__0.0__B5__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl',
    'xp2all_all_data__P50__M50__0.0__B10__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl',
    'xp2all_all_data__P50__M50__0.0__B20__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl',
]

filenames = [os.path.join(path, x) for x in triplet]


# do we add gray trajectories to show variance
add_gray_trajectories = True
smoother = "ewm"
# title = "E={n_experiments} M={n_arms} B={n_budget}"
title = "Budget {n_budget} / {n_arms} (E{n_experiments}$\\times$R{n_replications_per_experiment})"


# renamers to make the legend more compact
# XXX ignore x.alpha and x.gamma setting
labeller = {
    lib.RandomSubsetPolicy: lambda x: "Random",
    lib.Whittle: lambda x: "Whittle",
    lib.LGGT: lambda x: f"LGGT({x.threshold:2.1f})",
    lib.WIQL: lambda x: "WIQL",
    lib.UCWhittleExtreme: lambda x: "ucw-extreme",
    lib.UCWhittleUCB: lambda x: "ucw-ucb",
    # lib.UCWhittlePv: lambda x: repr(x),
    # lib.UCWhittlePvPriv: lambda x: repr(x),
    # lib.UCWhittleUCBPriv: lambda x: repr(x),
}

# z-order settings for the policies so that the really important curves are not occluded
zorders = {
    lib.RandomSubsetPolicy: -10,
    lib.Whittle: -10,
    lib.LGGT: 10,
    lib.WIQL: 5,
}

# how to denoise the time series
smoother_options = {
    "cumulative": (expandingmean, "Average cumulative reward"),
    "ewm": (partial(ewmmean, alpha=0.95), "Smoothed reward"),
}

# create three axes
fig, axes = plt.subplots(
    1, min(len(filenames), 3), dpi=120, figsize=(9, 3), sharey=False,  # sharex=True
)
has_axes_labels = False
with mpl.rc_context({"legend.fontsize": "x-small"}):
    # prepare a global color lookup table: Tableau10
    it = iter(mpl.cm.tab10.colors)
    colors, legend = defaultdict(partial(next, it)), {}

    # read result dump of each run
    smoother_fn, ylabel = smoother_options[smoother]
    for ax, run in zip(axes, filenames):
        # for each policy in the run
        for pol, output in pickle.load(open(run, "rb")):
            result = output["results"]  # (E, R, T)
            label = labeller[type(pol)](pol)
            zorder = zorders.get(type(pol), 0)

            # smooth the rewards long each trajectories
            rew_ert = smoother_fn(result["rewards"], axis=-1)

            # average trajectories over all experiments and replications
            line, *_ = ax.plot(
                rew_ert.mean(axis=(-3, -2)),
                color=colors[label],
                alpha=1.0,
                zorder=zorder,
            )

            # add trajectories averaged only across replications
            if add_gray_trajectories:
                ax.plot(
                    rew_ert.mean(axis=-2).T,
                    lw=1,
                    color="k",  # line.get_color(),
                    alpha=0.15,
                    zorder=zorder - 20,
                )

            # collect artists for the legend
            legend[label] = line

        # name the axes and the figure
        cfg = output["config"]
        ax.set_title(title.format_map(cfg))
        if not has_axes_labels:
            has_axes_labels = True
            ax.set_ylabel(ylabel)

    # create the legend
    labels, handles = zip(*legend.items())
    fig.legend(handles, labels, loc="upper center", ncol=len(legend))

    # aesthetics
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

plt.show()
