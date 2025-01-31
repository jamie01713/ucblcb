# Lcb Guided Greedy Thresholding algorithm for binary RMAB

## Setup

In general, the following setup should suffice for development or reproduction.

Unless you are or prefer using `conda`, or already have `micromamba`, please, follow the installation instructions of [mamba-org](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). That link provides you with many installation options, one of them being the following:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

To setup the required environment for the experiments, please, run the following block. If you are using or prefer to use `conda`, please, replace `micromamba` with `conda`.

```bash
# setup the developer's env for alsoservice
micromamba create -n ucblcb                    \
  "python>=3.11"                               \
  numpy                                        \
  scipy                                        \
  jax                                          \
  chex                                         \
  scikit-learn                                 \
  pandas                                       \
  gitpython                                    \
  matplotlib                                   \
  gymnasium                                    \
  "gurobi::gurobi=12"                          \
  jupyter                                      \
  tqdm                                         \
  "black[jupyter]"                             \
  nbdime                                       \
  && micromamba clean --all --yes
```

## Running experiments

The following command runs the experiment declared in `run_xp2_wiql_lggt.sh`, which runs a sweep over several settings of the budget, that the policy is allowed to interact with on every step. Replace `micromamba` with `conda` if required.

```bash
# from the root of the repo (replace micromamba with conda if required)
micromamba run -n ucblcb sh ./run_xp2_wiql_lggt.sh
```

This will create a folder `results_xp2_wiql_lggt` in the root of the repo, wherein, after a while, you will be able to find the PDFs with plots and python Pickles, containing the results of the experiment runs. The results contain information about the configuration of the experiment from the sweep, and the results collected during independent replications of the rollouts of the policies specified in the `.sh` script.

The name of the pickle file `.pkl` is in the following format:

```text
f"""
    {prefix}
    __P{n_population}
    __M{n_arms}
    __{noise}
    __B{n_budget}
    __E{n_experiments}
    __L{n_replications_per_experiment}
    __H{n_steps_per_replication}
    __{does_pulling_have_higher_probability_of_good_state}
    __{is_being_in_good_state_more_likely_to_lead_to_a_good_state}
    __{data_source}
    __{128 bit entropy}
"""
```

After the pickles with the results have been generated, you may re-create a plot similar to the tri-axis figure from the paper by running the following command, where the last three `.pkl` filenames should be replaced with the ones that you want.

```bash
micromamba run -n ucblcb \
    python make_triplet_figure.py --smoother=ewm P1000_M50_0.0_E11_R13_T500.pdf                                          \
    ./xp2all_all_run2_data__P1000__M50__0.0__B5__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl  \
    ./xp2all_all_run2_data__P1000__M50__0.0__B10__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl \
    ./xp2all_all_run2_data__P1000__M50__0.0__B20__E11__L13__H500__+ga__-go__random__B76A074C23C703767710E1D756F73AE9.pkl
```
