# UCB-LCB algorithm for RMAB

## Setup

In general, the following setup should suffice for development or reproduction

```bash
# ensure micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# setup the developer's env for alsoservice
# XXX '=' fuzzy prefix version match, '==' exact version match
# XXX micromamba deactivate && micromamba env remove -n ucblcb
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

```bash
# python main.py -N 10 -T 50 -B 3 -E 1
# python main.py -N 100 -T 500 -B 20 -E 30
ipython -i run_xp1_single.py --                \
    ucblcb.policies.RandomSubsetPolicy         \
    --params='{}'                              \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=100                               \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=500            \
    --n_steps_per_episode=20

ipython -i run_xp1_single.py --                \
    ucblcb.policies.UcbLcb                     \
    --params='{"threshold": 0.65}'             \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=100                               \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=500            \
    --n_steps_per_episode=20

ipython -i run_xp1_single.py --                \
    ucblcb.policies.Whittle                    \
    --params='{"gamma": 0.95}'                 \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=100                               \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=500            \
    --n_steps_per_episode=20

ipython -i run_xp1_single.py --                \
    ucblcb.policies.WIQL                       \
    --params='{"alpha": 0.5}'                  \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=100                               \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=500            \
    --n_steps_per_episode=20

# run xp1 on all policies and build comparative plots
ipython -i run_xp1.py --                       \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=100                               \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=100            \
    --n_steps_per_episode=100

ipython -i run_xp1.py --                       \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=50                                \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=10             \
    --n_steps_per_episode=1000

ipython -i run_xp1.py --                       \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=50                                \
    --n_budget=20                              \
    --n_experiments=31                         \
    --n_episodes_per_experiment=500            \
    --n_steps_per_episode=20

ipython -i run_xp1.py --                       \
    --entropy=B76A074C23C703767710E1D756F73AE9 \
    --path='./results'                         \
    --n_population=100                         \
    --n_arms=50                                \
    --n_budget=20                              \
    --n_experiments=501                        \
    --n_episodes_per_experiment=20             \
    --n_steps_per_episode=500
```
