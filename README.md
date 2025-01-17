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
  "numpy>=1.26"                                \
  "scipy>=1.13"                                \
  "jax>=0.4.35"                                \
  "scikit-learn>=1.5"                          \
  "pandas>=2"                                  \
  matplotlib                                   \
  "gymnasium"                                  \
  "gurobi::gurobi"                             \
  jupyter                                      \
  tqdm                                         \
  "black[jupyter]"                             \
  nbdime                                       \
  && micromamba clean --all --yes
```

## Running experiments

```bash
python main.py -N 100 -T 500 -B 20 -E 30
```
