prefix=lggt-wiql
# source=./good_instance.npz

n_experiments=101
n_population=50
n_arms=50
n_episodes_per_experiment=1
n_steps_per_episode=500
target=./results
noise=0.0

# grid (whitespace separated strings, or '' for empty)
n_budgets="5 10 15 20"
b_good_origin_flags="--no-good-origin ''"

# the master entropy of the entire suite of experiments (leave empty for system)
# entropy=
entropy=B76A074C23C703767710E1D756F73AE9

# in case we wan to run a subset of experiments (setting overides may overwrite
#  cached results in existing $target folders; set to 'null' for defaults)
# spec_override='null'
spec_override='{
    "ucblcb.policies.lcbggt.LGGT": {"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "ucblcb.policies.wiql.WIQL": {"gamma": [0.99], "alpha": [null, 0.5]}
}'

# budgets
for n_budget in $n_budgets; do
    for b_good_origin in $b_good_origin_flags; do
        python run_xp1.py                                            \
            --entropy=$entropy                                       \
            --source=$source                                         \
            --path=$target                                           \
            --prefix=$prefix                                         \
            --n_population=$n_population                             \
            $b_good_origin                                           \
            --n_arms=$n_arms                                         \
            --n_budget=$n_budget                                     \
            --n_experiments=$n_experiments                           \
            --n_episodes_per_experiment=$n_episodes_per_experiment   \
            --n_steps_per_episode=$n_steps_per_episode               \
            --noise=$noise                                           \
            --override="${spec_override}"
    done
done
