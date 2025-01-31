prefix=wiql-lggt
# source=./good_instance.npz

n_experiments=50
n_population=5000
n_arms=50
n_replications_per_experiment=13
n_steps_per_replication=1000
target=./results_xp2_wiql_lggt

# grid (whitespace separated strings, or '' for empty)
n_budgets="5 10 20"
noises="0.0"

# the master entropy of the entire suite of experiments (leave empty for system)
# entropy=
entropy=B76A074C23C703767710E1D756F73AE9

# in case we wan to run a subset of experiments (setting overides may overwrite
#  cached results in existing $target folders; set to 'null' for defaults)
# spec_override='null'
spec_override='{
    "ucblcb.policies.base.RandomSubsetPolicy": {},
    "ucblcb.policies.lcbggt.LGGT": {"threshold": [0.1, 0.4]},
    "ucblcb.policies.wiql.WIQL": {"gamma": [0.9], "alpha": [null]},
    "ucblcb.policies.whittle.Whittle": {"gamma": [0.9]},
    "ucblcb.policies.ucw.UCWhittleExtreme": {"gamma": [0.9], "C": [0.5]},
    "ucblcb.policies.ucw.UCWhittleUCBPriv": {"gamma": [0.9], "C": [0.5]},
    "ucblcb.policies.ucw.UCWhittlePvPriv": {"gamma": [0.9], "C": [0.5]}
}'

# budgets
for n_budget in $n_budgets; do
    for noise in $noises; do
        python run_xp2.py                                                    \
            --entropy=$entropy                                               \
            --source=$source                                                 \
            --path=$target                                                   \
            --prefix=$prefix                                                 \
            --n_population=$n_population                                     \
            --no_good_origin                                                 \
            --avg_over_experiments                                           \
            --n_arms=$n_arms                                                 \
            --n_budget=$n_budget                                             \
            --n_experiments=$n_experiments                                   \
            --n_replications_per_experiment=$n_replications_per_experiment   \
            --n_steps_per_replication=$n_steps_per_replication               \
            --noise=$noise                                                   \
            --override="${spec_override}"
    done
done
