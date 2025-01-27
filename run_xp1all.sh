n_experiments=101
n_population=50
n_arms=50
n_episodes_per_experiment=1
n_steps_per_episode=500
target=./results

n_budgets="5 10 15 20"
b_good_origin_flags="--no-good-origin ''"

# entropy=
entropy=B76A074C23C703767710E1D756F73AE9

# budgets
for n_budget in $n_budgets; do
    for b_good_origin in $b_good_origin_flags; do
        python run_xp1.py                                            \
            --entropy=$entropy                                       \
            --path=$target                                           \
            --n_population=$n_population                             \
            $b_good_origin                                           \
            --n_arms=$n_arms                                         \
            --n_budget=$n_budget                                     \
            --n_experiments=$n_experiments                           \
            --n_episodes_per_experiment=$n_episodes_per_experiment   \
            --n_steps_per_episode=$n_steps_per_episode
    done
done
