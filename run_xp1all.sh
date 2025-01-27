n_experiments=501
n_population=100
n_arms=50
n_episodes_per_experiment=1
n_steps_per_episode=500
target=./results

# entropy=
entropy=B76A074C23C703767710E1D756F73AE9

# budgets
for n_budget in 5 10 15 20; do
    python run_xp1.py                                            \
        --entropy=$entropy                                       \
        --path='$target'                                         \
        --n_population=$n_population                             \
        --n_arms=$n_arms                                         \
        --n_budget=$n_budget                                     \
        --n_experiments=$n_experiments                           \
        --n_episodes_per_experiment=$n_episodes_per_experiment   \
        --n_steps_per_episode=$n_steps_per_episode

done
