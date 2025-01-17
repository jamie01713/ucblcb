import numpy as np
import math


def UcbLcb(env, n_episodes, n_epochs, gamma,VERBOSE=False):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    agent     = np.arange(N)
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes
    env.reset_all()
    nu0 = np.zeros((N, n_states))
    nu0 = env.transitions[:, :, 0, 1]  # <- ORACLE
    mu = np.zeros((N, n_states))
    n_pull = np.zeros((N, n_states))
    LCB = np.zeros((N, n_states))
    UCB = np.zeros((N, n_states))
    action = np.zeros(N, dtype=np.int8)
    # env
    all_reward = np.zeros((n_epochs, T + 1))
    for epoch in range(n_epochs):
        if epoch != 0:
            env.reset_instance()

        print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward()
        for t in range(1, T+1):
            state = env.observe()

            candidate = np.where(LCB[:, state] >= gamma)[0]
            if np.size(candidate) >= budget:
                LCBTEMP = LCB.copy()
                LCBTEMP = LCBTEMP[agent, state]
                # print(LCBTEMP)
                active = agent[np.argsort(LCBTEMP)[-budget:]]

            else:
                active = candidate
                activetemp = active.copy()
                candidate1 = np.where(LCB[agent, state] < gamma)[0]
                UCBTEMP = UCB.copy()
                UCBTEMP = UCBTEMP[candidate1, state[candidate1]]
                candidate1 = agent[np.argsort(UCBTEMP)[-budget+np.size(candidate):]]
                active = np.hstack((activetemp, candidate1))

            for i in range(N):
                if i in active:
                    action[i] = 1

                else:
                    action[i] = 0

            next_state, reward, done, _ = env.step(action)
            rewards = env.get_rewards()
            for i in range(N):
                if i in active:
                    mu[i, state[i]] += rewards[i]
                    n_pull[i, state[i]] += 1

                    LCB[i, state[i]] = (
                        mu[i, state[i]] / n_pull[i, state[i]]
                        - nu0[i, int(state[i])]
                        - math.sqrt(
                            math.log(n_pull[i, int(state[i])] + 2)
                            / (n_pull[i, int(state[i])] + 2)
                        )
                    )

                    UCB[i, state[i]] = (
                        mu[i, int(state[i])] / n_pull[i, int(state[i])]
                        - nu0[i, int(state[i])]
                        + 1.714 * math.sqrt(
                            (
                                math.log(math.log(n_pull[i, int(state[i])] + 2))
                                + 2 * math.log(T * 10)
                            )
                            / (n_pull[i, int(state[i])] + 1)
                        )
                    )
            all_reward[epoch, t] = reward
    return (all_reward)
