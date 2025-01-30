import numpy as np
from numpy import ndarray

from numpy.random import Generator

import gurobipy as gp
from gurobipy import GRB

from .whittle import Whittle, batched_whittle_vi_inf_bisect

from ..envs.mdp import MDP


def ucb(n_kas, T: int, /, C: float = 1.0, delta: float = 1e-3):
    """UCB adjustment for probability estimates."""

    n_arms, n_actions, n_states = n_kas.shape

    # u_{aks} = \sqrt{
    #     \frac{
    #         \log\bigl(2 A S N (T+1)^4 \frac1\delta \bigr)
    #     }{n_{aks} + 1}
    # }
    # XXX why this formula? better citation needed
    log_term = (
        np.log(2)
        + (np.log(n_states) + np.log(n_actions) + np.log(n_arms))
        + 4 * np.log(T + 1)
        - np.log(delta)
    )
    return C * np.sqrt(2 * n_states * log_term / np.maximum(n_kas, 1))


class BaseUCWhittle(Whittle):
    n_actions: int = 2  # binary
    gamma: float

    # hidden parameter
    n_horizon: int

    # attributes
    # `n_{kas}` is the number times action `a` was played to arm `k` in state `s`
    n_kas_: ndarray[int]  # (N, A, S)

    # the current estimate of the `p_k(x=1 | s, a)` on the past `n_{aks}` data
    p_kas1_: ndarray[float]  # (N, A, S)

    # the whittle index
    whittle_ks_: ndarray[np.float32]  # (N, S)

    # internally managed experience replay buffer
    replay_: list[tuple]
    n_updates_: int

    # disable sneak-peek
    sneak_peek = None

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        gamma: float,
        C: float = 1.0,
        n_horizon: int = 20,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(n_max_steps, budget, n_states, gamma, random=random)

        assert isinstance(C, float) and C > 0
        self.C = C

        assert isinstance(n_horizon, int) and n_horizon > 0
        self.n_horizon = n_horizon

    def __repr__(self) -> str:
        # we report gamma (rl) as beta (contol theory)
        parstr = f"($\\beta$={self.gamma}, H={self.n_horizon}, C={self.C})"
        return type(self).__name__ + parstr

    def setup_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, fin, random=random)

        # init the arm-state-action value table
        shape = self.n_arms_in_, self.n_actions, self.n_states
        self.n_kas_ = np.zeros(shape, int)
        self.p_kas1_ = np.zeros(shape, float)

        self.replay_ = []
        self.whittle_ks_ = random.normal(size=(self.n_arms_in_, self.n_states))
        self.n_updates_ = 0

        return self

    def update_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        """Update the q-function estimate on the batch of transitions."""

        # collect the transitions into a buffer to be used for an all-at-once update
        self.replay_.append((obs, act, rew, new, fin))  # `(b, ...)`
        n_experience = sum(len(x) for x, *_ in self.replay_)
        if n_experience < self.n_horizon:
            return self

        # collate
        obs, act, rew, new, fin = map(np.concatenate, zip(*self.replay_))
        self.replay_.clear()
        self.n_updates_ += 1

        super().update_impl(obs, act, rew, new, fin, random=random)

        # update of the arm-state-action probability of x=1
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # `m_{aks}` the number of sample which had action `a` applied to arm `k`
        #  in state `s`
        m_kas = np.zeros_like(self.n_kas_, int)
        np.add.at(m_kas, (idx, act, obs), 1)
        self.n_kas_ += m_kas

        # m_{sk} \bar{x}_{m_{sk}} = \sum_{t: x^k_t = s, a^k_t > 0} r^k_t
        upd_kas = -m_kas * self.p_kas1_
        np.add.at(upd_kas, (idx, act, obs), new == 1)

        # update the average per arm-action-state probability estimate
        np.divide(upd_kas, self.n_kas_, where=self.n_kas_ > 0, out=upd_kas)
        self.p_kas1_ += upd_kas

        # get the extreme-points estimate of `p_k(x=1 | s, a)`
        self.compute_whittle()

        return self

    def compute_whittle(self):
        raise NotImplementedError

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    # decide_impl defaults to the implementation in Whittle.decide_impl


class UCWhittleExtreme(BaseUCWhittle):
    def compute_whittle(self):
        ub_kas = ucb(self.n_kas_, self.n_max_steps, C=self.C)
        xtreme = np.stack(
            (
                self.p_kas1_[:, 0, :] - ub_kas[:, 0, :],  # lcb for passive
                self.p_kas1_[:, 1, :] + ub_kas[:, 1, :],  # ucb for active
            ),
            axis=-2,
        )

        # clip and make into a proper Markov kernel
        p_kas1 = np.clip(xtreme, 0, 1)
        p_kasx = np.stack((1 - p_kas1, p_kas1), axis=-1)  # -> kasx

        # compute the whittle index for all arms and all states
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], p_kasx.shape)
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(p_kasx, r_kasx, gam=self.gamma)
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self


class UCWhittleUCB(BaseUCWhittle):
    def compute_whittle(self):
        ub_kas = ucb(self.n_kas_, self.n_max_steps, C=self.C)

        # clip and make into a proper Markov kernel
        p_kas1 = np.clip(self.p_kas1_ + ub_kas, 0, 1)
        p_kasx = np.stack((1 - p_kas1, p_kas1), axis=-1)  # -> kasx

        # compute the whittle index for all arms and all states
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], p_kasx.shape)
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(p_kasx, r_kasx, gam=self.gamma)
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self


def pv_problem_grb(
    p0: ndarray, p_lb: ndarray, p_ub: ndarray, /, rew: ndarray, *, gam: float
) -> gp.Model:
    r"""Solve the Pv-problem from [1]_ with pre-adjusted rewards.

    Notes
    -----
    Due to the extermal slow-ish solver, the :math:`\mathcal{P}_v` problem

    .. math::
        \mathcal{P}_v
            = \max_{v, p} \bigl\{
                \sum_s w_s v(s)
                \colon v = \max_a T^p_a v
                \,, p \in \mathcal{P}
                \,, T^p_a v[s]  % <<-- bellman backup for playing a at s
                    = \mathbb{E}_{p(r, x\mid s, a)} r + \gamma v(x)
            \bigr\}
            \,,

    followed by the Bisect-VI Whittle seems to be faster and more stable, than
    the subsidy problem :math:`\mathcal{P}_m`, despite sec 5.4 from [1]_
    > Computing a Whittle index involves binary search, solving value iteration
      at every step, so is quite computationally expensive.

    References
    ----------
    .. [1] Wang, Kai, Lily Xu, Aparna Taneja, and Milind Tambe, (2023) "Optimistic
       whittle index policy: Online learning for restless bandits." In Proceedings
       of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8,
       pp. 10131-10139.
       https://ojs.aaai.org/index.php/AAAI/article/view/26207
    """
    p0, p_lb, p_ub = np.broadcast_arrays(p0, p_lb, p_ub)
    batch, n_actions, n_states = p_lb.shape
    assert n_states == n_actions == 2, p_lb.shape
    assert np.all(p_lb <= p_ub)
    assert np.all((p_lb <= p0) & (p0 <= p_ub))

    Arms, States, Actions = tuple(range(batch)), (0, 1), (0, 1)

    # p_lb, p_ub is (N, 2, 2)
    # the values of the reward are baked into `good_to_act` and `good_in_good_state`
    # rew = np.broadcast_to(np.r_[0.0, 1.0], (n_arms, n_actions, n_states, n_states))
    # p_lb, p_ub is (N, 2, 2)
    m = gp.Model("UCW-Pv")
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)  # nonconvex constraints
    m.setParam("IterationLimit", 100)  # limit number of simplex iterations

    # `p_{as} \in [l_{as}, u_{as}]` is the probability `p(x=1 | s, a)`
    p = m.addVars(
        Arms,
        Actions,
        States,
        lb=np.clip(p_lb.ravel(), 0.0, 1.0),
        ub=np.clip(p_ub.ravel(), 0.0, 1.0),
        vtype=GRB.CONTINUOUS,
        name="p",
    )
    for k, v in p.items():
        v.Start = p0[k]

    # `v_{s}` is the value function `v(s)`
    V = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="V")

    # `q_{0s}` is the value function `q(s, a=0)`
    Q0 = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="Q0")
    m.addConstrs(
        (
            (
                (rew[k, 0, s, 0] + gam * V[k, 0]) * (1 - p[k, 0, s])
                + (rew[k, 0, s, 1] + gam * V[k, 1]) * p[k, 0, s]
            )
            == Q0[k, s]
            for k in Arms
            for s in States
        ),
        name="q0_defn",
    )

    # `q_{1s}` is the value function `q(s, a=1)`
    Q1 = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="Q1")
    m.addConstrs(
        (
            (
                (rew[k, 1, s, 0] + gam * V[k, 0]) * (1 - p[k, 1, s])
                + (rew[k, 1, s, 1] + gam * V[k, 1]) * p[k, 1, s]
            )
            == Q1[k, s]
            for k in Arms
            for s in States
        ),
        name="q1_defn",
    )

    # Bellman eqn. constraints `v_k(s) = \max_a \tilde{T}^{p_k}_a v_k[s]` where
    #     \tilde{T}^p_a J[s]
    #         = E_{p(x| a, s)} r(a, s, x) - \lambda_{ks} 1_{a==1} + \gamma J(x)
    m.addConstrs(
        (V[k, s] == gp.max_([Q0[k, s], Q1[k, s]]) for k in Arms for s in States),
        name="bellman",
    )

    # define the objective
    m.setObjective(sum(V.values()), GRB.MAXIMIZE)

    # solve
    m.optimize()

    return np.reshape([v.x for v in p.values()], p_lb.shape)


class UCWhittlePv(BaseUCWhittle):
    r"""Upper Confidence Whittle policy with Pv problem for optimistic transition
    probabilities within the confidence box followed by Bisect-VI whittle index
    computation.

    Notes
    -----
    The policy interacts in the env for no more no less than :math:`T` -- the maximal
    allotted budget of interactions in the environment. :math:`T` is directly related
    to sample complexity and the :math:`T` used in regret bounds. After every
    interaction (assuming non-batched rollout, see `play.py:rollout`), the policy
    receives (via `.update`) the observed transition :math:`s_t, a_t \to r_t, s_{t+1}`
    form the environment. Now it is up to the policy's internal logic to decide what
    to do with this new observation. Some policies, like `Whittle` ignore this data,
    because they are privy to the privileged information about the true Markov kernel
    :math:`p_K(x\mid s, a)` and the true reward function :math:`r(s, a, x)` from the
    environment itself (see `sneak_peek`). Other policies, like WIQL and LGGT, use
    this transition to __immediately__ update their internal state (e.g. incremental
    averages, q-function estimates etc.).

    The UCW-family of policies, however, instead of updating immediately, stores
    this new transition in an experience replay buffer [2]_. The buffer is not used
    as dataset of past interactions for off-policy updates [3]_ (ch. 11), but rather
    for delaying the full update and re-computing of the Whittle indices. Thus,
    despite interacting with the env on __every__ step, the UCW policy, unlike the
    above mentioned ones, updates itself only every `n_horizon` steps (i.e. when
    the experience buffer overflows). This means that during these `n_horizon` steps,
    UCW pulls the arms based on slightly stale, un-updated whittle subsidies. This
    keeps the algorithm on-policy with policy improvement steps.

    This is a minor refactor of the pseudocode from [1]_ sec. 5. The first change is
    that the concept of episodes is abandoned, and, instead, the policy's stepping
    is tied to the global number of interactions in the env (via `env.step`).

    This is the pseudocode of this implementation in terms of the lines of the
    pseudocode in [1]_:

        1. initialize :math:`\pi^{(0)}` to a random subset policy
        2. set the whittle subsidies :math:`\lambda^{(0)}` to zero
        3. set :math:`\tau = 0`
        4. for t = 0, 1, 2, ... do
           41. step through env with the current :math:`\pi^{(\tau)}` collecting
               the t-th `sa -> rx` transition into the buffer
               - line 8 (the about H-steps)
           42. if :math:`t < (\tau + 1) H`, goto next iteration of the loop 4.
           43. update the running :math:`\hat{p}_k(x=1\mid s, a)` estimate and the
               state-action counters :math:`n_k(s, a)` on the data from the buffer
               - line 9
           44. reset the buffer
           45. solve the :math:`\mathcal{P}_v` problem for the new confidence
               box and retrieve the optimistic :math:`p_k(x=1\mid s, a)` solution
               - line 6
           46. compute the whittle indices :math:`S \mapsto \lambda_{ks}` using
               Bisect-VI (see `whittle.py`)
               - line 7 (unclear at which state the top-k lambda on line 10)
           47. compute the :math:`\pi^{(\tau+1)}` and increment :math:`\tau`

    The second change is related our introduction a multiplier for the UCB, which
    is :math:`C=1` in [1]_ eqn. (9). We noticed that the confidence box for the
    transition probability estimate is so large, as to frequently getting clipped
    by :math:`[0, 1]^{K \lvert S \rvert \lvert A \rvert}`. This dramatically slowed
    down the convergence of the UCW-policy. By using the value of :math:`\frac{1}{10}`
    we manged to recover competitive performance of the UCW-family.

    Finally, as of 2025-01-30, this implementation forces :math:`\lambda` in
    :math:`\mathcal{P}_v` of [1]_ to zero because from their pseudocode the meaning
    of top-:math:`k` of :math:`\lambda` is unclear, because in order to be playable
    by the policy on line 8 of their pseudocode, the whittle index of the k-th arm
    must be computed for all potential :math:`S` states of that arm. Lucky for us,
    the :math:`p \in \arg\max_{p_{kas} \in [u_{kas}, l_{kas}]} \mathcal{P}_v(p)`
    with :math:`\lambda = 0` already yields competitive performance.

    References
    ----------
    .. [1] Wang, Kai, Lily Xu, Aparna Taneja, and Milind Tambe, (2023) "Optimistic
       whittle index policy: Online learning for restless bandits." In Proceedings
       of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8,
       pp. 10131-10139.
       https://ojs.aaai.org/index.php/AAAI/article/view/26207

    .. [2] William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio,
       Hugo Larochelle, Mark Rowland, Will Dabney, (2020) "Revisiting Fundamentals
       of Experience Replay" Proceedings of the 37th International Conference on
       Machine Learning, PMLR 119:3061-3071, 2020.
       https://proceedings.mlr.press/v119/fedus20a.html

    .. [3] Richard S. Sutton, Andrew G. Barto (2018) "Reinforcement Learning:
       An Introduction" Second edition, MIT Press, 2018, ISBN 9780262352703
    """

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        gamma: float,
        C: float = 0.1,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(
            n_max_steps, budget, n_states, gamma, C, n_horizon=20, random=random
        )

        assert isinstance(C, float) and C > 0
        self.C = C

    def compute_whittle(self):
        ub_kas = ucb(self.n_kas_, self.n_updates_, C=self.C)  # narrower CI

        # plrepare the feasible transition box
        p_lb = np.clip(self.p_kas1_ - ub_kas, 0, 1)
        p_ub = np.clip(self.p_kas1_ + ub_kas, 0, 1)
        if not hasattr(self, "p_opt_"):
            self.p_opt_ = (p_lb + p_ub) / 2

        # project into the new box
        p_opt_ = np.clip(self.p_opt_, p_lb, p_ub)

        # split arms into chunks and solve Pv for each
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], (*self.p_kas1_.shape, 2))

        sols, p1, n_chunk_size = [], 0, 5  # 10n < 200 (upper limit for nonlin cons)
        while p1 < len(self.p_kas1_):
            p0, p1 = p1, p1 + n_chunk_size
            sol = pv_problem_grb(
                p_opt_[p0:p1],
                p_lb[p0:p1],
                p_ub[p0:p1],
                r_kasx[p0:p1],
                gam=self.gamma,
            )
            sols.append(sol)

        self.p_opt_ = np.concatenate(sols, axis=0)

        # compute the whittle index for all arms and all states
        p_kasx = np.stack((1 - self.p_opt_, self.p_opt_), axis=-1)  # -> kasx
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(p_kasx, r_kasx, gam=self.gamma)
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self


def pv_problem_grb_oracle(
    x0: ndarray,
    p_k0s1: ndarray,
    lb_k1s1: ndarray,
    ub_k1s1: ndarray,
    /,
    rew: ndarray,
    *,
    gam: float,
) -> gp.Model:
    r"""Solve the Pv-problem from [1]_ with pre-adjusted rewards and frozen
    :math:`p_k(x=1 \mid s, a=0)`.

    Notes
    -----
    The problem is

    .. math::
        \mathcal{P}_v
            = \max_{v, p} \bigl\{
                \sum_s w_s v(s)
                \colon v = \max_a T^p_a v
                \,, p \in \mathcal{P}
                \,, p_{k0sx} \text{ fixed}
                \,, T^p_a v[s]  % <<-- bellman backup for playing a at s
                    = \mathbb{E}_{p(r, x\mid s, a)} r + \gamma v(x)
            \bigr\}
            \,.

    References
    ----------
    .. [1] Wang, Kai, Lily Xu, Aparna Taneja, and Milind Tambe, (2023) "Optimistic
       whittle index policy: Online learning for restless bandits." In Proceedings
       of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8,
       pp. 10131-10139.
       https://ojs.aaai.org/index.php/AAAI/article/view/26207
    """

    x0, p_k0s1, lb_k1s1, ub_k1s1 = np.broadcast_arrays(x0, p_k0s1, lb_k1s1, ub_k1s1)
    batch, n_states = lb_k1s1.shape
    assert n_states == 2, lb_k1s1.shape
    assert np.all(lb_k1s1 <= ub_k1s1)
    assert np.all((lb_k1s1 <= x0) & (x0 <= ub_k1s1))

    Arms, States = tuple(range(batch)), (0, 1)

    m = gp.Model("UCW-Pv+Priv")
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)  # nonconvex constraints
    m.setParam("IterationLimit", 100)  # limit number of simplex iterations

    # `p_{k1s} \in [l_{k1s}, u_{k1s}]` is the probability `p(x=1 | s, a=1)`
    p_k1s1 = m.addVars(
        Arms,
        States,
        lb=np.clip(lb_k1s1.ravel(), 0.0, 1.0),
        ub=np.clip(lb_k1s1.ravel(), 0.0, 1.0),
        vtype=GRB.CONTINUOUS,
        name="p",
    )
    for k, v in p_k1s1.items():
        v.Start = x0[k]

    # `v_{s}` is the value function `v(s)`
    V = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="V")

    # `q_{0s}` is the value function `q(s, a=0)`
    Q0 = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="Q0")
    m.addConstrs(
        (
            (
                # (rew[k, 0, s, 0] + gam * V[k, 0]) * (1 - p_k0s1[k, s])
                # + (rew[k, 0, s, 1] + gam * V[k, 1]) * p_k0s1[k, s]
                rew[k, 0, s, 0]
                + p_k0s1[k, s] * (rew[k, 0, s, 1] - rew[k, 0, s, 0])
                + gam * (V[k, 0] + p_k0s1[k, s] * (V[k, 1] - V[k, 0]))
            )
            == Q0[k, s]
            for k in Arms
            for s in States
        ),
        name="q0_defn",
    )

    # `q_{1s}` is the value function `q(s, a=1)`
    Q1 = m.addVars(Arms, States, vtype=GRB.CONTINUOUS, name="Q1")
    m.addConstrs(
        (
            (
                (rew[k, 1, s, 0] + gam * V[k, 0]) * (1 - p_k1s1[k, s])
                + (rew[k, 1, s, 1] + gam * V[k, 1]) * p_k1s1[k, s]
            )
            == Q1[k, s]
            for k in Arms
            for s in States
        ),
        name="q1_defn",
    )

    # Bellman eqn. constraints `v_k(s) = \max_a \tilde{T}^{p_k}_a v_k[s]` where
    #     \tilde{T}^p_a J[s]
    #         = E_{p(x| a, s)} r(a, s, x) - \lambda_{ks} 1_{a==1} + \gamma J(x)
    m.addConstrs(
        (V[k, s] == gp.max_([Q0[k, s], Q1[k, s]]) for k in Arms for s in States),
        name="bellman",
    )

    # define the objective
    m.setObjective(sum(V.values()), GRB.MAXIMIZE)

    # solve
    m.optimize()

    return np.reshape([v.x for v in p_k1s1.values()], lb_k1s1.shape)


class UCWhittlePvPriv(UCWhittlePv):
    def sneak_peek(self, env, /) -> None:
        """Break open the black box and rummage in it for unfair advantage.

        Notes
        -----
        Make sure NOT to reset the GT advantage in `setup_impl` if `sneak-peek`
        is called before the very first update (which automatically does setup).
        """

        # ideally, whatever we get a hold of here should be estimated from some
        #  sample collected burn-in period by standard means
        if not isinstance(env, MDP):
            raise NotImplementedError(type(env))

        # Get the true transition probability `p_k(x=1 \mid a=0, s)` from arm `k`
        #  at state `s` to state `x=1` if it is not pulled `a=0`
        self.p_k0s1_ = env.kernels[:, 0, :, 1]  # XXX (N, S)

    def compute_whittle(self):
        ub_kas = ucb(self.n_kas_, self.n_updates_, C=self.C)

        # plrepare the feasible transition box
        lb_k1s1 = np.clip(self.p_kas1_ - ub_kas, 0, 1)[:, 1, :]
        ub_k1s1 = np.clip(self.p_kas1_ + ub_kas, 0, 1)[:, 1, :]
        if not hasattr(self, "p_opt_"):
            self.p_opt_ = (lb_k1s1 + ub_k1s1) / 2

        # project into the new box
        p_opt_ = np.clip(self.p_opt_, lb_k1s1, ub_k1s1)

        # split arms into chunks and solve Pv for each
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], (*self.p_kas1_.shape, 2))

        sols, p1, n_chunk_size = [], 0, 5  # 8n < 200 (upper limit for nonlin cons)
        while p1 < len(self.p_kas1_):
            p0, p1 = p1, p1 + n_chunk_size
            sol = pv_problem_grb_oracle(
                p_opt_[p0:p1],
                self.p_k0s1_[p0:p1],
                lb_k1s1[p0:p1],
                ub_k1s1[p0:p1],
                r_kasx[p0:p1],
                gam=self.gamma,
            )
            sols.append(sol)

        self.p_opt_ = np.concatenate(sols, axis=0)  # (N, S) p_k(x=1 \mid s, a=1)

        # compute the whittle index for all arms and all states
        p_kas1 = np.stack((self.p_k0s1_, self.p_opt_), axis=-2)  # (N, A, S)
        p_kasx = np.stack((1 - p_kas1, p_kas1), axis=-1)  # -> kasx
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(p_kasx, r_kasx, gam=self.gamma)
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self
