import numpy as np
from numpy import ndarray

from numpy.random import Generator

import gurobipy as gp
from gurobipy import GRB

from .whittle import Whittle, batched_whittle_vi_inf_bisect


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
        self.replay_.append((obs, act, rew, new, fin))
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
    r"""
    Due to the extermal solver, the :math:`\mathcal{P}_v` problem

    .. math::
        \mathcal{P}_v
            = \max_{v, p} \bigl\{
                \sum_s w_s v(s)
                \colon v = \max_a T^p_a v
                \,, p \in \mathcal{P}
                \,, T^p_a v[s]
                    = \mathbb{E}_{p(r, x\mid s, a)} r + \gamma v(x)
            \bigr\}
            \,,

    followed by Biscet-VI Whittle is faster, than the subsidy problem
    :math:`\mathcal{P}_m`.

    > Computing a Whittle index involves binary search, solving value iteration
      at every step, so is quite computationally expensive.
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
    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        gamma: float,
        C: float = 1.0,
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
