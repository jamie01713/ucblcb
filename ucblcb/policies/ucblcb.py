import numpy as np
from numpy import ndarray

from numpy.random import Generator

from .base import BasePolicy


class UcbLcb(BasePolicy):
    """UCBLCB policy for binary multi-arm mdps."""

    gamma: float
    n_states: int = 2  # binary
    n_actions: int = 2  # binary

    # attributes
    # `n_{sa}` (pseudo-)count of pulls of arm `a` at state `s`
    n_pulls_sa_: ndarray[int]  # (S, N)

    # `q_{sa}` -- estimated rew for pulling arm `a` at state `s`  (not qfn, since
    #  it has no lookahead: `q_{sa} ~ E_{xr|sa} r_{sax} + \gamma \hat{v}_x` with
    #  `\hat{v}_x = \max_k q_{xk}`)
    avg_rew_sa_: ndarray[float]  # (S, N)

    #  its lower- and upper- confidence bounds
    q_lcb_: ndarray[float]  # (S, N)
    q_ucb_: ndarray[float]  # (S, N)

    def __init__(
        self,
        budget: int,
        /,
        n_states: int,
        *,
        gamma: float,
        random: Generator = None,
    ) -> None:
        super().__init__(budget)
        assert n_states > 0
        self.n_states = n_states

        assert isinstance(gamma, float) and 0 <= gamma <= 1
        self.gamma = gamma

    def setup_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, random=random)

        # init the state-arm tables
        shape = self.n_states, self.n_arms_in_
        self.n_pulls_sa_ = np.zeros(shape, int)
        self.avg_rew_sa_ = np.zeros(shape, float)

        # prepare the lower- and upper- confidence bounds
        self.q_lcb_ = np.full(shape, -np.inf, float)
        self.q_ucb_ = np.full(shape, +np.inf, float)

        return self

    def update_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        super().update_impl(obs, act, rew, new, random=random)

        # update of the state-arm pull q-value tables
        # XXX this does `\mu_{n+m} - \mu_n = \frac{m}{n+m} (\bar{x}_m - \mu_n)`
        #     with per-state-arm m, n, and \bar{x}_m
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # m_{sa} = \sum_{t: x_{ta} = s} 1_{a_{ta} > 0}
        # XXX scatter-add:
        #   update_[s, a] = \sum_{b: obs[b, a] = s, act[b, a] > 0} rew[b, a]
        m_sa = np.zeros_like(self.n_pulls_sa_)
        np.add.at(m_sa, (obs, idx), act != 0)  # `m_{sa}`

        # make sure to keep track of pull counts (any non-zero action)
        self.n_pulls_sa_ += m_sa

        # m_{sa} \bar{x}_{m_{sa}} = \sum_{t: x_{ta} = s, a_{ta} > 0} r_{ta}
        upd_sa = -m_sa * self.avg_rew_sa_
        val_ = np.where(act != 0, rew, 0.0)
        np.add.at(upd_sa, (obs, idx), val_)

        # update the average per state-arm reward estimate (q-value of pull)
        np.divide(upd_sa, self.n_pulls_sa_, where=self.n_pulls_sa_ > 0, out=upd_sa)
        self.avg_rew_sa_ += upd_sa

        # deal with lcb/ucb

        return self

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""

        raise NotImplementedError

    # uninitialized_decide_impl defaults to `randomsubset(.budget)`
