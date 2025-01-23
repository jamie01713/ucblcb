import numpy as np
from numpy import ndarray

from numpy.random import Generator

from .base import BasePolicy
from ..envs.mdp import MDP


def lcb(N, T=None, *, C: float = 1.0):
    r"""unsigned LCB term from [1]_ for :math:`N` samples

    .. math::

        \sqrt{\frac{\log (2 + N)}{2 + N}}

    References
    ----------
    .. [1] APA Citation Needed
    """

    Np1 = 1 + N  # XXX why `+1` here and on the line below?
    # XXX the `\log` in the numerator is novelty!
    return C * np.sqrt(np.log(1 + Np1) / (1 + Np1))
    # XXX shouldn't we return to `+\infty` if N is zero, i.e. no samples, to indicate
    #  complete uncertainty, maximal absence of confidence? same with ucb


def ucb(N, T, *, C: float = 1.714):
    r"""UCB double-log term from [1]_ for :math:`N` samples out of :math:`T`

    .. math::

        \sqrt{\frac{\log \log (2 + N) + 2 \log 10 T }{1 + N}}

    References
    ----------
    .. [1] Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021).
       "Time-uniform, nonparametric, nonasymptotic confidence sequences."
       Ann. Statist. 49(2): 1055-1080 (April 2021). https://doi.org/10.1214/20-AOS1991
    """

    Np1 = 1 + N  # XXX why `+1` here?
    return C * np.sqrt((np.log(np.log(Np1 + 1)) + 2 * np.log(T * 10)) / Np1)


class UcbLcb(BasePolicy):
    """UCBLCB policy for binary joint-control multi-arm mdps."""

    n_actions: int = 2  # binary
    threshold: float

    # attributes
    # `n_{sa}` (pseudo-)count of pulls of arm `a` at state `s`
    n_pulls_sa_: ndarray[int]  # (S, N)

    # `q_{sa}` -- estimated immediate reward for pulling arm `a` at state `s` (not
    #  quite q-fun, since assumes one-shot interaction and no policy in the future
    #  trajectory). Also, it has no look-ahead:
    #     \hat{q}_{sa} \approx E_{xr|sa} r_{sax} + \gamma \hat{v}_x \,,
    #  with
    #     \hat{v}_x = q_{xk_x}\,, k_x \in \arg\max_k q_{xk} \,.
    avg_rew_sa_: ndarray[float]  # (S, N)

    #  its lower- and upper- confidence bounds
    q_lcb_: ndarray[float]  # (S, N)
    q_ucb_: ndarray[float]  # (S, N)

    def __repr__(self) -> str:
        return type(self).__name__ + f"(threshold={self.threshold})"

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        threshold: float,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(n_max_steps, budget, n_states)

        assert isinstance(threshold, float) and 0 <= threshold <= 1
        self.threshold = threshold

    def sneak_peek(self, env, /) -> None:
        """Brake open the black box and rummage in it for unfair advantage.

        Notes
        -----
        The title is self-explanatory.

        Any policy that makes use of this method on an environment that it is
        played in should be ashamed of itself, condemned by its peers, and
        shunned by everybody.
        """

        # Access to the env's internals
        assert isinstance(env, MDP)
        raise NotImplementedError

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

        return self

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""
        # `obs` is (batch, n_arms) and, if initialized, `n_arms == .n_arms_in_`
        # XXX usually `batch == 1`, i.e. single-element batch of observations.
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # get `(b, a) -> ucb_[obs[b, a], a]` and lcb -- the upper/lower confidence
        #  bounds of each arm at its observed state in the batch
        # XXX each arms gets at most one pull per step so
        #     `np.sum(self.n_pulls_sa_) <= self.n_max_steps`
        lcb_ = self.avg_rew_sa_ - lcb(self.n_pulls_sa_, self.n_max_steps)
        ucb_ = self.avg_rew_sa_ + ucb(self.n_pulls_sa_, self.n_max_steps)
        lcb_ba, ucb_ba = lcb_[obs, idx], ucb_[obs, idx]

        # lexsort: first order the arms (in each batch item) by increasing ucb,
        #  then stably sort by thresholded lcb in ascending order. Thresholding
        #  makes the sorting key insensitive to values lower than `\gamma`, hence
        #  preserving the order-by-ucb. which is what we actually need in the algo!
        # XXX in `.lexsort` the last key in the tuple is primary
        order = np.lexsort((ucb_ba, np.clip(lcb_ba, min=self.threshold)), axis=-1)

        # b -> {a: lcb_[obs[b, a], a] < \tau}
        # get the binary interaction mask (integers)
        # XXX we don't care if we are under budget
        subsets = np.zeros(order.shape, int)
        np.put_along_axis(subsets, order[..., -self.budget :], 1, axis=-1)

        return subsets
