import numpy as np
from numpy import ndarray

from numpy.random import Generator

from .base import BasePolicy
from ..envs.mdp import MDP


def lcb(N, T=None, /, C: float = -1.0):
    r"""unsigned LCB term for :math:`N` samples

    .. math::

        \sqrt{\frac{\log (2 + N)}{2 + N}}
    """

    Np1 = 1 + N  # XXX why `+1` here and on the line below?
    # XXX the `\log` in the numerator is novelty!
    return -C * np.sqrt(np.log(1 + Np1) / (1 + Np1))
    # return 0.0
    # return -C * np.sqrt((np.log(np.log(Np1 + 1)) + 2 * np.log(T * 10)) / Np1)
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


class LGGTWUF(BasePolicy):
    """Lcb-guided greedy thresholding policy for binary joint-control multi-arm mdps."""

    n_actions: int = 2  # binary
    threshold: float

    # attributes
    # `n_{sk}` (pseudo-)count of pulls of arm `k` at state `s`
    n_pulls_sk_: ndarray[int]  # (S, N)

    # `q_{sk}` -- estimated immediate reward for pulling arm `k` at state `s` (i.e.
    #  playing `a_k = 1`). This is not quite a q-fun, since it assumes one-off
    #  interaction and no future policy to follow). Also, it has no look-ahead:
    #     \hat{q}_{kas} \approx E_{p_k(x|s, a)} r_{kasx} + \gamma \hat{v}_{kx} \,,
    #  with
    #     \hat{v}_{kx} = q_{k x a_{kx}} \,, a_{kx} \in \arg\max_a q_{kax} \,.
    avg_rew_sk_: ndarray[float]  # (S, N)

    # expected reward for not-pulling an arm `k` (i.e. playing `a_k = 0`) at state `s`
    phi_sk_: ndarray[float]  # (S, N)

    def __repr__(self) -> str:
        return type(self).__name__ + f"($\\gamma$={self.threshold},$C$={self.C})"

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        threshold: float,
        C: float,
        C1: float,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(n_max_steps, budget, n_states, random=random)

        assert isinstance(threshold, float) and -100 <= threshold <= 3
        self.threshold = threshold
        self.C = C
        self.C1 = C1

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

        # Get the expected reward `\phi_k(a=0, s)` from arm `k` at state `s` if
        #  it is not pulled `a=0`: `\phi_k(a, s) = \sum_x r_k(a, s, x) p_k(x | a, s)`
        # XXX note the __transposed__ output dims: `phi_sk_` is `(S, N,)`!
        self.phi_sk_ = np.einsum("kasx,kasx->ask", env.kernels, env.rewards)[0]

    def setup_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, fin, random=random)

        # init the state-arm tables
        shape = self.n_states, self.n_arms_in_
        self.n_pulls_sk_ = np.zeros(shape, int)
        self.avg_rew_sk_ = np.zeros(shape, float)

        # expected reward of arm `k` leaving state `s` under action `a_k = 0`
        # XXX DO NOT RESET privilieged information!
        if not hasattr(self, "phi_sk_"):
            self.phi_sk_ = np.zeros((self.n_states, self.n_arms_in_), float)

        return self

        # SANITY CHECK (remove when publishing)
        self.run_sum_rew_sk_ = np.zeros(shape, float)

    def update_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        super().update_impl(obs, act, rew, new, fin, random=random)

        # update of the state-arm pull q-value tables
        # XXX this does `\mu_{n+m} - \mu_n = \frac{m}{n+m} (\bar{x}_m - \mu_n)`
        #     with per-state-arm m, n, and \bar{x}_m
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # m_{sk} = \sum_{t: x^k_t = s} 1_{a^k_t > 0}
        # XXX scatter-add:
        #   update_[s, k] = \sum_{b: obs[b, k] = s, act[b, k] > 0} rew[b, k]
        m_sk = np.zeros_like(self.n_pulls_sk_)
        np.add.at(m_sk, (obs, idx), act != 0)  # `m_{sk}`

        # make sure to keep track of pull counts (any non-zero action)
        self.n_pulls_sk_ += m_sk

        # m_{sk} \bar{x}_{m_{sk}} = \sum_{t: x^k_t = s, a^k_t > 0} r^k_t
        upd_sk = -m_sk * self.avg_rew_sk_
        val_ = np.where(act != 0, rew, 0.0)
        np.add.at(upd_sk, (obs, idx), val_)

        # update the average per state-arm reward estimate (q-value of pull)
        np.divide(upd_sk, self.n_pulls_sk_, where=self.n_pulls_sk_ > 0, out=upd_sk)
        self.avg_rew_sk_ += upd_sk

        return self

        # SANITY CHECK: test out incremental mean update against a runnning sum
        np.add.at(self.run_sum_rew_sk_, (obs, idx), val_)

        # mean_rew_sk_ is zero if n_pulls_sk_ is zero, otherwise it is the ratio
        #   of run_sum_rew_sk_ to n_pulls_sk_ (see docs pf numpy.divide)
        mean_rew_sk_ = np.zeros_like(self.run_sum_rew_sk_)
        np.divide(
            self.run_sum_rew_sk_,
            self.n_pulls_sk_,
            where=self.n_pulls_sk_ > 0,
            out=mean_rew_sk_,  # <- INPALCE
        )
        assert np.allclose(self.avg_rew_sk_, mean_rew_sk_)

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""
        # `obs` is (batch, n_arms) and, if initialized, `n_arms == .n_arms_in_`
        # XXX usually `batch == 1`, i.e. single-element batch of observations.
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # get incremental reward: force it to +infty if the arm has never been
        #  interacted with, -- this guarantees that it will be pulled!
        inc_rew_sk_ = np.where(
            self.n_pulls_sk_ > 0, self.avg_rew_sk_ - self.phi_sk_, np.inf
        )

        # get `(b, k) -> ucb_[obs[b, k], k]` and lcb -- the upper/lower confidence
        #  bounds of each arm `k` at its observed state in the batch item `b`
        # XXX each arms gets at most one pull per step so
        #     `np.sum(self.n_pulls_sk_) <= self.n_max_steps`
        lcb_ = inc_rew_sk_ - lcb(self.n_pulls_sk_, self.n_max_steps, self.C)
        sample_ = inc_rew_sk_ + lcb(self.n_pulls_sk_, self.n_max_steps, self.C1)
        ucb_ = inc_rew_sk_ + ucb(self.n_pulls_sk_, self.n_max_steps)
        sample_bk, lcb_bk, ucb_bk = sample_[obs, idx], lcb_[obs, idx], ucb_[obs, idx]

        # lexsort: first order the arms (in each batch item) by increasing ucb,
        #  then stably sort by thresholded lcb in ascending order. Thresholding
        #  makes the sorting key insensitive to values lower than `\gamma`, hence
        #  preserving the order-by-ucb. which is what we actually need in the algo!
        # XXX in `.lexsort` the last key in the tuple is primary
        order = np.lexsort((ucb_bk, np.clip(lcb_bk, min=self.threshold)), axis=-1)
        m, n = order.shape
        B = self.budget

        # 1) Create an (m x B) array initialized to -1.
        #    This will hold up to B valid columns per row.
        subsets = np.full((m, B), -1, dtype=int)

        # 2) For each row, take the top-B columns (from order[..., -B:]),
        #    and filter by sample_bk > 0.
        for i in range(m):
            # Extract the top-B columns for row i (shape: (B,))
            top_cols = order[i, -B:]

            # Filter: keep only those with sample_bk > 0
            valid_mask = sample_bk[i, top_cols] >= 0
            valid_cols = top_cols[valid_mask]

            # If we have valid columns, put them in subsets[i], up to B
            if valid_cols.size > 0:
                used_count = min(B, valid_cols.size)
                subsets[i, :used_count] = valid_cols[:used_count]
                # remaining positions stay at -1

        # 3) Determine how many columns each row actually used ( != -1 ), then
        #    take the min across rows. That's how many columns *all* rows have.
        used_per_row = np.sum(subsets != -1, axis=1)  # shape: (m,)
        if used_per_row.size == 0:
            # Edge case: if m=0 or something unexpected
            final_cols = 0
        else:
            # The smallest number of used columns among all rows
            final_cols = used_per_row.min()

        # 4) Slice 'subsets' to keep only those columns used by *every* row
        subsets = subsets[:, :final_cols]

        # b -> {k: lcb_[obs[b, k], k] < \tau}
        # get the binary interaction mask (integers)
        # XXX we don't care if we are under budget
        subsetsr = np.zeros(lcb_bk.shape, int)
        np.put_along_axis(subsetsr, subsets, 1, axis=-1)

        return subsetsr
