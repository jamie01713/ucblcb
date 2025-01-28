import numpy as np
from numpy import ndarray

from numpy.random import Generator

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
    alpha: float | None
    gamma: float

    # attributes
    # `n_{kas}` is the number times action `a` was played to arm `k` in state `s`
    n_kas_: ndarray[int]  # (N, A, S)

    # the current estimate of the `p_k(x=1 | s, a)` on the past `n_{aks}` data
    p_kas1_: ndarray[float]  # (N, A, S)

    # the whittle index
    whittle_ks_: ndarray[np.float32]  # (N, S)

    # disable sneak-peek
    sneak_peek = None

    def setup_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, fin, random=random)

        # init the arm-state-action value table
        shape = self.n_arms_in_, self.n_actions, self.n_states
        self.n_kas_ = np.zeros(shape, int)
        self.p_kas1_ = np.zeros(shape, float)

        return self

    def update_impl(self, /, obs, act, rew, new, fin, *, random: Generator = None):
        """Update the q-function estimate on the batch of transitions."""
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
        ub_kas = ucb(self.n_kas_, self.n_max_steps)
        xtreme = np.stack(
            (
                self.p_kas1_[:, 0, :] - ub_kas[:, 0, :],  # lcb for passive
                self.p_kas1_[:, 1, :] + ub_kas[:, 1, :],  # ucb for active
            ), axis=-2
        )

        # clip and make into a proper Markov kernel
        p_kas1 = np.clip(xtreme, 0, 1)
        p_kasx = np.stack((1 - p_kas1, p_kas1), axis=-1)  # -> kasx

        # compute the whittle index for all arms and all states
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], p_kasx.shape)
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(
            p_kasx, r_kasx, gam=self.gamma
        )
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self


class UCWhittleUCB(BaseUCWhittle):
    def compute_whittle(self):
        ub_kas = ucb(self.n_kas_, self.n_max_steps)

        # clip and make into a proper Markov kernel
        p_kas1 = np.clip(self.p_kas1_ + ub_kas, 0, 1)
        p_kasx = np.stack((1 - p_kas1, p_kas1), axis=-1)  # -> kasx

        # compute the whittle index for all arms and all states
        r_kasx = np.broadcast_to(np.r_[0.0, 1.0], p_kasx.shape)
        _, lam_ks_, _ = batched_whittle_vi_inf_bisect(
            p_kasx, r_kasx, gam=self.gamma
        )
        self.whittle_ks_ = np.asarray(lam_ks_)

        return self
