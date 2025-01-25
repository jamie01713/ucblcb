import numpy as np
from numpy import ndarray

from numpy.random import Generator

from .base import BasePolicy, random_subset


class WIQL(BasePolicy):
    r"""Whittle index-based Q-learning of [1]_.

    Notes
    -----
    In classical tabular Q-learning, the state-action value function is learnt by,
    essentially, doing an :math:`\alpha`-gradient descent step on the TD(0)-error
    loss (:math:`s, a \to r, x`):

    .. math::

        \hat{Q}(s_t, a_t)
            \lefthookarrow \hat{Q}(s_t, a_t)
            + \alpha \mathbb{E}_{p(r_{t+1}, s_{t+1} |mid s_t, a_t)} \bigl(
                r_{t+1} + \gamma_{s_{t+1}} \hat{V}(s_{t+1}) - \hat{Q}(s_t, a_t)
            \bigr)
        \,,

    where the expectation is estimated by a sample average over the collected
    experience, and :math:`\gamma_x` is zero if the destination state :math:`x`
    is terminal or equals the discount factor :math:`\gamma \in [0, 1)` otherwise.
    The state value function is bootstrapped using the current estimate of
    :math:`\hat{Q}` and defined as :math:`\hat{V}(x) = \max_a \hat{Q}(x, a)`.

    Now, sec 5. of [1]_ reads (emphasis added):
    > We adopt Q-Learning for RMABs and __store Q values separately for each
      arm.__ Q values of state-action pairs are typically used for selecting
      the best action for an arm at each state; however, for RMAB, the problem
      is to select M arms. Our action-selection method ensures that an arm
      whose estimated benefit is higher is more likely to get selected.

    Therefore, WIQL maintains an independet Q-function approximation for each arm
    and learns them based on their individual :math:`(s_t, a_t, r_{t+1}, s_{t+1})`
    transitions.

    In deep Q-learning methods the Q-function is approximated by a neural network, and
    to avoid the convergence issues due to appxorimation of the backup, a secondary
    "frozen" target Q-function is introduced to estimate :math:`\hat{V}(s_{t+1})`.
    In small discrete state-action space setting it is tractable to use an exact
    parameterization of the q-function, i.e. as a table indexed by states and actions.

    References
    ----------
    .. [1] Biswas, A., Aggarwal, G., Varakantham, P., & Tambe, M. (2021). "Learn
       to intervene: An adaptive learning policy for restless bandits in application
       to preventive healthcare" In Proceedings of the Thirtieth International Joint
       Conference on Artificial Intelligence, IJCAI-21. Pages 4039-4046.
       https://doi.org/10.24963/ijcai.2021/556
    """

    n_actions: int = 2  # binary
    alpha: float
    gamma: float
    epsilon: float

    # attributes
    # \hat{q}_{aks}
    #     \approx E_{p_k(x|s, a)} r_{kasx} + \gamma \max_a q_{akx}
    qval_aks_: ndarray[float]  # (A, N, S)
    n_pulls_aks_: ndarray[int]  # (A, N, S)

    def __repr__(self) -> str:
        return type(self).__name__ + f"(alpha={self.alpha}, gamma={self.gamma})"

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        alpha: float,
        gamma: float,
        epsilon: float = 0.05,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(n_max_steps, budget, n_states)

        assert isinstance(alpha, float) and 0 <= alpha <= 1
        self.alpha = alpha

        assert isinstance(gamma, float) and 0 <= gamma < 1
        self.gamma = gamma

        assert isinstance(epsilon, float) and 0 <= epsilon < 1
        self.epsilon = epsilon

    def setup_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, random=random)

        # init the arm-state-action value table
        shape = self.n_actions, self.n_arms_in_, self.n_states
        self.qval_aks_ = np.zeros(shape, float)
        self.n_pulls_aks_ = np.zeros(shape, int)
        # self.qval_aks_ = random.normal(size=shape)

        # the expected subsidy
        self.lam_ks_ = np.zeros(shape[1:], float)

        return self

    def update_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        super().update_impl(obs, act, rew, new, random=random)

        # update of the arm-state-actionq-value tables
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # m_{aks} = \sum_{j: x^k_j = s, a^k_j = a} 1
        m_aks = np.zeros_like(self.n_pulls_aks_, int)
        np.add.at(m_aks, (act, idx, obs), 1)
        self.n_pulls_aks_ += m_aks

        # compute `\delta^0_{kj} = r_{kj} + \gamma v_{k x_j} - q_{k s_j a_j}`, where
        #  `v_{ks} = \max_u q_{ksu}`, for all transitions `j` (we do have j+1 here,
        #  since we have agreed that `s, a -->> r, x`)
        # gam = np.where(fin > 0, 0.0, self.gamma)  # fin: +1 terminated, -1 truncated
        gam = self.gamma
        val_star = np.max(self.qval_aks_, axis=0)[idx, new]  # (..., N)
        td0 = rew + gam * val_star - self.qval_aks_[act, idx, obs]  # (..., N)

        # get the sum of errors at each arm-state-action
        td0_aks = np.zeros_like(self.qval_aks_)
        np.add.at(td0_aks, (act, idx, obs), td0)

        # make a step in direction of reducing the td(0) error
        np.divide(td0_aks, m_aks, where=m_aks > 0, out=td0_aks)
        self.qval_aks_ += self.alpha * td0_aks

        return self

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""
        # `obs` is (batch, n_arms) and, if initialized, `n_arms == .n_arms_in_`
        # XXX usually `batch == 1`, i.e. single-element batch of observations.
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # the estimated subsidy for not playing `a=1` for arm `k` in state `s`
        m_ak = self.n_pulls_aks_[:, idx, obs]  # (A, ..., N)
        q_ak = self.qval_aks_[:, idx, obs]  # (A, ..., N)
        q_0k = np.where(m_ak[0] > 0, q_ak[0], -np.inf)  # (..., N)
        q_1k = np.where(m_ak[1] > 0, q_ak[1], np.inf)  # (..., N)
        lam_order = np.argsort(q_1k - q_0k, -1)  # (..., N)

        # get the number of arms in multi-arm 2d+ data
        n_samples, n_arms, *_ = np.shape(obs)

        # sample random subsets for each sample and then pick which action to play
        rnd_subset = random_subset(random, n_arms, self.budget, size=n_samples)

        # get the binary interaction mask (integers)
        lam_subset = np.zeros(lam_order.shape, int)
        np.put_along_axis(lam_subset, lam_order[..., -self.budget :], 1, axis=-1)

        # pick which samples are to be played with randomly
        is_random = random.uniform(size=(n_samples, 1)) < self.epsilon
        return np.where(is_random, rnd_subset, lam_subset)
