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

    Therefore, WIQL maintains an independent Q-function approximation for each arm
    and learns them based on their individual :math:`(s_t, a_t, r_{t+1}, s_{t+1})`
    transitions.

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
    n_aks_: ndarray[int]  # (A, N, S)

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
        assert isinstance(gamma, float) and 0 <= gamma < 1
        assert isinstance(epsilon, float) and 0 <= epsilon < 1
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def setup_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        """Initialize the ucb-lcb state from the transition."""
        super().setup_impl(obs, act, rew, new, random=random)

        # init the arm-state-action value table
        shape = self.n_actions, self.n_arms_in_, self.n_states
        self.qval_aks_ = np.zeros(shape, float)  # random.normal(size=shape)
        self.n_aks_ = np.zeros(shape, int)

        return self

    def update_impl(self, /, obs, act, rew, new, *, random: Generator = None):
        """Update the q-function estimate on the batch of transitions."""

        # let the base class update the its state
        super().update_impl(obs, act, rew, new, random=random)

        # update of the arm-state-action q-value tables
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # `m_{aks}` the number of sample which had action `a` applied to arm `k`
        #  in state `s`
        # XXX the `obs-act-rew-new` arrays have shape `(batch, N)`
        # XXX strictly speaking in RL terms, our env has vector state space (vector
        #  `s \in S^N` indicating state `s_k` of arm `k`) and vector action space
        #  where `a \in [A]^N` is a vector of control signals applied to each arm:
        #  `a_k` tells us what was issued to the `k`-th arm.
        m_aks = np.zeros_like(self.n_aks_, int)
        np.add.at(m_aks, (act, idx, obs), 1)
        self.n_aks_ += m_aks

        # compute the temporal-difference target for the observed SARX data and each arm
        #  `\delta^0_{jk} = r_{jk} + \gamma \max_u q_k(x_{jk}, u)`
        #  for transition `s_j, a_j \to r_j, x_j` from the batch (origin state `s_j`
        #  consequence state `x_j`).
        td_jk = rew + self.gamma * np.max(self.qval_aks_, axis=0)[idx, new]

        # get the gradient of the td-error ell-2 loss:
        #   \frac1J \sum_{jk} \frac12 (\delta^0_{jk} - q_k(s_{jk}, a_{jk}))^2
        #  the gradient wrt `q_k` as a function is
        #   g: (s, a)
        #       \mapsto \frac1J \sum_{j \colon s_{jk} = s, a_{jk} = a}
        #           (q_k(s_{jk}, a_{jk}) - \delta^0_{jk})
        grad_aks = np.zeros_like(self.qval_aks_)
        qval_jk = self.qval_aks_[act, idx, obs]
        np.add.at(grad_aks, (act, idx, obs), qval_jk - td_jk)
        np.divide(grad_aks, m_aks, where=m_aks > 0, out=grad_aks)

        # make a step in direction of reducing the td(0) error
        # functional gradient descent: q \lefthookarrow q + \alpha \nabla_q L(q)
        self.qval_aks_ -= self.alpha * grad_aks

        return self

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""

        # `obs` is (batch, n_arms) and, if initialized, `n_arms == .n_arms_in_`
        # XXX usually `batch == 1`, i.e. single-element batch of observations.
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # get the number of arms in multi-arm 2d+ data
        n_samples, n_arms, *_ = np.shape(obs)  # XXX `...` below is `n_samples`

        # the estimated subsidy for not playing `a=1` for arm `k` in state `s`
        m_ak = self.n_aks_[:, idx, obs]  # (A, ..., N)
        q_ak = self.qval_aks_[:, idx, obs]  # (A, ..., N)

        # if an arm `k` has never been acted upon by `a` in state `s`, then the
        #  q-value estimate `q_k(s, a)` is undefined.
        q_0k = np.where(m_ak[0] > 0, q_ak[0], -np.inf)
        q_1k = np.where(m_ak[1] > 0, q_ak[1], np.inf)

        # prepare action vector using on the estimated advantages of playing `a_k = 1`
        #  for arm `k` in its current state `s_k`: the higher the advantage, then
        #  more reason to pull. Whittle index's subsidy `\lambda_k` is, roughly,
        #     q_k(s_k, a=1) - \lambda_k \approx q_k(s, a=0) \,.
        subset_lam = np.zeros((n_samples, n_arms), int)
        top_k = np.argsort(q_1k - q_0k, -1)[..., -self.budget :]  # (..., N)
        np.put_along_axis(subset_lam, top_k, 1, axis=-1)

        # draw a random subset for each sample and then pick which action to play
        #  with probability `self.epsilon` (for \epsilon-greedy)
        subset_rnd = random_subset(random, n_arms, self.budget, size=n_samples)

        is_random = random.uniform(size=(n_samples, 1)) < self.epsilon
        return np.where(is_random, subset_rnd, subset_lam)
