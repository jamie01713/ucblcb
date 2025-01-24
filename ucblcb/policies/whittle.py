import numpy as np
from numpy import ndarray

from numpy.random import Generator

from .base import BasePolicy
from .ucblcb import UcbLcb  # we borrow update and setup
from ..envs.mdp import MDP

from scipy.optimize import linprog
from scipy import sparse as sp
from collections.abc import Callable
from functools import partial


is_allclose = partial(np.allclose, rtol=1e-3, atol=1e-5)


def lp_inf(ker: ndarray, rew: ndarray, *, gam: float) -> ndarray:
    r"""Solve the LP equivalent of the infinite horizon optimal control.

    Parameters
    ----------
    ker : array-like of float of shape (..., n_actions, n_states, n_states)
        The Markov transition kernel :math:`P^a_{sx} = p(s \to x \mid s, a)`
        (einsum signature "...asx").

    rew : array-like of float of shape (..., n_actions, n_states, n_states)
        The expected reward :math:`R^a_{sx} = E(R \mid s, a, x)` due to
        :math:`s \to x` under :math:`a` (einsum signature "...asx").

    gam : float
        The discount factor of the MDP.

    Returns
    -------
    val_ : array-like of float of shape (..., n_states,)
        The LP solution to the Bellman optimality equation (the fixed point).

    Notes
    -----
    The infinite horizon bellman optimality equation reads :math:`V = T_\ast V`
    (point-wise over the state space :math:`S`), where :math:`T_\ast = \max_a T_a`
    is the Bellman optimality operator and :math:`T_a` is the Bellman backup
    operator acting on functions :math:`S \to \mathbb{R}`, and defined for
    :math:`s \in S` and :math:`a \in A` as

    .. math::

        T_a J[s]
            = \mathbb{E}_{p(r, x \mid a, s)} r + \gamma J(x)

    (we denote the destination states by :math:`x \in X`, :math:`X = S`).

    The problem of finding the solution :math:`V^\ast` can be equivalently represented
    in the form of a linear program, see sec. 2.2 in [1]_. In particular, for any weight
    vector :math:`w > 0` it holds that :math:`V^\ast` solves the following LP

    .. math::

        \min_V \biggl\{
            \sum_s w_s V(s)
                \colon V(s) \geq T_a V[s]
                \,, \forall a \in A
                \,, s \in S
            \biggr\}
        \,.

    This LP is feasible since the operator :math:`T_\ast` is a contraction for
    :math:`\gamma \in [0, 1)`, and thus has a fixed point.

    The constraints can be trivially rewritten in vector form, since
    :math:`T_a J[s] = \sum_x p_{asx} r_{asx} + \gamma \sum_x p_{asx} J(x)`.
    Letting :math:`P^a = (p_{asx})_{sx}` be the :math:`S \times X` transition
    kernel, and :math:`R^a = (\sum_x p_{asx} r_{asx})_s` be the expected reward
    vector after action :math:`a \in A`, the above LP can be rewritten thus

    .. math::

        \min_{v\in \mathbb{R}^X} \Bigl\{
                w^\top v
                \colon (\gamma P^a - I) v \leq -R^a
                \,, \forall a \in A
            \Bigr\}
        \,.

    The proof in sec. 2.2 in [1]_ follows from the fact that :math:`T_\ast` is a
    :math:`\gamma`-contraction and a monotonic oprator, and that the set of feasibile
    :math:`V \colon S \to \mathbb{R}` of the LP is equivalent to :math:`F_1(T_ast)`,
    where :math:`F_m` is the set of :math:`m`-cycle fixed poitns

    .. math::

        F_m(T)
            = \biggl\{
                V \colon S \to \mathbb{R}
                \colon V = T^m V
            \biggr\}
        \,.

    The equality of the feasibility set enables us to kick-start value iterations
    :math:`V^m = T_\ast V^0` for any LP-feasible :math:`V^0`, which then provide us
    with an "improving" candidate for any feasible incumbent. Furthermore, the
    contraction property implies that that :math:`F_1(T_ast) = F_m(T_ast)`. Indeed,
    by default, :math:`F_1 \subseteq F_m` for all :math:`m \geq 1`, but due to being
    a contraction we have (:math:`\|\cdot\|_\infty` norm)

    ..math::

        \|T U - T V\|
            \leq \gamma \|U - V\|
        \,,

    whence :math:`\|V - T V\| = \|T^m V - T T^m V\| \leq \gamma^m \|V - T V\|`.

    References
    ----------
    .. [1] Daniel Adelman, Adam J. Mersereau, (2008) Relaxations of Weakly Coupled
       Stochastic Dynamic Programs. Operations Research 56(3):712-727.
       https://doi.org/10.1287/opre.1070.0445
       (paywalled)
    """
    assert 0 <= gam < 1

    # `ker` and `rew` must broadcast to each other
    ker, rew = np.broadcast_arrays(ker, rew)  # (N, A, S, X)

    *batch, n_actions, n_states, n_states_ = ker.shape
    assert n_states == n_states_, ker.shape

    # prepare Abc
    c = np.ones((*batch, n_states), ker.dtype)

    # make a block diagonal matrix from (a, s)-flattened blocks `\gamma P^a - I`
    mat = gam * ker - np.eye(n_states, n_states_)
    A_ub = sp.block_diag(np.reshape(mat, (-1, n_actions * n_states, n_states_)))
    b_ub = -np.sum(ker * rew, -1).ravel()

    # `highs` does not use x0, but the following is feasible (for unbatched ker)
    # x0 = np.full(n_states, rew.max() / (1 - gam))
    # violation = np.maximum(A_ub @ x0 - b_ub, 0.0)
    # assert np.allclose(violation, 0.0, atol=1e-6)
    sol = linprog(c.ravel(), A_ub, b_ub, bounds=None, method="highs", x0=None)
    if not sol.success:
        raise ValueError("Could not solve Bellman optimality equation")

    return np.reshape(sol.x.astype(ker.dtype), c.shape)


def qvalue(ker: ndarray, rew: ndarray, val: ndarray, *, gam: float) -> ndarray:
    r"""the Bellman optimality operator :math:`T_\ast` on the given value function.

    Parameters
    ----------
    ker : array-like of float of shape (..., n_actions, n_states, n_states)
        The Markov transition kernel :math:`P^a_{sx} = p(s \to x \mid s, a)`
        (einsum signature "...asx").

    rew : array-like of float of shape (..., n_actions, n_states, n_states)
        The expected reward :math:`R^a_{sx} = E(R \mid s, a, x)` due to
        :math:`s \to x` under :math:`a` (einsum signature "...asx").

    val : array-like of float of shape (..., n_states,)
        The initial value function approximation :math:`v(s)` for
        the value-to-go in state :math:`s` (einsum signature "...s").

    gam : float
        The discount factor of the MDP.

    Returns
    -------
    qfun : array-like of float of shape (..., n_actions, n_states,)
        The q-function :math:`(a, s) \mapsto T_a v[s]`.

    Notes
    -----
    We denote the destination state by :math:`x \in X` with :math:`X = S`, and define
    the Bellman backup operator on functions :math:`S \to \mathbb{R}` as
    .. math::

        T_a J[s]
            = \mathbb{E}_{p(r, x \mid a, s)} r + \gamma J(x)
        \,,

    for :math:`s \in S` and :math:`a \in A`. The Bellman optimality operator is given
    by :math:`T_\ast V[s] = \max_a T_a V[s]`.
    """
    # `ker` and `rew` must broadcast to each other
    ker, rew = np.broadcast_arrays(ker, rew)  # (N, A, S, X)

    # `ker` and `rew` broadcast to `(..., A, S, X)`
    *batch, n_actions, n_states, n_states_ = ker.shape
    assert n_states == n_states_

    # the state-value vector `val` must broadcast over batch and state
    val = np.broadcast_to(val, (*batch, n_states))  # (..., S)

    # q-value backup (X === S)
    # `ker` is (..., A, S, X)
    # `rew` is (..., A, S, X)  # multiplication broadcasts with rew automatically
    # `val` is (...,       S)  # `S = X` need to inject unit dims!
    return np.sum(ker * (rew + gam * np.expand_dims(val, (-3, -2))), -1)


def bellman(ker: ndarray, rew: ndarray, val: ndarray, *, gam: float) -> ndarray:
    r"""the Bellman optimality operator :math:`T_\ast` on the given value function.

    Returns
    -------
    val : array-like of float of shape (..., n_states,)
        The Bellman backup on the value function :math:`s \mapsto T_\ast v[s]`.
        See `qvalue`.
    """

    # use the optimal policy on the q-function `(a, s) \mapsto T_a V[s]`  to get
    #  `V \colon s \mapsto T_\ast J[s] = \max_a T_a J[s]`
    # XXX v(s) = \mathbb{E}_{\pi(a | s)} q(s, a) for \pi_\ast(s) \in \arg\max_a q(s, a)
    return np.max(qvalue(ker, rew, val, gam=gam), -2)  # (..., A, S) -> (..., S)


def vi_inf(
    ker: ndarray,
    rew: ndarray,
    /,
    val: ndarray = None,
    *,
    gam: float,
    is_allclose: Callable = is_allclose,
) -> tuple[int, ndarray]:
    r"""Infinite horizon value iteration on a discrete MDP `(P, R, gam)`.

    Parameters
    ----------
    ker : array-like of float of shape (..., n_actions, n_states, n_states)
        The Markov transition kernel :math:`P^a_{sx} = p(s \to x \mid s, a)`
        (einsum signature "...asx").

    rew : array-like of float of shape (..., n_actions, n_states, n_states)
        The expected reward :math:`R^a_{sx} = E(R \mid s, a, x)` due to
        :math:`s \to x` under :math:`a` (einsum signature "...asx").

    val : array-like of float of shape (..., n_states,)
        The initial value function approximation :math:`v_s` for
        the value-to-go in state :math:`s` (einsum signature "...s").

    gam : float
        The discount factor of the MDP.

    is_allclose : callable, default=`np.allclose`
        The numerical convergence criterion.

    Returns
    -------
    n_iters : int
        The total number of Bellman optimality operator applications it took to
         reach two consecutive value functions that are within the specified tolerance.

    val : array-like of float of shape (n_states,)
        The numerical fixed point of the Bellman optimality operator
        :math:`T_\ast v \approx v`.
    """
    assert 0 <= gam < 1

    # kickstart the VI with `v_1 = T_\ast v_0` (fixed point iters)
    val_ = val = bellman(ker, rew, 0.0 if val is None else val, gam=gam)

    # use the while-loop to get the FP of T_\ast
    n_iters, is_first = 1, True

    # stop if close within relative tolerance, but not on the first iteration
    while is_first or not is_allclose(val_, val):
        # apply the bellman operator to the current v-fun
        val, val_ = bellman(ker, rew, val, gam=gam), val
        n_iters, is_first = n_iters + 1, False

    return n_iters, val


def qfun_vi_inf(
    ker: ndarray, rew: ndarray, val: ndarray, *, gam: float
) -> tuple[int, ndarray]:
    """optimal q-value of the infinite horizon control through value iterations."""

    # fp iterations complexity proportional to :math:`\frac1{1 - \gamma}`
    n_iter, val = vi_inf(ker, rew, val, gam=gam)
    return n_iter + 1, qvalue(ker, rew, val, gam=gam)


def aug_qfun_vi_inf(
    ker: ndarray, rew: ndarray, lam: ndarray, *, gam: float
) -> tuple[int, ndarray]:
    r"""The optimal q-value for the one reward-augmented binary-action problem.

    Notes
    -----
    The following computes :math:`V^H(s)` for every :math:`s \in S` via the
    lookahead recurrence with constant :math:`\lambda \in \mathbb{R}` action-
    dependent augmentation

    .. math::

        V^{h+1} = \max_u T_u V^h - \lambda 1_{u \neq 0} \,, h < H

    where :math:`\gamma \in [0, 1]` is the discount factor.
    """
    # `ker` and `rew` must broadcast to each other
    ker, rew = np.broadcast_arrays(ker, rew)  # (..., A, S, X)

    *batch, n_actions, n_states, n_states_ = ker.shape
    assert n_states == n_states_, ker.shape

    # augment the "active" action reward with the action-1 break even cost `lam`
    # which must have shape `(..., S)` and then get the optimal q-value
    lam = np.expand_dims(lam, (-1, -3))  # (..., S) -> (..., 1, S, 1)
    return qfun_vi_inf(ker, rew[..., 1:, :, :] - lam, 0.0, gam=gam)

    r"""The optimal q-value of the augmented process at each initial state
    and the static lambda, corresponding to that initial state.

    Notes
    -----
    For the given process :math:`k` and the vector :math:`(\lambda_x)_{x \in S}`
    this procedure computes

    .. math::

        Q_k^H(a, \cdot; \lambda_x)
            = (\tilde{T}^{(k)}_{a\mid \lambda_x})^{H-1} V_k^0[\cdot]

    with

    .. math::
        \tilde{T}^{(k)}_{a \mid \lambda} J = T^{(k)}_a J - \lamdba^\top A_k a

    where, __IMPORTANTLY__, :math:`\lambda` is __CONSTANT__ wrt intermediate state,
    action, and horizon!

    The returned value of the "diagonal" :math:`Q_k^H(a, x, \lambda_x)` for
    :math:`x \in S`.
    """


def whittle_vi_inf_bisect(
    ker: ndarray, rew: ndarray, *, gam: float, is_allclose: Callable = is_allclose
) -> tuple[int, ndarray]:
    r"""Compute per-state Whittle index using the bisection method.

    Notes
    -----
    The following procedure computes the Whittle index :math:`\lambda_s` by doing
    a bisection search, [1]_ p. 62. For each initial :math:`s \in S`, the Whittle
    index :math:`\lambda_{ks}` of the process (arm) :math:`k`, solves

    .. math ::

        \min_{\lambda \geq 0} V_k^H(s)

    where :math:`V_k^h` is defined through the functional backup recurrence

    .. math::

        V_k^{h+1}
            = \max \bigl\{ T^{(k)}_1 V_k^h - \lambda, T^{(k)}_0 V_k^h \bigr\}
            \,, h < H
        \,,

    with :math:`T^{(k)}_a` is the Bellman operator of the :math:`k`-th process, and
    :math:`H` is the planning horizon. In this implementation :math:`H = \infty`,
    and we use Value Iterations to compute the numeric fixed point of the process's
    Bellman Optimality operator.

    The fact that the :math:`V_k^h` recurrence involves the maximum of TWO alternatives,
    because of the binary action space, enables one to find the :math:`\lambda_{ks}`
    by solving for an equilibrium action price, such that

    .. math::

        T^{(k)}_1 V_k^{H-1}[s] - \lambda = T^{(k)}_0 V_k^{H-1}[s]
        \,.

    This break-even value represents the _budget efficiency_, i.e. how much better
    the long-term outcomes of a process become if it were acted upon with action
    :math:`a_k = 1` now at state :math:`s`.

    Monotonicity of the Bellman operator with respect to rewards implies that we can
    compute the individual budget efficiency for each process and then pick the top
    ones, see [1]_ p. 63 and appendix A.

    Although, this approximation [1]_ to restless bandit problem assumes :math:`\lambda`
    to be constant throughout the planning horizon :math:`h < H` and independent of
    the intermediate states :math:`x \in S`, the equilibrium __STILL__ depends on
    the initial state :math:`s \in S`, at which we compute :math:`V_k^H`.

    References
    ----------
    .. [1] Hawkins, J. T. (2003). "A Langrangian decomposition approach to weakly
       coupled dynamic optimization problems and its applications" (Doctoral
       dissertation, Massachusetts Institute of Technology).
       https://dspace.mit.edu/handle/1721.1/29599
    """

    ker, rew = np.broadcast_arrays(ker, rew)
    *batch, n_actions, n_states, n_states_ = ker.shape
    assert n_actions == 2

    # prepate the lower/upper- bounds for the bisection search
    # setup the bisection search for equilibrium lambdas for each initial state
    # XXX bounds is (lb, ub), each array of float of shape (..., n_states,)
    lb = np.broadcast_to(rew.min(), (*batch, n_states))
    ub = np.broadcast_to(rew.max(), (*batch, n_states))

    def bisect_(c, /, lb, x, ub):
        return np.where(c, x, lb), np.where(c, ub, x)
        # (x[j], ub[j]) if c[j] else (lb[j], x[j])

    # loop until all `lb` and `ub` are close
    n_total_iters = 0
    while not is_allclose(lb, ub):
        # bisect the current interval
        lam = (lb + ub) / 2

        # get the optimal q-function for the processes with cost-adjusted rewards
        n_iters, qval_kas = aug_qfun_vi_inf(ker, rew, lam, gam=gam)  # (..., A, S)
        q0_ks, q1_ks = qval_kas[..., 0, :], qval_kas[..., 1, :]

        # choose the half-interval based on q0 and q1: if q1 is better than q0,
        #  then the `a=1` cost `lam` is too low, and must be increased
        lb, ub = bisect_(q0_ks <= q1_ks, lb, lam, ub)
        n_total_iters += n_iters

    # recompute the final v-function at the mid-point lambda
    lam = (lb + ub) / 2
    n_iters, qval_kas = aug_qfun_vi_inf(ker, rew, lam, gam=gam)
    return n_total_iters + n_iters, (lam, np.max(qval_kas, -2))


class Whittle(BasePolicy):
    """Whittle index policy based on a known MDP model."""

    n_actions: int = 2  # binary
    gamma: float

    # attributes
    whittle_ks_: ndarray[float]  # (N, S)
    value_ks_: ndarray[float]  # (N, S)

    def __repr__(self) -> str:
        return type(self).__name__ + f"(gamma={self.gamma})"

    def __init__(
        self,
        n_max_steps: int | None,
        budget: int,
        n_states: int,
        /,
        gamma: float,
        *,
        random: Generator = None,
    ) -> None:
        super().__init__(n_max_steps, budget, n_states)

        assert isinstance(gamma, float) and 0 <= gamma < 1
        self.gamma = gamma

    def sneak_peek(self, env, /) -> None:
        """Brake open the black box and rummage in it for unfair advantage."""

        # ideally, whatever we get a hold of here should be estimated from some
        #  sample collected burn-in period by standard means
        if not isinstance(env, MDP):
            raise NotImplementedError(type(env))

        # precompute the whittle index for all arms and all states
        (self.n_whittle_iters_, (self.whittle_ks_, self.value_ks_)) = (
            whittle_vi_inf_bisect(env.kernels, env.rewards, gam=self.gamma)
        )

    # uninitialized_decide_impl defaults to `randomsubset(.budget, obs.shape[1])`

    def decide_impl(self, random: Generator, /, obs):
        """Decide the action to play to each arm at the provided observed state."""
        # `obs` is (batch, n_arms) and, if initialized, `n_arms == .n_arms_in_`
        # XXX usually `batch == 1`, i.e. single-element batch of observations.
        idx = np.broadcast_to(np.arange(self.n_arms_in_)[np.newaxis], obs.shape)

        # pick the precomputed whittle index at the current observed state of each arm
        lam_bk = self.whittle_ks_[idx, obs]

        # lexsort: first order the arms (in each batch item) by increasing independent
        #  random draws from `U[01, 1]`, then stably sort by the whittle index in
        #  ascending order. Sampels from the uniform automatically randomly resolve
        #  ties in the whittle index.
        order = np.lexsort((random.random(lam_bk.shape), lam_bk), axis=-1)

        # get the binary interaction mask (integers)
        # XXX we don't care if we are under budget
        subsets = np.zeros(order.shape, int)
        np.put_along_axis(subsets, order[..., -self.budget :], 1, axis=-1)

        return subsets
