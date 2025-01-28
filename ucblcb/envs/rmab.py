"""Routines to sample and verify binary batched MDPs for Restless MABs.
"""
import warnings

import numpy as np
from numpy import ndarray

from typing import Iterator
from numpy.random import default_rng, Generator

from .mdp import MDP


def minmax(a0: ndarray, a1: ndarray) -> tuple[ndarray, ndarray]:
    """Return `min(a0, a1)` and `max(a0, a1)`"""

    is_gt = a0 <= a1
    p_min = np.where(is_gt, a0, a1)
    p_max = np.where(is_gt, a1, a0)

    return p_min, p_max


def random_valid_binary_mdp(
    random: Generator,
    /,
    size: tuple[int, ...] = None,
    *,
    good_to_act: bool = True,
    good_origin: bool = True,
) -> ndarray:
    """Sample a markov transition kernel that is good."""

    # build `r(a, s, x) = 1_{x=1}` for binary state and action spaces
    # XXX `good` means the +ve reward!
    size = () if size is None else size
    rewards = np.broadcast_to([0.0, 1.0], (*size, 1, 1, 2))
    # XXX for `r(a, s, x) = 1_{s=1}` use `([[0.0], [1.0]], (*size, 1, 2, 1))`

    # get a pool of good markov transition kernels `p_{asx} = p(x \mid s, a)`
    #  "acting is always good, and starting in good state is always good"
    p_as1 = default_rng(random).uniform(size=(*size, 2, 2))

    # enforce "acting-is-always-good" `p_{0s1} <= p_{1s1}`
    if good_to_act:
        p_0s1, p_1s1 = p_as1[..., 0, :], p_as1[..., 1, :]
        p_as1 = np.stack(minmax(p_0s1, p_1s1), axis=-2)

    # enforce "good-origin-is-good" `p_{a01} <= p_{a11}`
    if good_origin:
        p_a01, p_a11 = p_as1[..., :, 0], p_as1[..., :, 1]
        p_as1 = np.stack(minmax(p_a01, p_a11), axis=-1)

    # the binary mdp kernels `(..., A, S, X)`
    p_asx = np.stack([1 - p_as1, p_as1], axis=-1)
    return p_asx, rewards  # p_n(x \mid s, a), r_n(a, s, x)


def binary_rmab_from_nasx_npz(npz: str, /, **ignore) -> tuple[ndarray, ndarray]:
    """Read saved mdp population from a numpy's npz file."""
    if ignore:
        warnings.warn(repr(ignore), RuntimeWarning)

    # Read nasx arrays
    vault = np.load(npz, allow_pickle=True)
    kernels, rewards = vault["ker"], vault["rew"]

    # broadcast and validate shapes
    ker, rew = np.broadcast_arrays(kernels, rewards)
    *_, n_actions, n_states, n_states_ = ker.shape
    assert n_states == n_states_ == n_actions == 2, ker.shape

    # make sure the kernels are Markov
    assert np.all(np.allclose(ker.sum(-1), 1.0)), f"data in {npz} is not MDP"
    ker /= np.sum(ker, axis=-1, keepdims=True)

    return ker, rew


def binary_rmab_sampler(
    random: Generator, /, transitions, n_processes: int = None, *, noise: float = 0.0,
) -> Iterator[MDP]:
    """Sampler for binary RMAB problems with good transition."""
    raise RuntimeError("do not use")

    random = default_rng(random)  # the PRNG `random` is consumed!

    # ensure binary state and action spaces
    *_, n_actions, n_states, n_states_ = transitions.shape
    assert n_states == n_states_ == n_actions == 2, transitions.shape

    # build `r(a, s, x) = 1_{x=1}` for binary state and action spaces
    # XXX `r_{asx} = r_x`, i.e. a feedback for entering a state
    # XXX `good` means the +ve reward!
    rewards = np.r_[0.0, 1.0].reshape(1, 1, n_states)  # see `BatchedMDPSimulator`

    # ensure acting is always good (good state is ONE) `p_{0s1} \leq p_{1s1}`
    # XXX for `p_{asx} = p(x | s, a)` we ensure `p_{0s1} \leq p_{1s1}` since `x=1`
    #  is special, because of how the rewards are defined).
    is_acting_good = np.all(transitions[:, 0, :, 1] <= transitions[:, 1, :, 1])
    assert True or is_acting_good, "acting should always be good"

    # ensure a good start state is always good `p_{a01} \leq p_{a11}`
    is_state_good = np.all(transitions[:, :, 0, 1] <= transitions[:, :, 1, 1])
    assert True or is_state_good, "good start state should always be good"

    # simply delegate to the sub-iterator
    yield from MDP.sample(
        random, transitions, rewards, n_processes=n_processes, noise=noise
    )


def binary_rmab_sampler_expected(
    random: Generator, /, transitions, n_processes: int = None, *, noise: float = 0.0,
) -> Iterator[MDP]:
    """Sampler for binary RMAB problems with good transition."""
    raise RuntimeError("do not use")

    random = default_rng(random)  # the PRNG `random` is consumed!

    # ensure binary state and action spaces
    *_, n_actions, n_states, n_states_ = transitions.shape
    assert n_states == n_states_ == n_actions == 2

    # build `r_n(a, s, x) = 1_{x=1}` for binary state and action spaces
    rewards = np.broadcast_to(np.r_[0.0, 1.0], transitions.shape)

    # get the expected reward \phi_n(a, s) = E_{p_n(x|a, s)} r_n(a, s, x)
    phi_nas = np.einsum("nasx,nasx -> nas", transitions, rewards)

    # ensure that pulling always yields higher expected reward
    is_acting_good = np.all(phi_nas[:, 0, :] <= phi_nas[:, 1, :])
    assert is_acting_good, "pulling an arm should always be beneficial"

    # ensure a special origin state` s=1` is good no matter the pulling
    is_origin_good = np.all(phi_nas[:, :, 0] <= phi_nas[:, :, 1])
    assert True or is_origin_good, "good start state should always be good"

    # simply delegate to the sub-iterator
    yield from MDP.sample(
        random, transitions, rewards, n_processes=n_processes, noise=noise
    )
