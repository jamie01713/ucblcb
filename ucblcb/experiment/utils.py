"""Experiment seeding, ground truth peeker, and miscellaneous tools.
"""

import re
import warnings

from importlib import import_module

import numpy as np
from scipy import signal

from numpy import ndarray
from numpy.random import Generator, SeedSequence

from functools import wraps

from collections.abc import Callable

from ..envs.base import Env
from ..policies.base import BasePolicy


def sq_spawn(
    sq: SeedSequence | Generator, /, *shapes: tuple[int, ...], axis: int = 0
) -> ndarray[SeedSequence | Generator]:
    """Prepare seeds for the experiment."""

    # produce shaped seed sequence spawns
    arys = []
    for shape in shapes:
        # get an zero-data shape and spawn from the seed sequence
        # XXX see `np.broadcast_shapes(...)`
        x = np.empty(shape, dtype=np.dtype([]))
        sprout = np.reshape(sq.spawn(x.size), x.shape)

        arys.append(sprout)

    # broadcast and stack them over the specified dim
    return np.stack(np.broadcast_arrays(*arys), axis) if len(shapes) > 1 else arys[0]


def sneak_peek(
    pol: BasePolicy, /, method: str = "sneak_peek", *, honest: bool = True
) -> Callable[[Env], None]:
    """Break open the black box and let the policy rummage in it for unfair advantage.

    Notes
    -----
    The title is self-explanatory.

    Any policy that implements the method and/or makes use of it on an environment
    that it is played in should be ashamed of itself, condemned by its peers, and
    shunned by everybody.
    """

    peeker = getattr(pol, method, None)
    if not callable(peeker):
        return lambda _: None

    if not honest:
        return peeker

    @wraps(peeker)
    def peeker_with_alert(env: Env) -> None:
        message = f"policy {pol}({id(pol)}) took a peek into the black box"
        warnings.warn(message, RuntimeWarning)
        return peeker(env)

    return peeker_with_alert


def snapshot_git(path=None, *, diff: bool = True) -> dict:
    """Save some minimal version control info so that past results
    are easier to recover if something gets irreversibly forgotten.
    """

    try:
        from git import Repo, InvalidGitRepositoryError

        # open the repo
        repo = Repo(path, search_parent_directories=True)

        # get the head commit sha
        head = str(repo.head.commit)

        # get diff againts head
        diff = repo.git.diff(None) if diff else ""

    except (InvalidGitRepositoryError, ImportError):
        head, diff = "", ""

    return {"head": head, "diff": diff}


def from_qualname(spec: str | type) -> type:
    """Parse the specified qualname, import it and return the type."""
    if isinstance(spec, type):
        return spec

    # remove the text-wrapper resulting from `str(type(obj))`
    match = re.fullmatch(r"^<(?:class)\s+'(.*)'>$", spec)
    qualname = spec if match is None else match.group(1)

    # import from built-ins if no module was detected
    module, dot, name = qualname.rpartition(".")
    return getattr(import_module(module if dot else "builtins"), name)


def populate(mod, /, *bases) -> tuple[type]:
    """Enumerate all top-level classes that are derived from the given bases."""

    classes = []
    for x in dir(mod):
        obj = getattr(mod, x)
        # keep classes derived from any of hte bases, but not bases themselves
        if isinstance(obj, type) and issubclass(obj, bases):
            if obj not in bases:
                classes.append(obj)

    return classes


def ewmmean(x: ndarray, /, alpha: float = 0.7, axis: int = -1) -> ndarray:
    """Exponential smoothing of the values along the axis."""

    # numerator and denominator for rational transfer function of the filter
    # y[n] = alpha * y[n-1] + (1 - alpha) * x[n]
    b, a = np.r_[1.0 - alpha], np.r_[1.0, -alpha]
    if axis is None:
        x, axis = x.ravel(), 0

    # `lfilter_zi` returns the multiplier for the initial conditions
    zi = np.take(x, np.r_[0], axis=axis) * signal.lfilter_zi(b, a)

    # `lfilter` returns `y_n` satisfying
    #    a_0 y_n = \sum_{j \geq 0} b_j x_{n-j} - \sum_{k \geq 1} a_k y_{n - k}
    y, zf = signal.lfilter(b, a, x, axis=axis, zi=zi)
    return y


def expandingmean(x: ndarray, /, axis: int = -1) -> ndarray:
    """Expanding window average of the values along the axis."""

    if axis is None:
        return x.cumsum(None) / np.arange(1, 1 + x.size)

    # prepare the denominator
    axis = axis + np.ndim(x) if axis < 0 else axis
    size = np.arange(1, 1 + x.shape[axis])
    np.expand_dims(size, tuple(range(axis + 1, np.ndim(x))))

    # compute the cumulative sum and return the mean
    return np.cumsum(x, axis=axis) / size
