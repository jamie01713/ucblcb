# why not use pydantic and manage experiments via json?
import json

from argparse import ArgumentParser

from typing import Any
from collections.abc import Callable


def make(p: ArgumentParser, /, *args: str) -> ArgumentParser:
    # Ideally we would like to inspect the keyword signature of a function
    #  and populate the argparser with correct types and defaults
    for arg in args:
        allowed_args[arg](p)  # use defaults atm

    return p


allowed_args: dict[str, Callable[[ArgumentParser], None]] = {}


def allow(fn):
    assert callable(fn)
    allowed_args[fn.__name__] = fn
    return fn


@allow
def path(p, /, default: str = "./") -> None:
    p.add_argument(
        "--path",
        type=str,
        default=default,
        help="The folder to store the resulting pickle and the figures",
    )


@allow
def prefix(p, /, default: str = "") -> None:
    p.add_argument(
        "--prefix",
        required=False,
        type=str,
        default=default,
        help="extra prefix to add to the filenames",
    )


@allow
def n_arms(p, /, default: int = 100) -> None:
    p.add_argument(
        "--n_arms",
        "-N",
        type=int,
        default=default,
        help="The number or arms in the binary MDP environment",
    )


@allow
def n_budget(p, /, default: int = 20) -> None:
    p.add_argument(
        "--n_budget",
        "-B",
        type=int,
        default=default,
        help="The number of arms the policy is allowed to pull at each step",
    )


@allow
def n_steps_per_episode(p, /, default: int = 20) -> None:
    p.add_argument(
        "--n_steps_per_episode",
        "-H",
        type=int,
        default=default,
        help="The maximal number of steps in one episodes rollout",
    )


@allow
def n_steps_per_replication(p, /, default: int = 20) -> None:
    p.add_argument(
        "--n_steps_per_replication",
        "-T",
        type=int,
        default=default,
        help="The total number of steps in rollout",
    )


@allow
def n_episodes_per_experiment(p, /, default: int = 500) -> None:
    p.add_argument(
        "--n_episodes_per_experiment",
        "-T",
        type=int,
        default=default,
        help="The number of episodes to play in one experiment replication",
    )


@allow
def n_replications_per_experiment(p, /, default: int = 500) -> None:
    p.add_argument(
        "--n_replications_per_experiment",
        "-R",
        type=int,
        default=default,
        help="The number of replications one experiment.",
    )


@allow
def n_experiments(p, /, default: int = 30) -> None:
    p.add_argument(
        "--n_experiments",
        "-E",
        type=int,
        default=default,
        help="The total number of independent replications to run",
    )


@allow
def n_population(p, /, default: int = 100) -> None:
    p.add_argument(
        "--n_population",
        "-P",
        type=int,
        default=default,
        help="The size of pool of MDP arms from which environment are sampled",
    )


@allow
def no_good_to_act(p, /, default: bool = False) -> None:
    p.add_argument(
        "--no_good_to_act",
        required=False,
        action="store_true",
        help="enforce good-to-act peroperty of MDPs",
    )


@allow
def no_good_origin(p, /, default: bool = False) -> None:
    p.add_argument(
        "--no_good_origin",
        required=False,
        action="store_true",
        help="enforce good-origin peroperty of MDPs",
    )


@allow
def avg_over_experiments(p, /, default: bool = False) -> None:
    p.add_argument(
        "--avg_over_experiments",
        required=False,
        action="store_true",
        help="Average over the experiment re-runs as well",
    )


@allow
def noise(p, /, default: float = 0.0) -> None:
    p.add_argument(
        "--noise",
        required=False,
        type=float,
        default=default,
        help="Independent gaussian noise added to the reward",
    )


@allow
def source(p, /, default: str = "") -> None:
    p.add_argument(
        "--source",
        type=str,
        default=default,
        help="An npz-file to load transitions and rewards from "
        "(leave empty for random pool).",
    )


@allow
def override(p, /, default: Any = None) -> None:
    p.register("type", "json", json.loads)
    p.add_argument(
        "--override",
        required=False,
        type="json",
        default=json.dumps(default),
    )


@allow
def entropy(p, /, default: int = 156594300754618296129554326137117204672) -> None:
    # https://www.random.org/cgi-bin/randbyte?nbytes=16&format=h
    p.register("type", "hexadecimal", lambda x: int(x, 16) if x else None)
    p.add_argument(
        "--entropy",
        required=False,
        type="hexadecimal",
        default=f"{default:032X}",
        help="128-bit seed for the experiment (leave empty to use system entropy)",
    )
