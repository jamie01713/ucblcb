import numpy as np
from numpy import ndarray

from scipy.special import softmax
from functools import partial  # noqa: F401
from numpy.random import default_rng, SeedSequence, Generator

from typing import TypeVar
from typing_extensions import Self


Observation = State = TypeVar("State", bound=ndarray[int])
Action = TypeVar("Action", bound=ndarray[int])
Reward = TypeVar("Reward", bound=float) | ndarray[float]


class BatchMDPSimulator:
    """Not a gynmasium env"""

    kernels: ndarray[float]  # (N, S, A, X) markov transition kernel
    rewards: ndarray[float]  # (N, S, A, X) MDP's reward (see `.__init__`)
    designation: ndarray[int]  # (P,) population member to MDP appointment P -> N

    initial: State  # (P,) the current hidden state of each pop member's mdp instance
    seed: None | int

    # the state of the simulator
    random_: Generator
    state_: State  # (P,) the current hidden state of each pop member's mdp instance
    observation_: Observation  # the recent observation vector
    n_steps_: int

    def __init__(
        self,
        /,
        kernels,
        rewards,
        *,
        initial: State = None,
        designation: ndarray[int] = None,
    ) -> None:
        # make sure the kernel and rewards broadcast and have shape (N, S, A, X)
        # XXX rewards's shape uniquely determines the expected reward function
        # * `(S, A, X)` -- `r(s, a, x)`
        # * `(1, A, 1)` -- `r(a)`
        # * `(1, 1, X)` -- `r(x)`
        # basically, unit dims correspond to "independece" on the respective piece
        #  of the state-action-next_state transition.
        shape = np.broadcast_shapes(kernels.shape, (1, 1, 1, 1))
        kernels, rewards = np.broadcast_arrays(np.broadcast_to(kernels, shape), rewards)

        # check kernel consistency
        n_types, n_states, n_actions, n_states_ = kernels.shape
        assert n_states == n_states_, kernels.shape
        assert np.allclose(kernels.sum(-1), 1.0), "markov kernel is not a probability"
        assert np.all((0 <= kernels) & (kernels <= 1)), "probability must be in [0, 1]"

        # save the MDP specs
        self.kernels = kernels / np.sum(kernels, axis=-1, keepdims=True)
        self.rewards = rewards

        # deal with member types assignment
        if designation is not None:
            designation = np.asarray(designation, dtype=int, copy=False)
            assert np.all((0 <= designation) & (designation < n_types)), designation

        self.designation = designation

        # handle the initial state
        n_population = n_types if self.designation is None else len(self.designation)
        if initial is None:
            initial = np.asarray(initial, dtype=int, copy=False)
            assert np.all((0 <= initial) & (initial < n_states))
            assert len(initial) == n_population, initial.shape

        self.initial, self.seed = initial, None

        # init the runtime attrs
        self.n_steps_ = self.state_ = self.observation_ = None

        # draw a default seed
        self.random_ = None

    def reset_seed(self, seed) -> SeedSequence:
        # init the seed sequence and update the `.seed` if one was provided
        if isinstance(seed, SeedSequence):
            seedseq = seed

        else:
            seedseq = SeedSequence(self.seed if seed is np._NoValue else seed)
        self.seed = self.seed if seed is np._NoValue else seedseq.entropy
        return seedseq

    def reset(self, /, seed: None | int | SeedSequence = np._NoValue) -> Observation:
        self.random_ = default_rng(self.reset_seed(seed))

        # sample the initial state at random
        n_types, n_states, _, _ = self.kernels.shape
        n_population = n_types if self.designation is None else len(self.designation)
        if self.initial is None:
            init_state_ = self.random_.choice(n_states, size=n_population, p=None)

        else:
            assert len(self.initial) == n_population
            init_state_ = self.initial.copy()

        # the observation is the state (complete information MDP)
        # XXX `self.random_` is updated inplace!
        self.state_, self.n_steps_ = init_state_, 0
        return self.state_.copy(), {}

    def step(self, /, actions: Action) -> tuple[Observation, Reward, bool, dict]:
        # get the member to mdp designation
        index = self.designation
        if index is None:
            index = np.arange(len(self.kernels))

        # draw the next state and take the reward
        probas = self.kernels[index, self.state_, actions]
        next_state_ = self.random_.multinomial(1, probas, size=None).argmax(-1)
        reward_ = self.rewards[index, self.state_, actions, next_state_]

        # update the full state
        self.state_, self.n_steps_ = next_state_, self.n_steps_ + 1

        # return the next state and the reward feedback
        # XXX shouldn't it be a vector of bool?
        return self.state_.copy(), reward_.astype(float), False, {}

    def get_observation(self, /, state: State) -> Observation:
        return state.copy()

    # runtime props
    @property
    def n_population(self):
        return len(self.kernels if self.designation is None else self.designation)

    n_types = property(lambda self: self.kernels.shape[0])
    n_states = property(lambda self: self.kernels.shape[1])  # == .shape[3]
    n_actions = property(lambda self: self.kernels.shape[2])

    def __repr__(self) -> str:
        return f"MDP({self.n_population}, S={self.n_states}, A={self.n_actions})"


class BasePolicy:
    """The base policy interface."""

    def update(
        self,
        /,
        obs: Observation,
        act: Action,
        rew: Reward,
        new_obs: Observation,
    ) -> Self:
        return self

    def decide(self, /, obs: Observation) -> Action:
        raise NotImplementedError


class Random_policy(BasePolicy):
    def __init__(self, seed):
        self.random_ = default_rng(seed)

    def decide(self, obs):
        return self.random_.choice(2, size=len(obs))


def spawn_rmab(
    transitions,
    /,
    designation=None,
    initial=None,
    seed=None,
) -> BatchMDPSimulator:
    # ensure binary state and action spaces
    *_, n_states, n_actions, n_states_ = transitions.shape
    assert n_states == n_states_ == n_actions == 2

    # ensure acting is always good (good state is ONE)
    is_acting_good = np.all(transitions[:, :, 0, 1] <= transitions[:, :, 1, 1])
    assert True or is_acting_good, "acting should always be good"

    # ensure a good start state is always good
    is_state_good = np.all(transitions[:, 0, :, 1] <= transitions[:, 1, :, 1])
    assert True or is_state_good, "good start state should always be good"

    # build `r(s, a, x) = 1_{x=1}` for binary state and action spaces
    rewards = np.r_[0.0, 1.0].reshape(1, 1, n_states)  # see `BatchMDPSimulator`

    env = BatchMDPSimulator(
        transitions, rewards, designation=designation, initial=initial
    )
    env.reset_seed(seed)
    return env


def run_one_episode(
    pol: BasePolicy,
    /,
    env: BatchMDPSimulator,
    *,
    n_steps: int,
) -> tuple[BasePolicy, ndarray[float]]:
    """Play a single episode with the given policy"""
    history = []

    # reset the env and get its initial observation
    # XXX set the initial rewards to NaN, because `.reset` sort of implies that
    #  the next `.step` is the VERY first interactions with the MDP. Hence,
    #  there is simply no reward to be had because of not prior interaction
    (obs, _), rew_ = env.reset(), np.full(env.n_population, np.nan)  # (r_0, x_0)
    for step in range(n_steps):
        # `act` is `a_t` array of int of shape (P,) of int
        act = pol.decide(obs)  # x_t --pol-->> a_t
        obs_, rew_, _, _ = env.step(act)  # (x_t, a_t) --env-->> (r_{t+1}, x_{t+1})

        # update the policy with the observed transition
        # (x_t, a_t, r_{t+1}, x_{t+1}) --pol-->> pol'
        pol = pol.update(obs, act, rew_, obs_)

        # keep track of the total reward
        history.append(rew_)

    # return rewards history from the batch
    return pol, np.stack(history, axis=0)


def run_episodic(
    pol_spawn,
    env_spawn,
    /,
    initial,
    designation,
    *,
    n_steps: int,
    n_episodes: int,
    seed: int = None,
):
    # get the seeds
    sk, *seedseqs = SeedSequence(seed).spawn(1 + n_episodes)

    # the policy is reset at the start of the episodic interaction
    history = []
    pol = pol_spawn(seed=sk)
    for episode, sk in enumerate(seedseqs):
        # spawn a new env with new initial state (cohort designation comes from elsewhere)
        env = env_spawn(
            designation=designation[episode], initial=initial[episode], seed=sk
        )

        # interact with a new mdp env for `n_steps`
        _, rewards = run_one_episode(pol, env, n_steps=n_steps)
        history.append(np.sum(rewards, axis=-1))

    return pol, np.stack(history, axis=0)


random_ = default_rng(None)

n_types, n_states, n_actions, temperature = 7, 2, 2, 0.5
logits = random_.normal(size=(n_types, n_states, n_actions, n_states))
kernel = softmax(logits / temperature, axis=-1)

n_population, n_episodes, n_steps = 1000, 100, 100
designation = random_.integers(n_types, size=(n_episodes, n_population))
initial = random_.choice(n_states, size=(n_episodes, n_population), p=None)

out = run_episodic(
    Random_policy,
    partial(spawn_rmab, kernel),
    initial,
    designation,
    n_steps=n_steps,
    n_episodes=n_episodes,
    seed=None,
)


def UCWhittle_value() -> None:
    n_instances = n_episodes = n_steps = None
    init_states = cohort_designation = seed = ...
    kernels = rewards = ...
    policy = ...

    # init states are `n_epochs, n_episodes, n_states`
    # env.reset_all()
    total_rewards = np.zeros((n_instances, n_episodes, n_steps + 1), float)
    for instance in range(n_instances):
        # env.reset_instance()

        policy.reset()
        # spawn a new env with new initial state and cohort desgination
        env = BatchMDPSimulator(
            kernels,
            rewards,
            initial=init_states[instance, episode],
            designation=cohort_designation[instance],
        )
