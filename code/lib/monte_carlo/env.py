from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('FourRooms-v0')

    Note: the max_episode_steps option controls the time limit of the environment.
    You can remove the argument to make FourRooms run without a timeout.
    """
    register(id="FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=459)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class Slip(IntEnum):
    NONE = 0
    NEGATIVE = 1
    POSITIVE = 2


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
    """
    Helper function to map action to changes in x and y coordinates
    Args:
        action (Action): taken action
    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


class FourRoomsEnv(Env):
    """Four Rooms gym environment.

    This is a minimal example of how to create a custom gym environment. By conforming to the Gym API, you can use the same `generate_episode()` function for both Blackjack and Four Rooms envs.
    """

    def __init__(self, goal_pos=(10, 10), max_timesteps=459) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )
        self.max_timesteps = max_timesteps
        self.t = 0

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos
        self.t = 0
        return self.agent_pos

    def _slip_action(self, action: Action, slip: Slip):
        """
        Apply a chosen slip type to an action
        """

        if slip == Slip.NONE:
            return Action(action)

        if slip == Slip.NEGATIVE:
            action -= 1
            if action < 0:
                action = len(Action) - 1
        if slip == Slip.POSITIVE:
            action += 1
            if action >= len(Action):
                action = 0

        action_taken = Action(action)
        return action_taken

    def _generate_slip(self) -> Slip:
        """
        Generate a slip type via probability distribution function
        over all slip types
        """
        # action pdf: .8  .1  .1
        action_cdf = [.8, .9, 1.]
        x = self.np_random.random()
        a = 0
        while x > action_cdf[a]:
            a += 1
        slip = Slip(a)
        return slip

    def _domain_slip(self, action: Action) -> Action:
        """
        Apply domain noise to an action. Choose a type of
        slip from a probability distribution function of
        all types of slips.
        """
        slip = self._generate_slip()
        action_taken = self._slip_action(action, slip)
        return action_taken

    def _is_oob(self, pos: Tuple[int, int]) -> bool:
        """
        Check if pos is out of bounds
        """
        if pos[0] < 0 or pos[1] < 0:
            return True
        if pos[0] > self.observation_space[0].n \
            or pos[1] > self.observation_space[1].n:
            return True
        return False

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """
        self.t += 1

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        elif self.t >= self.max_timesteps:
            done = True
            reward = 0.0
        else:
            done = False
            reward = 0.0

        # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0
        action_taken = self._domain_slip(action)

        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0
        dxdy = actions_to_dxdy(action_taken)
        next_pos = (self.agent_pos[0] + dxdy[0], self.agent_pos[1] + dxdy[1])

        # TODO check if next position is feasible
        # If the next position is a wall or out of bounds, stay at current position
        # Set self.agent_pos
        if next_pos in self.walls or self._is_oob(next_pos):
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}
