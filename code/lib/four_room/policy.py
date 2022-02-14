# Greg Attra
# 01/22/2021

# A policy which maps from S -> A, typically learned through
# experience

from math import floor
from random import Random
from typing import Dict
from lib.four_room.simulate import Action, State

class Policy:
    def map(self, state: State) -> Action:
        """
        Mao from a state to an action (which yields highest expected
        reward)
        """
        pass


class ManualPolicy(Policy):
    prompt: str = '                            \
        Choose an action for state {state}  \n \
            - w: UP                         \n \
            - a: LEFT                       \n \
            - s: DOWN                       \n \
            - d: RIGHT                      \n \
    '

    action_key_idx_map: Dict[str, int] = {
        'w': 1,
        'a': 0,
        's': 3,
        'd': 2
    }

    def map(self, state: State) -> Action:
        """
        Get action from user input
        """
        x, y = state
        action_key = input(self.prompt.format(state=f'[x: {x}, y: {y}]'))
        action_idx = self.action_key_idx_map[action_key]

        assert action_idx < len(Action) and action_idx >= 0, \
            f'Invalid action value. Must be between 0 and {len(Action)}\n'
            
        action = Action(action_idx)
        return action


class RandomPolicy(Policy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.rand: Random = Random(seed)

    def map(self, state: State) -> Action:
        """
        Generate a random action
        """
        action_idx = floor(self.rand.random() * len(Action))
        return Action(action_idx)


# Same Action for N Steps Policy
class SANSPolicy(RandomPolicy):
    def __init__(self, n_steps: int) -> None:
        super().__init__()
        self.n_steps: int = n_steps
        self.countdown: int = 0
        self.curr_action: Action = None

    def map(self, _: State) -> Action:
        """
        Returns the same action regardless of state
        """
        if self.countdown == 0 or self.curr_action is None:
            self.curr_action = super().map((0, 0))  # get random action
            self.countdown = self.n_steps

        self.countdown -= 1
        return self.curr_action
