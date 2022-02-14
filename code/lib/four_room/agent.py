# Greg Attra
# 01/22/2022

# An agent which interacts in the four room simulation to learn a policy

from lib.four_room.policy import Policy
from lib.four_room.simulate import Simulation, State, Transition


class Agent:
    def __init__(
            self,
            init_state: State,
            policy: Policy,
            sim: Simulation) -> None:
        self.policy: Policy = policy
        self.sim: Simulation = sim
        self.state: State = init_state
        self.reward: int = 0

    def step(self) -> Transition:
        """
        Choose an action given the current state using a policy
        and run that action through the simulator to get the next
        state and reward
        """
        action = self.policy.map(self.state)
        transition = self.sim.simulate(self.state, action)

        return transition

    def update(self, transition: Transition) -> None:
        """
        Set the new state and add to the reward
        """
        self.state = transition.state
        self.reward += transition.reward
