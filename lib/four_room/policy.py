# Greg Attra
# 01/22/2021

# A policy which maps from S -> A, typically learned through
# experience

from lib.four_room.simulate import Action, State

class Policy:
    def map(self, state: State) -> Action:
        """
        Mao from a state to an action (which yields highest expected
        reward)
        """
        pass


class ManualPolicy():
    prompt: str = '                            \
        Choose an action for state {state}  \n \
            - 0: UP                         \n \
            - 1: DOWN                       \n \
            - 2: LEFT                       \n \
            - 3: RIGHT                      \n \
    '

    def map(self, state: State) -> Action:
        """
        Get action from user input
        """
        x, y = state
        action_idx = input(self.prompt.format(state=f'[x: {x}, y: {y}]'))
        action_idx = int(action_idx)
        print(action_idx)

        assert action_idx < len(Action) and action_idx >= 0, \
            f'Invalid action value. Must be between 0 and {len(Action)}\n'
            
        action = Action(action_idx)
        print(action)
        
        return action
