#!/usr/bin/env python

# Greg Attra
# 01/22/2022

# Driver class for manually controlling an agent in the four room problem


from lib.four_room.agent import Agent
from lib.four_room.policy import ManualPolicy
from lib.four_room.simulate import Action, MapCell, Simulation
from lib.four_room.map import map_instantiate
from lib.four_room.visualize import visualize_scenario


def main():
    mapp = map_instantiate((11, 11), (10, 10))
    sim = Simulation(mapp)
    init_state = sim.reset()
    policy = ManualPolicy()
    agent = Agent(init_state, policy, sim)

    print(len(Action))
    while True:
        visualize_scenario(agent, sim)
        transition = agent.step()
        agent.update(transition)
        x, y = agent.state
        if MapCell(mapp[y, x]) == MapCell.GOAL:
            print("Goal reached! +1 Reward!")
        if x == 0 and y == 0:
            print("Action taken from goal. Resetting to state [x: 0, y: 0]")


if __name__ == '__main__':
    main()
