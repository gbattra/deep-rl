# Greg Attra
# 01/23/2022

# Function for building the four room map

import numpy as np

from typing import Tuple
from lib.four_room.simulate import MapCell


def map_instantiate(map_size: Tuple[int, int], map_goal: Tuple[int, int]):
    mapp = np.zeros(map_size)
    walls = [
        ([0, 5], [5, 5]),
        ([5, 0], [5, 10]),
        ([5, 4], [10, 4])
    ]
    doors = [
        (1, 5), (5, 1), (5, 8), (8, 4)
    ]
    for start, end in walls:
        start_x, start_y = start
        end_x, end_y = end
        dx = end_x - start_x
        dy = end_y - start_y
        y = start_y
        for _ in range(dy + 1):
            x = start_x
            for _ in range(dx + 1):
                mapp[y, x] = MapCell.WALL.value
                x += 1
            y += 1

    for door in doors:
        x, y = door
        mapp[y, x] = MapCell.EMPTY.value

    goal_x, goal_y = map_goal
    mapp[goal_y, goal_x] = MapCell.GOAL.value

    return mapp
    