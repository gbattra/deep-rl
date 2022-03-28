# Greg Attra
# 03/25/22

'''
Feature extractors
'''

import numpy as np

from typing import Callable, List, Tuple

from lib.function_approximation.four_rooms import FourRooms


def get_feature_extractor(
        N: int,
        extract: Callable,
        aggregator: Callable) -> Callable:
    return lambda s: extract(aggregator(s), N)


def one_hot_encode(
        s: int,
        feat_size: int) -> np.ndarray:
    enc = np.zeros(feat_size)
    enc[s] = 1
    return enc


def xy_features(
        s: int,
        x_dim: int) -> np.ndarray:
    y = s // x_dim
    x = s % x_dim
    return np.array([y, x, 1.])


def xy_poly_features(
        s: int,
        x_dim: int) -> np.ndarray:
    y = s // x_dim
    x = s % x_dim

    return np.array([
        y,
        x,
        x*y,
        y*y,
        x*x,
        1.
    ])


def all_features(
        s: int,
        arena: np.ndarray,
        doors: List[Tuple[int, int]],
        goal: Tuple[int, int]):
    x_dim = arena.shape[1]
    y = (s // x_dim)
    x = (s % x_dim)
    w_up = arena[y-1, x] == FourRooms.WALL if y-1 >= 0 else 1
    w_down = arena[y+1, x] == FourRooms.WALL if y+1 < arena.shape[0] else 1
    w_left = arena[y, x-1] == FourRooms.WALL if x-1 <= 0 else 1
    w_right = arena[y, x+1] == FourRooms.WALL if x+1 < arena.shape[1] else 1
    d_up = (y-1, x) in doors
    d_down = (y+1, x) in doors
    d_left = (y, x-1) in doors
    d_right = (y, x+1) in doors
    active_doors = np.zeros(4)
    in_door = np.zeros(4)
    if d_up or d_down or d_left or d_right:
        for i, door in enumerate(doors):
            if (
                (y-1, x) == door or (y+1, x) == door \
                or (y, x+1) == door or (y, x-1) == door
            ):
                active_doors[i] = 1 if door == (y, x) else 0
            if door == (y, x):
                in_door[i] = 1
    in_room = np.zeros(4)
    in_room[0] = 1 if y > 5 and x < 5 else 0
    in_room[1] = 1 if y < 5 and x < 5 else 0
    in_room[2] = 1 if y > 6 and x > 5 else 0
    in_room[3] = 1 if y < 6 and x > 5 else 0

    g_up = (y-1, x) == goal
    g_right = (y, x+1) == goal
    feats = [
        y/x_dim,
        x/x_dim,
        (goal[0] - y)/x_dim,
        (goal[1] - x)/x_dim,
        int(g_up),
        int(g_right),
        int(w_up),
        int(w_down),
        int(w_left),
        int(w_right),
        int(d_up),
        int(d_down),
        int(d_left),
        int(d_right),
        int(w_up),
        int(w_down),
        int(w_left),
        int(w_right),
        int(d_up),
        int(d_down),
        int(d_left),
        int(d_right),
        active_doors[0],
        active_doors[1],
        active_doors[2],
        active_doors[3],
        in_door[0],
        in_door[1],
        in_door[2],
        in_door[3],
        in_room[0],
        in_room[1],
        in_room[2],
        in_room[3],
        1.
    ]
    return np.array(feats)
