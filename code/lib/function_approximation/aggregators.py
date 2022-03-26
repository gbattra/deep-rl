# Greg Attra
# 03/25/22

'''
Aggregator methods
'''

from typing import Callable


def segment(s: int, x_dim: int, seg_size: int) -> int:
    y = s // x_dim
    x = s % x_dim
    y_group = y // seg_size
    g = (y_group * x_dim) + x
    return g

def zone(s: int, zone_size: int) -> int:
    pass
