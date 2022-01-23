# Greg Attra
# 01/22/2022

import numpy as np
from lib.four_room.simulate import MapCell, Simulation
from lib.four_room.map import map_instantiate


def test_domain_slip():
    """
    Test that the domain slip functionality produces
    a set of chosen slips matching the pdf of all slip
    types
    """
    sim = Simulation(np.zeros((10, 10)))
    results = np.zeros(3)
    for i in range(1000):
        slip = sim._generate_slip()
        results[slip.value] += 1
    
    results /= 1000

    assert abs(np.sum(np.subtract(results, np.array((.8, .1, .1))))) <= .1


def test_invalid_state():
    mapp = map_instantiate((11, 11), (10, 10))
    mapp[4, 4] = MapCell.WALL.value
    mapp[0, 0] = MapCell.EMPTY.value
    sim = Simulation(mapp)
    assert not sim._valid_state((4, 4)), \
        'Invalid state should return false'
    assert sim._valid_state((0, 0))
