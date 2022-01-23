# Greg Attra
# 01/22/2022

import numpy as np
from four_room.simulate import Simulation


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
