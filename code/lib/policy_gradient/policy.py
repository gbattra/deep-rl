# Greg Attra
# 04.09.22

'''
Functions for policies
'''

import numpy as np


def softmax_policy(x: np.ndarray, O: np.ndarray) -> np.ndarray:
    q_vals = x.dot(O)
    return np.exp(q_vals) / np.sum(np.exp(q_vals))
