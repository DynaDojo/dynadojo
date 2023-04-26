import numpy as np


def mean_squared_error(x, y):
    error = np.linalg.norm(x - y, axis=0)
    return np.mean(error ** 2)