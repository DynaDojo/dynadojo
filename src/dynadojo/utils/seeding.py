import contextlib
import random

import numpy as np


@contextlib.contextmanager
def temp_numpy_seed(seed):
    """
    Function from Paul Panzer:
    * https://stackoverflow.com/a/49557127/15001799
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def temp_random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)