from typing import Callable

import numpy as np
import cellpylib as cpl

from dynascale.abstractions import Challenge

RNG = np.random.default_rng()


class CAChallenge(Challenge):
    def __init__(self, latent_dim, embed_dim, C_range=(-10, 10)):
        self.rule = RNG.integers(0, 255)  # TODO: make better RNG rule generator; investigate (https://cellpylib.org/reference.html?highlight=binaryrule#module-cellpylib.rule_tables)
        super().__init__(latent_dim, embed_dim)

    def _make_data(self, timesteps: int, n: int = None, init_conds: np.ndarray = None, control=None, in_dist: bool = True) -> np.ndarray:
        data = []
        init_conds = init_conds or RNG.integers(0, 1, endpoint=True, size=(n, self.embed_dim))
        control = np.zeros((timesteps, self.embed_dim)) if control is None else control
        for x0 in init_conds:
            cellular_automata = np.clip([x0 + control[0]], 0, 1).astype(np.int32)
            for t in range(1, timesteps):
                cellular_automata = cpl.evolve(cellular_automata, timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, 30), r=self.latent_dim)
                cellular_automata[-1] = np.clip(cellular_automata[-1] + control[t], 0, 1).astype(np.int32)
            data.append(cellular_automata)
        data = np.array(data)
        return data

    def _calc_error(self, x, y):
        """
        Return the hamming distance
        # TODO: look up pytorch/tensorflow documentation and copy/paste and do superclass inheritance
        :param x:
        :param y:
        :return:
        """
        return np.count_nonzero(x == y) / self.latent_dim
