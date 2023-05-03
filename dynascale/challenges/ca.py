from typing import Callable

import numpy as np
import cellpylib as cpl

from dynascale.abstractions import Challenge

RNG = np.random.default_rng()


class CAChallenge(Challenge):
    def __init__(self, latent_dim, embed_dim, C_range=(-10, 10)):
        self.rule = RNG.integers(0, 255)  # TODO: make better RNG rule generator; investigate (https://cellpylib.org/reference.html?highlight=binaryrule#module-cellpylib.rule_tables)
        super().__init__(latent_dim, embed_dim)

    def _make_init_conds(self, n: int, in_dist=True) -> (np.ndarray, np.ndarray):
        init_conds = []
        for _ in range(n):
            cellular_automaton = cpl.init_random(self.embed_dim).squeeze()
            init_conds.append(cellular_automaton)
        init_conds = np.array(init_conds)
        return init_conds

    def _make_data(self, init_conds: np.ndarray, timesteps: int, control=None) -> (np.ndarray, np.ndarray):
        data = []
        control = np.zeros((timesteps, self.embed_dim)) if control is None else control
        for x0 in init_conds:
            cellular_automata = np.array([x0])
            for t in range(timesteps - 1):  # TODO: check if we should do this? Do we ignore control[0]?
                cellular_automata = cpl.evolve(cellular_automata, timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, 30), r=self.latent_dim)
                cellular_automata[t] = np.clip(cellular_automata[0, t] + control[t], 0, 1).astype(np.int32)
            data.append(cellular_automata)
        data = np.array(data)
        return data, data[:, -1]

    def _calc_error(self, x, y):
        return np.count_nonzero(x == y) / self.latent_dim
