"""

"""
import cellpylib as cpl
import numpy as np
from joblib import Parallel, delayed

from ..abstractions import AbstractSystem
from ..utils.seeding import temp_numpy_seed, temp_random_seed


class CASystem(AbstractSystem):
    def __init__(self, latent_dim = 2, embed_dim = 64, in_dist_p=0.25, out_dist_p=0.75, mutation_p=0.00, seed=None):
        super().__init__(latent_dim, embed_dim, seed=seed)
        self._rng = np.random.default_rng(seed=seed)
        self._rule_table = self._get_rule_table()
        self._in_dist_p = in_dist_p
        self._out_dist_p = out_dist_p
        self._mutation_p = mutation_p

    def _get_rule_table(self):
        # NOTE: cpl.random_rule_table uses np.random and random so we monkeypatch the global numpy seed to be the system seed
        lambda_val = self._rng.uniform()
        with temp_random_seed(self._seed):
            with temp_numpy_seed(self._seed):
                rule_table, _, _ = cpl.random_rule_table(lambda_val=lambda_val, k=2, r=self.latent_dim,
                                                         strong_quiescence=True,
                                                         isotropic=True)
        return rule_table

    @AbstractSystem.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self._rule_table = self._get_rule_table()

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        if in_dist:
            return self._rng.binomial(1, self._in_dist_p, size=(n, self.embed_dim))
        else:
            return self._rng.binomial(1, self._out_dist_p, size=(n, self.embed_dim))

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        def get_trajectory(x0, u):
            cellular_automata = np.clip([x0 + u[0]], 0, 1).astype(np.int32)
            for t in range(1, timesteps):
                cellular_automata = cpl.evolve(cellular_automata,
                                               timesteps=2,
                                               apply_rule=lambda n, c, t: cpl.table_rule(n, self._rule_table),
                                               r=self.latent_dim)
                cellular_automata[-1] = np.clip(cellular_automata[-1] + u[t], 0, 1).astype(np.int32)
                if noisy:
                    mask = self._rng.binomial(1, self._mutation_p, size=(self.embed_dim,)).astype(bool)
                    cellular_automata[-1][mask] = (~cellular_automata[-1][mask].astype(bool)).astype(np.int32)
            return cellular_automata

        data = Parallel(n_jobs=4)(delayed(get_trajectory)(x0, u) for x0, u in zip(init_conds, control))
        data = np.array(data)
        return data

    def calc_error(self, x, y):
        # averaged across all samples and all predicted timesteps
        return 1 - (np.count_nonzero(x == y) / self.embed_dim) / len(y) / len(y[1])

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.sum(control, axis=(1, 2))
