
import cellpylib as cpl
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from dynascale.abstractions import AbstractSystem

RNG = np.random.default_rng()


class CASystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim, in_dist_p=0.25, out_dist_p=0.75, mutation_p=0.01):
        super().__init__(latent_dim, embed_dim)
        lambda_val = RNG.uniform()
        self.rule_table, _, _ = cpl.random_rule_table(lambda_val=lambda_val, k=2, r=self.latent_dim,
                                                      strong_quiescence=True,
                                                      isotropic=True)
        self._in_dist_p = in_dist_p
        self._out_dist_p = out_dist_p
        self._mutation_p = mutation_p

    @AbstractSystem.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        lambda_val = RNG.uniform()
        self.rule_table, _, _ = cpl.random_rule_table(lambda_val=lambda_val, k=2, r=self.latent_dim,
                                                      strong_quiescence=True,
                                                      isotropic=True)

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        if in_dist:
            return RNG.binomial(1, self._in_dist_p, size=(n, self.embed_dim))
        else:
            return RNG.binomial(1, self._out_dist_p, size=(n, self.embed_dim))

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        def get_trajectory(x0, u):
            cellular_automata = np.clip([x0 + u[0]], 0, 1).astype(np.int32)
            for t in range(1, timesteps):
                cellular_automata = cpl.evolve(cellular_automata,
                                               timesteps=2,
                                               apply_rule=lambda n, c, t: cpl.table_rule(n, self.rule_table),
                                               r=self.latent_dim)
                cellular_automata[-1] = np.clip(cellular_automata[-1] + u[t], 0, 1).astype(np.int32)
                if noisy:
                    mask = RNG.binomial(1, self._mutation_p, size=(self.embed_dim,)).astype(bool)
                    cellular_automata[-1][mask] = (~cellular_automata[-1][mask].astype(bool)).astype(np.int32)
            return cellular_automata

        data = Parallel(n_jobs=4)(delayed(get_trajectory)(x0, u) for x0, u in zip(init_conds, control))
        data = np.array(data)
        return data

    def calc_loss(self, x, y):
        # averaged across all samples and all predicted timesteps
        return (np.count_nonzero(x == y) / self.embed_dim) / len(y) / len(y[1])

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.sum(control, axis=(1, 2))