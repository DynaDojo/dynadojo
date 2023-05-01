from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from dynascale.abstractions import Factory

RNG = np.random.default_rng()


class LDSFactory(Factory):
    def __init__(self, latent_dim, embed_dim, A_range=(-10, 10), B_range=(-10, 10), C_range=(-10, 10)):
        self._A_range = A_range
        self._B_range = B_range
        self._C_range = C_range
        self.A = RNG.uniform(*A_range, (latent_dim, latent_dim))
        self.B = RNG.uniform(*B_range, (latent_dim, embed_dim))
        self.C = RNG.uniform(*C_range, (latent_dim, embed_dim))
        super().__init__(latent_dim, embed_dim)

    @Factory.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self.B = RNG.uniform(*self._B_range,
                             (self.latent_dim, self.embed_dim))  # TODO: verify that embed dim is updated
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    @Factory.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self.A = RNG.uniform(*self._A_range, (self.latent_dim, self.latent_dim))
        self.B = RNG.uniform(*self._B_range, (self.latent_dim, self.embed_dim))
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    @staticmethod
    def _make_staircase_func(timesteps: int, control: np.ndarray) -> Callable:
        time = np.linspace(0, 1, endpoint=True, num=timesteps)

        def staircase_func(t):
            i = np.argmin(np.abs(t - time))
            return control[i]

        return staircase_func

    def _make_init_conds(self, n: int, in_dist=True, *args, **kwargs):
        bounds = (0, 10) if in_dist else (-10, 0)
        return RNG.uniform(*bounds, (n, self.latent_dim))

    def _make_data(self, init_conds: np.ndarray, timesteps: int, control=None) -> np.ndarray:
        control = np.zeros((timesteps, self.embed_dim)) if control is None else control
        control_func = self._make_staircase_func(timesteps, control)
        def dynamics(t, y): return self.A @ y + self.B @ control_func(t)
        data = []
        for y0 in init_conds:
            sol = solve_ivp(dynamics, t_span=[0, 1], y0=y0, vectorized=False,
                            t_eval=np.linspace(0, 1, endpoint=True, num=timesteps))  # TODO: add vectorization
            data.append(sol.y)
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.C
        final_conds = data[:, -1] @ np.linalg.pinv(self.C)
        return data, final_conds
