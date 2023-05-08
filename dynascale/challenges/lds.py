from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from dynascale.abstractions import Challenge

RNG = np.random.default_rng()


class LDSChallenge(Challenge):
    def __init__(self, latent_dim, embed_dim, A_range=(-10, 10), B_range=(-10, 10), C_range=(-10, 10),
                 init_cond_in_dist_range = (0, 10), init_cond_out_dist_range = (-10, 0)):
        self._A_range = A_range
        self._B_range = B_range
        self._C_range = C_range
        self._init_cond_in_dist_range = init_cond_in_dist_range
        self._init_cond_out_dist_range = init_cond_out_dist_range
        self.A = RNG.uniform(*A_range, (latent_dim, latent_dim))
        self.B = RNG.uniform(*B_range, (latent_dim, embed_dim))
        self.C = RNG.uniform(*C_range, (latent_dim, embed_dim))
        super().__init__(latent_dim, embed_dim)

    @Challenge.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self.B = RNG.uniform(*self._B_range, (self.latent_dim, self.embed_dim))
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    @Challenge.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self.A = RNG.uniform(*self._A_range, (self.latent_dim, self.latent_dim))
        self.B = RNG.uniform(*self._B_range, (self.latent_dim, self.embed_dim))
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        init_cond_range = self._init_cond_in_dist_range if in_dist else self._init_cond_out_dist_range
        return RNG.uniform(*init_cond_range, (n, self.embed_dim))

    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int) -> np.ndarray:
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.C)
        time = np.linspace(0, 1, endpoint=True, num=timesteps)
        for x0, u in zip(init_conds, control):
            def control_func(t):
                i = np.argmin(np.abs(t - time))
                return u[i]

            def dynamics(t, x):
                return self.A @ x + self.B @ control_func(t)

            sol = solve_ivp(dynamics, t_span=[0, 1], y0=x0, vectorized=False, t_eval=time)
            data.append(sol.y)
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.C
        return data

    def _calc_loss(self, x, y) -> float:
        error = np.linalg.norm(x - y, axis=0)
        return np.mean(error ** 2)