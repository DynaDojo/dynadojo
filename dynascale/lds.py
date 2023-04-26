from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .abstractions import Factory

RNG = np.random.default_rng()


class LDSFactory(Factory):
    def _make_data(self, n: int, timesteps: int, latent_dim: int, embed_dim: int, control: Callable,
                   A_range=(-10, 10), B_range=(-10, 10), init_cond_range=(-10, 10), W_range=(-10, 10),
                   *args, **kwargs) -> np.ndarray:
        A = RNG.uniform(*A_range, (latent_dim, latent_dim))  # TODO: make sure A is non-degenerate
        B = RNG.uniform(*B_range, (latent_dim, latent_dim))  # TODO: fix generator parameters
        init_conds = RNG.uniform(*init_cond_range, (n, latent_dim))
        u = lambda t: np.zeros(latent_dim) if control is None else control  # TODO: make staircase, embed dimension (make B nonsquare)
        fun = lambda t, y: A @ y + B @ u(t)
        data = []
        for y0 in init_conds:
            sol = solve_ivp(fun, t_span=[0, 1], y0=y0, t_eval=np.linspace(0, 1, endpoint=True, num=timesteps),
                            vectorized=False, dense_output=True)  # TODO: add vectorization
            data.append(sol.y)
        W = RNG.uniform(*W_range, (latent_dim, embed_dim))
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ W
        return data
