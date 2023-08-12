import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group
from sklearn.metrics import mean_squared_error

from dynascale.abstractions import AbstractSystem

from GEMFPy import *

RNG = np.random.default_rng()


class SIRSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 beta=0.4,
                 gamma=0.1,     
                 init_cond_in_dist_range=(0, 0.5),
                 init_cond_out_dist_range=(0.51, 1.0),
                 ):
        super().__init__(latent_dim, embed_dim)
        self._beta=beta,
        self._gamma=gamma,  
        self._N = latent_dim
        
        assert any(map(lambda x: x < 1.0, init_cond_in_dist_range))
        assert any(map(lambda x: x < 1.0, init_cond_out_dist_range))
        self._init_cond_in_dist_range = init_cond_in_dist_range
        self._init_cond_out_dist_range = init_cond_out_dist_range

       

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        init_cond_range = self._init_cond_in_dist_range if in_dist else self._init_cond_out_dist_range
        I0 = RNG.uniform(*init_cond_range) * self._N
        R0 = 0
        S0 = self._N - I0 - R0

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        time = np.linspace(0, 1, num=timesteps)

        def dynamics(t, x, u):
            S, I, R = x
            dSdt = -(1-u)*beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
            if noisy:
                return self.A @ x + self.B @ u[i] + RNG.normal(scale=self._scale, size=(self.latent_dim))
            else:
                return self.A @ x + self.B @ u[i]

        for x0, u in zip(init_conds, control):
            sol = solve_ivp(dynamics, t_span=[0, 1], y0=x0, t_eval=time, dense_output=True, args=(u,))
            data.append(sol.y)
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.C
        return data

    def calc_loss(self, x, y) -> float:
        # TODO: add more details
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self.embed_dim  # TODO: ask Max
