from typing import Callable

from tqdm.auto import tqdm
import numpy as np
from scipy.integrate import solve_ivp

from dynascale.abstractions import Challenge

RNG = np.random.default_rng()


class LDSChallenge(Challenge):
    def __init__(self, latent_dim, embed_dim,
                 Q_range=(-2, 2),
                 eig_range=(-2, 2),
                 B_range=(-1, 1),
                 C_range=(-1, 1),
                 init_cond_in_dist_range = (0, 10),
                 init_cond_out_dist_range = (-10, 0),
                 scale=0.01
                 ):
        super().__init__(latent_dim, embed_dim)
        self._Q_range = Q_range
        self._eig_range = eig_range
        self._B_range = B_range
        self._C_range = C_range
        self._init_cond_in_dist_range = init_cond_in_dist_range
        self._init_cond_out_dist_range = init_cond_out_dist_range
        self._scale = scale

        self.A = self._make_A()
        self.B = RNG.uniform(*B_range, (latent_dim, embed_dim))
        self.C = RNG.uniform(*C_range, (latent_dim, embed_dim))

    def _make_A(self):
        # 0.5 prob that all eigenvalues are real
        eigs_all_real = RNG.uniform() <= 0.5
        eigs_all_real = False  # TODO: remove debugging
        if eigs_all_real:
            eigs = RNG.uniform(*self._eig_range, size=self._latent_dim)
            Q = RNG.uniform(*self._Q_range, size=(self.latent_dim, self.latent_dim))
        else:
            num_complex = RNG.integers(0, self.latent_dim // 2, endpoint=True)  # counts number of conjugate pairs
            if num_complex % 2 == 1:
                num_complex -= 1
            num_real = self.latent_dim - 2 * num_complex

            # make real eigenvalue and eigenvectors
            real_eigs = RNG.uniform(*self._eig_range, size=num_real)
            real_eigenvectors = RNG.uniform(*self._Q_range, size=(num_real, self.latent_dim))

            # make complex eigenvalues and eigenvectors with conjugates
            complex_eigs = RNG.uniform(*self._eig_range, size=num_complex) + 1j * RNG.uniform(*self._eig_range, size=num_complex)
            complex_eigenvectors = RNG.uniform(*self._Q_range, size=(num_complex, self.latent_dim)) + 1j * RNG.uniform(*self._Q_range, size=(num_complex, self.latent_dim))

            eigs = np.concatenate((real_eigs, complex_eigs, complex_eigs.conjugate()))
            Q = np.concatenate((real_eigenvectors, complex_eigenvectors, complex_eigenvectors.conjugate()))

            # shuffle columns
            ind = RNG.permutation(self.latent_dim)
            eigs = eigs[ind]
            Q = Q[ind]

        A = Q @ np.diag(eigs) @ np.linalg.inv(Q)
        return A

    @Challenge.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self.B = RNG.uniform(*self._B_range, (self.latent_dim, self.embed_dim))
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    @Challenge.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self.A = self._make_A()
        self.B = RNG.uniform(*self._B_range, (self.latent_dim, self.embed_dim))
        self.C = RNG.uniform(*self._C_range, (self.latent_dim, self.embed_dim))

    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        init_cond_range = self._init_cond_in_dist_range if in_dist else self._init_cond_out_dist_range
        return RNG.uniform(*init_cond_range, (n, self.embed_dim))

    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.C)
        time = np.linspace(0, 1, endpoint=True, num=timesteps)
        for x0, u in zip(init_conds, control):
            def control_func(t):
                i = np.argmin(np.abs(t - time))
                return u[i]

            def dynamics(t, x):
                if noisy:
                    return self.A @ x + self.B @ control_func(t) + RNG.normal(scale=self._scale, size=(self.latent_dim))
                else:
                    return self.A @ x + self.B @ control_func(t)

            sol = solve_ivp(dynamics, t_span=[0, 1], y0=x0, vectorized=False, t_eval=time)
            data.append(sol.y)
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.C
        return data

    def _calc_loss(self, x, y) -> float:
        error = np.linalg.norm(x - y, axis=0)
        return np.mean(error ** 2)