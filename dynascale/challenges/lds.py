from scipy.stats import ortho_group
import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
from tqdm.auto import tqdm
from multiprocessing import Pool


from dynascale.abstractions import Challenge

RNG = np.random.default_rng()


class LDSChallenge(Challenge):
    def __init__(self, latent_dim, embed_dim,
                 # negative eigenvalues produce stable linear system (https://en.wikipedia.org/wiki/Stability_theory)
                 A_eigval_range=(-5, 0),
                 A_eigvec_range=(-1, 1),
                 BC_sv_range=(-1, 1),
                 init_cond_in_dist_range=(0, 10),
                 init_cond_out_dist_range=(-10, 0),
                 scale=0.01
                 ):
        super().__init__(latent_dim, embed_dim)
        self._A_eigval_range = A_eigval_range
        self._A_eigvec_range = A_eigvec_range
        self._BC_sv_range = BC_sv_range
        self._init_cond_in_dist_range = init_cond_in_dist_range
        self._init_cond_out_dist_range = init_cond_out_dist_range
        self._scale = scale

        self.A = self._make_A()
        self.B = self._make_BC()
        self.C = self._make_BC()

    @staticmethod
    def _eigenvalues_to_matrix(dim, eig_range, eigvec_range):
        # use eigendecomposition to produce a real-valued matrix with target eigenvalues
        if RNG.uniform() <= 0:  # TODO: change
            # 50% chance to have all real eigenvalues
            eigenvalues = RNG.uniform(*eig_range, size=dim)
            Q = RNG.uniform(*eigvec_range, size=(dim, dim))
            Q = sp.linalg.orth(Q)  # orthogonal matrices preserve operator norms
        else:
            # 50% chance to have some complex eigenvalues
            num_complex = RNG.integers(0, dim, endpoint=True)
            if num_complex % 2 == 1:
                num_complex -= 1
            num_real = dim - num_complex

            real_eigenvalues = RNG.uniform(*eig_range, size=num_real)
            real_eigenvectors = RNG.uniform(*eigvec_range, size=(dim, num_real))

            # complex eigenvectors and eigenvalues must have conjugate pairs in the diagonalization for A to be real
            complex_eigenvalues = RNG.uniform(*eig_range, size=num_complex // 2) + 1j * RNG.uniform(*eig_range, size=num_complex // 2)
            complex_eigenvectors = RNG.uniform(*eigvec_range, size=(dim, num_complex // 2)) + 1j * RNG.uniform(*eigvec_range, size=(dim, num_complex // 2))

            eigenvalues = np.concatenate((real_eigenvalues, complex_eigenvalues, complex_eigenvalues.conjugate()))
            Q = np.hstack((real_eigenvectors, complex_eigenvectors, complex_eigenvectors.conjugate()))

            # shuffle columns
            pi = RNG.permutation(dim)
            Q = Q[pi]
            eigenvalues = eigenvalues[pi]

        M = Q @ np.diag(eigenvalues) @ np.linalg.inv(Q)
        return M.real

    @staticmethod
    def _singular_values_to_matrix(m, n, sv_range):
        U = ortho_group.rvs(m)
        sigma = np.eye(m, n) * RNG.uniform(*sv_range, size=n)
        V = ortho_group.rvs(n)
        M = U @ sigma @ V
        return M

    def _make_A(self):
        return self._eigenvalues_to_matrix(self.latent_dim, self._A_eigval_range, self._A_eigvec_range)

    def _make_BC(self):
        return self._singular_values_to_matrix(self.latent_dim, self.embed_dim, self._BC_sv_range)

    @Challenge.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self.B = self._make_BC()
        self.C = self._make_BC()

    @Challenge.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self.A = self._make_A()
        self.B = self._make_BC()
        self.C = self._make_BC()

    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        init_cond_range = self._init_cond_in_dist_range if in_dist else self._init_cond_out_dist_range
        return RNG.uniform(*init_cond_range, (n, self.embed_dim))

    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.C)
        time = np.linspace(0, 1, num=timesteps)

        def dynamics(t, x, u):
            i = np.argmin(np.abs(t - time))
            if noisy:
                return self.A @ x + self.B @ u[i] + RNG.normal(scale=self._scale, size=(self.latent_dim))
            else:
                return self.A @ x + self.B @ u[i]

        for x0, u in tqdm(zip(init_conds, control), total=len(init_conds), leave=False):
            sol = solve_ivp(dynamics, t_span=[0, 1], y0=x0, t_eval=time, dense_output=True, args=(u,))
            data.append(sol.y)
        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.C
        return data

    def _calc_loss(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def _calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2).sum()  # TODO: check w/ others
