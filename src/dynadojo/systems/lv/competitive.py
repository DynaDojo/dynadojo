import numpy as np
from scipy.integrate import solve_ivp

from ...abstractions import AbstractSystem

"""
Competitive Lotka Volterra, generalized to n-species

Complexity
-------
self.latent_dim, controls number of species and rank(A) of interaction matrix

Competitive interaction dynamics
-------
Interspecies have only positive weighted interactions, a[i][j] ≥ 0 for all i, j
Intraspecies set to 1, a[i][i] = 1 for all i
Growth rate for each species is positive, r[i] > 0 for all i
Each derviate divided by species carrying capacity, K[i]

With n ≥ 5 species, asymptotic behavior occurs, including a fixed point, a limit cycle, an n-torus, or attractors [Smale]

More info: https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations

"""


class CompetitiveLVSystem(AbstractSystem):
    def __init__(self, latent_dim=2, embed_dim=2,
                 minK=1,
                 maxK=10,
                 noise_scale=0.01,
                 IND_range=(0.1, 0.5),  # prevent spawning extinct species
                 OOD_range=(0.5, 0.9),
                 R_range=(0.0, 0.5),
                 interspecies_range=(0, 1),
                 seed=None):
        super().__init__(latent_dim, embed_dim, seed)

        assert embed_dim == latent_dim

        self.noise_scale = noise_scale
        self.minK = minK,
        self.maxK = maxK,
        self._rng = np.random.default_rng(seed)

        self.R_range = R_range
        self.interspecies_range = interspecies_range

        self.R = self._make_R()  # Growth Rate
        self.K = self._make_K(self.minK, self.maxK)  # Carrying capacity

        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        # r[i] must be positive
        return np.abs(self._rng.normal(self.R_range[0], self.R_range[1], (self._latent_dim)))

    def _make_K(self, minK, maxK):
        return self._rng.uniform(minK, maxK, (self._latent_dim))

    def _make_A(self):
        # inter-species is all positive in competitive
        A = np.abs(self._rng.normal(
            self.interspecies_range[0], self.interspecies_range[1], (self._latent_dim, self._latent_dim)))
        for i in range(self._latent_dim):
            for j in range(self._latent_dim):
                if i == j:
                    A[i][j] = 1  # intra-species is all 1 in competitive

        return A / self.K

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        all = []
        for _ in range(n):
            x0 = []
            for s in range(self._latent_dim):
                if in_dist:
                    number = int(self._rng.uniform(
                        self.IND_range[0] * self.K[s], self.IND_range[1] * self.K[s]))
                else:
                    number = int(self._rng.uniform(
                        self.OOD_range[0] * self.K[s], self.OOD_range[1] * self.K[s]))
                number = np.max([1, number])
                x0.append(number)
            all.append(x0)

        return all

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        time = np.linspace(0, timesteps, timesteps)

        def dynamics(t, X, u):
            i = np.argmin(np.abs(t - time))

            if noisy:
                noise = self._rng.normal(
                    0, self.noise_scale, (self.latent_dim))
                dX = (X * self.R * ((1 - (self.A @ X))+noise)) + u[i]
            else:
                dX = (X * self.R * ((1 - (self.A @ X)))) + u[i]
            return dX

        sol = []
        if control is not None:
            for x0, u in zip(init_conds, control):
                sol = solve_ivp(dynamics, t_span=[
                                0, timesteps], y0=x0, t_eval=time, dense_output=True, args=(u,))

        else:
            for x0 in init_conds:
                u = np.zeros((timesteps, self.latent_dim))
                sol = solve_ivp(dynamics, t_span=[
                                0, timesteps], y0=x0, t_eval=time, dense_output=True, args=(u,))
                data.append(sol.y)

        data = np.transpose(np.array(data), axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
