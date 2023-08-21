from dynadojo.abstractions import AbstractSystem

from scipy.integrate import solve_ivp
import scipy as scipy
import numpy as np

"""
PreyPredator Lotka Volterra, generalized to n-species

Complexity
-------
self.latent_dim, controls number of species and rank(A) of interaction matrix

Prey Predator interaction dynamics
-------
There are nPrey, settable by parameter or randomly assigned.
Prey:
- positive growth rate (grow without predator)
- slight negative intraspecies interaction (prevents infinite growth)
- have 0 interaction with other prey
- have a negative interaction with predators (-1 * predator's positive interaction)

Predator:
- negative growth rate (starve without prey)
- negative intraspecies interaction (they compete heavily over preys)
- can interact with other predators (multiple trophic levels)
- have a positive interaction with preys (-1 * prey's negative interaction)

"""

# TODO seed, comment description, cleanup init conds


class PreyPredatorSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 minK=1,
                 maxK=10,
                 noise_scale=0.05,
                 IND_range=(0.1, 0.5),  # prevent spawning extinct species
                 OOD_range=(0.5, 0.9),
                 nPrey=None,
                 seed=None):
        super().__init__(latent_dim, embed_dim)

        assert embed_dim == latent_dim
        if nPrey:
            assert nPrey <= latent_dim

        self.noise_scale = noise_scale
        self.minK = minK
        self.maxK = maxK

        self.nPrey = nPrey
        if (not self.nPrey):
            if latent_dim == 1:
                self.nPrey = np.random.randint(0, 1)
            else:
                self.nPrey = np.random.randint(1, self.latent_dim)

        self.K = self._make_K(self.minK, self.maxK)  # Carrying capacity
        self.R = self._make_R()  # Growth Rate
        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        R = []
        for i in range(self._latent_dim):
            r = np.random.normal(0.0, 0.5)
            if i < self.nPrey:
                # R[i] must be positive for prey
                r = np.abs(r)
            else:
                # R[i] must be negative for predators
                r = -1*np.abs(r)
            R.append(r)
        return R

    def _make_K(self, minK, maxK):
        K = []
        for i in range(self._latent_dim):
            if i < self.nPrey:
                k = np.random.uniform(minK, maxK*2)
            else:
                k = np.random.uniform(minK, maxK)
            K.append(k)
        return K

    def _make_A(self):
        A = np.random.normal(0, 1, (self._latent_dim, self._latent_dim))
        for i in range(self._latent_dim):
            for j in range(self._latent_dim):
                if i == j:
                    if i < self.nPrey:
                        # intraspecies prey is not harsh, but needed negative to prevent infinite growth
                        A[i][j] = -1 * np.abs(np.random.normal(0, 0.01))
                    else:
                        A[i][j] = -1 * np.abs(np.random.normal(0, 0.1))
                elif i < self.nPrey:
                    # two preys do not interact
                    if j < self.nPrey:
                        A[i][j] = 0
                        A[j][i] = 0
                    # prey is negative interaction of predator
                    else:
                        # no interaction probability
                        if (np.random.random() < 0.1):
                            A[i][j] = 0
                            A[j][i] = 0
                        else:
                            A[i][j] = -1 * np.abs(A[j][i])
                            A[j][i] = np.abs(A[j][i])

                elif i >= self.nPrey:
                    if j >= self.nPrey:
                        # no interaction probability
                        if (np.random.random() < 0.1):
                            A[i][j] = 0
                            A[j][i] = 0
                        # two predators CAN interact
                        else:
                            A[i][j] = -1 * np.abs(A[j][i])
                            A[j][i] = np.abs(A[j][i])

        return A / self.K

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            temp = []
            for s in range(self._latent_dim):
                if in_dist:
                    number = int(np.random.uniform(
                        self.IND_range[0] * self.maxK, self.IND_range[1] * self.maxK))
                else:
                    number = int(np.random.uniform(
                        self.OOD_range[0] * self.maxK, self.OOD_range[1] * self.maxK))
                number = np.max([1, number])
                temp.append(number)
            x0.append(temp)

        return x0

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        time = np.linspace(0, timesteps, timesteps)

        def dynamics(t, X, u):
            i = np.argmin(np.abs(t - time))
            dX = []
            if noisy:
                noise = np.random.normal(
                    0, self.noise_scale, (self.latent_dim))
            else:
                noise = np.zeros((self.latent_dim))

            dX = X*(self.R + self.A@X + noise) + u[i]

            return dX

        sol = []
        if control:
            for x0, u in zip(init_conds, control):
                sol = solve_ivp(dynamics, t_span=[
                                0, timesteps], y0=x0, t_eval=time, dense_output=True, args=(u,))
                data.append(sol.y)

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
