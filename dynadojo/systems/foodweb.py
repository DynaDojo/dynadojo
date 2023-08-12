from dynascale.abstractions import AbstractSystem

from scipy.integrate import odeint
import scipy as scipy
import numpy as np

RNG = np.random.default_rng()


"""
Competitive Lotka Volterra, generalized to n-species

Complexity
-------
self.embed_dim, controls number of species and rank(A) of interaction matrix

Competitive interaction dynamics
-------
Interspecies have only positive weighted interactions, a[i][j] ≥ 0 for all i, j
Intraspecies set to 1, a[i][i] = 1 for all i
Growth rate for each species is positive, r[i] > 0 for all i
Each derviate divided by species carrying capacity, K[i]

With n ≥ 5 species, asymptotic behavior occurs, including a fixed point, a limit cycle, an n-torus, or attractors [Smale]

More info: https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations

"""
class CLVSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 minK=1, 
                 maxK=1000, 
                 noise_scale=0.0000001,
                 IND_range=(0.1, 0.5), # prevent spawning extinct species
                 OOD_range=(0.5, 0.9),
                 seed=None):
        super().__init__(latent_dim, embed_dim)

        self._noise_scale = noise_scale
        self.minK=minK, 
        self.maxK=maxK, 
        self._rng = np.random.default_rng(seed)

        self.R = self._make_R() # Growth Rate
        self.K = self._make_K(self.minK, self.maxK) # Carrying capacity per species

        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        return np.random.normal(0.5, 0.1, (self._embed_dim)) # r[i] must be positive
    
    def _make_K(self, minK, maxK):
        return np.random.uniform(minK, maxK, (self._embed_dim))

    def _make_C(self, minC, maxC):
            return np.random.uniform(minC, maxC, (self._embed_dim))

    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


    def _make_A(self):
        A = np.random.normal(1, 0.3, (self._embed_dim, self._embed_dim)) #inter-species is all positive in competitive
        for i in range(self._embed_dim):
            for j in range(self._embed_dim):
                if i == j:
                    A[i][j] = 1 #intra-species is all 1 in competitive
    
        return A / self.K

    @AbstractSystem.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self.R = self._make_R() # Growth Rate
        self.K = self._make_K(self.minK, self.maxK) # Carrying capacity per species

        self.A = self._make_A()

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            temp = []
            for s in range(self._embed_dim):
                if in_dist:
                    number = int(np.random.uniform(self.IND_range[0] * self.K[s], self.IND_range[1] * self.K[s]))
                else:
                    number = int(np.random.uniform(self.OOD_range[0] * self.K[s], self.OOD_range[1] * self.K[s]))
                temp.append(number)
            x0.append(temp)
        return x0


    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:

        time = np.linspace(0, timesteps, timesteps)

        def dynamics(X, t, u):
            i = np.argmin(np.abs(t - time))

            if noisy:
                noise = np.random.normal(0,0.01,(self.embed_dim))/self.K
                print(noise)
                dX = (X * self.R *((1 - (self.A @ X))))
                print(dX)
                dX += noise
                #print(dX)
                print('-'*10)
            else:
                dX = (X * (self.R - (self.A @ X))) + u[i]
            return dX
        
        sol = []
        if control:
            for x0, u in zip(init_conds, control):
                sol.append(odeint(dynamics, x0, time, args=(u,)))

        else:
            for x0 in init_conds:
                u = np.zeros((timesteps, self.embed_dim))
                sol.append(odeint(dynamics, x0, time, args=(u,)))

        return sol

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
