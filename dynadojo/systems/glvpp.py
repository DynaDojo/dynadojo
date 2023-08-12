from dynascale.abstractions import AbstractSystem


from scipy.integrate import odeint
import scipy as scipy
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from sympy import symbols, solve

RNG = np.random.default_rng()

# Generalized Lotka Volterra, Predator-Prey
# Complexity: embed_dim, which controls number of species, and thus rank(A) of interaction matrix
class GLVPPSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 minK = 1, 
                 maxK=1000, 
                 minC=1,
                 maxC=30,
                 IND_range=(0.1, 0.35),
                 OOD_range=(0.36, 0.6)):
        super().__init__(latent_dim, embed_dim)

        self.R = self._make_R() # Growth Rate
        self.K = self._make_K(minK, maxK) # Carrying capacity per species
        self.C = self._make_C(minC, maxC) # Carrying capacity per species

        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        return np.random.normal(0, 0.25, (self._embed_dim))
    
    def _make_K(self, minK, maxK):
        return np.random.uniform(minK, maxK, (self._embed_dim))

    def _make_C(self, minC, maxC):
            return np.random.uniform(minC, maxC, (self._embed_dim))

    def is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    # TODO: allow for set number of prey species
    def _make_A(self):
        # first set inter-species interaction, rewrite diagonal for intra-species

        A = np.zeros((self.embed_dim, self.embed_dim))
        for i in range(self._embed_dim):
            for j in range(self._embed_dim):
                
                if j < i:
                    print("bottom half")
                    A[i][j] = A[j][i] * -1
                elif i == j:
                    print("diagonal")
                    if self.R[i] < 0:
                        A[i][j] = 0
                    else:
                        A[i][j] = (self.R[i]*self.C[i])/self.K[i]
                else:
                    print("top half")
                    A[i][j] = np.random.normal(0, 2)

        return A 

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

        def dynamics(X, t):
            H = []
            for x in X:
                if x<0:
                    H.append(0)
                else:
                    H.append(1)
            result = X*self.R * (1 - X*H/self.K) - (self.A@X)/self.C
            
            #print(result)
            return result
        
        print(self.K)
        print(self.A)
        print(self.R)
        sol = []
        for n in init_conds:
            sol.append(odeint(dynamics, n, time))

        return sol

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
