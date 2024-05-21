"""
Kuramoto Oscillators
"""

import numpy as np
from scipy.integrate import ode

import numpy as np
import scipy as sp

from ..abstractions import AbstractSystem

"""
Adapted from D. Laszuk, "Python implementation of Kuramoto systems," 2017-, [Online] Available: http://www.laszukdawid.com/codes
Research Source: Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3

Complexity
-------
self.latent_dim, controls number of oscillators

Other params
-------
FREQ_range, controls range of W of frequencies for each oscillators
COUPLE_range, controls range of K of  frequencies for each oscillators
self.dt, controls the step size between 0 - timesteps

"""

class KuramotoSystem(AbstractSystem):

    """
    Kuramoto, generalized to n oscillators

    Adapted from D. Laszuk, "Python implementation of Kuramoto systems," 2017-, [Online] Available: http://www.laszukdawid.com/codes
    Research Source: Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence (Vol. 19). doi: doi.org/10.1007/978-3-642-69689-3

    Example
    ---------
    >>> from dynadojo.systems.kuramoto import KuramotoSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.kuramoto import plot
    >>> latent_dim = 4
    >>> embed_dim = 4
    >>> timesteps = 300
    >>> n = 1
    >>> system = SystemChecker(KuramotoSystem(dim, embed_dim))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], timesteps=timesteps, max_lines=100)

    .. image:: ../_images/kuramoto.png

    """
    def __init__(self, latent_dim, embed_dim,
                 IND_range=(0, 10),
                 OOD_range=(10, 30),
                 FREQ_range=(0, 50),
                 COUPLE_range=(-20, 20),
                 dt=0.02,
                 noise_scale=0.6,
                 seed=None):

        """
        Initializes a KuramotoSystem instance.

        Parameters
        -------------
        latent_dim : int
            Number of oscillators
        embed_dim : int
            Must be the same as latent_dim
        IND_range : tuple
            In-distribution range of starting trajectory values.
        OOD_range : tuple
            Out-of-distribution range of starting trajectory values.
        FREQ_range : tuple
            Controls range of W of frequencies for each oscillators
        COUPLE_range : tuple
            Controls range of K of frequencies for each oscillators
        dt : float
            The timeinterval between timesteps, or the granularity of the simulation
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """
        super().__init__(latent_dim, embed_dim, seed)

        assert embed_dim == latent_dim

        self.noise_scale = noise_scale
        self.IND_range = IND_range
        self.OOD_range = OOD_range

        self._rng = np.random.default_rng(seed)

        self.dtype = np.float32
        self.dt = dt

        self.FREQ_range = FREQ_range
        self.COUPLE_range = COUPLE_range

        self.W = self._make_W()
        self.K = self._make_K()

    def _make_W(self):
        return self._rng.uniform(self.FREQ_range[0], self.FREQ_range[1], (self.latent_dim))

    def _make_K(self):
        _K = self._rng.uniform(
            self.COUPLE_range[0], self.COUPLE_range[1], (self.latent_dim, self.latent_dim))
        _K2 = self._rng.uniform(0, 1, (self.latent_dim, self.latent_dim))
        return np.dstack((_K, _K2)).T

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            if in_dist:
                x0.append(self._rng.uniform(
                    self.IND_range[0], self.IND_range[1], (self.latent_dim)))
            else:
                x0.append(self._rng.uniform(
                    self.OOD_range[0], self.OOD_range[1], (self.latent_dim)))
        return np.array(x0)

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        T = np.arange(0, timesteps * self.dt, self.dt)

        def kuramoto_ODE(t, y, arg):
            w, k, noisy = arg
            yt = y[:, None]
            dy = y-yt
            phase = w.astype(self.dtype)
            if noisy:
                noise = self._rng.normal(
                    0, self.noise_scale, (self.latent_dim))
                phase += noise
            for m, _k in enumerate(k):
                phase += np.sum(_k*np.sin((m+1)*dy), axis=1)

            return phase

        def solve(t, x0, U):
            kODE = ode(kuramoto_ODE)
            kODE.set_integrator("dopri5")

            # Set parameters into model
            kODE.set_initial_value(x0, t[0])
            kODE.set_f_params((self.W, self.K, noisy))

            phase = np.empty((self.latent_dim, len(t))) # n_osc x timesteps

            # Run ODE integrator
            for idx, _t in enumerate(t[1:]):
                phase[:, idx] = kODE.y + U[idx]
                kODE.integrate(_t)

            phase[:, -1] = kODE.y 

            return phase

        sol = []
        if control is not None:
            for x0, U in zip(init_conds, control):
                sol = solve(T, x0, U)
                data.append(sol)

        else:
            for x0 in init_conds:
                U = np.zeros((timesteps, self.latent_dim))
                sol = solve(T, x0, U)
                data.append(sol)
        
       
        data = np.transpose(np.array(data), axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
