"""
Competitive Lotka Volterra
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from ...abstractions import AbstractSystem
from ...utils.lv import plot

class CompetitiveLVSystem(AbstractSystem):
    """
    Competitive Lotka Volterra, generalized to n-species
    
    Interspecies interaction have only positive weighted interactions, a[i][j] ≥ 0 for all i, j
    Intraspecies interaction set to 1, a[i][i] = 1 for all i
    Growth rate for each species is positive, r[i] > 0 for all i
    Each derviate divided by species carrying capacity, K[i]

    With n ≥ 5 species, asymptotic behavior occurs, including a fixed point, a limit cycle, an n-torus, or attractors [Smale]

    More info: https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations

    Example
    ---------
    >>> from dynadojo.systems.lv import CompetitiveLVSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.lv import plot
    >>> latent_dim = 5
    >>> embed_dim = 5
    >>> timesteps = 100
    >>> n = 10
    >>> system = SystemChecker(CompetitiveLVSystem(dim, embed_dim))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, specieslabels=["black bear", "salmon", "eagle", "lion", "dolphin", "tiger"], max_lines=100)

    .. image:: ../_images/competitive.png

    """

    def __init__(self, latent_dim=2, embed_dim=2,
                 minK=1,
                 maxK=10,
                 IND_range=(0.1, 0.5),
                 OOD_range=(0.5, 0.9),
                 R_dist=(0.0, 0.5),
                 inter_dist=(0, 1),
                 noise_scale=0.01,
                 seed=None):
        """
        Initializes a CompetitiveLVSystem instance.

        Parameters
        -------------
        latent_dim : int
            The number of species, the rank(A) of interaction matrix
        embed_dim : int
            Must be the same as latent_dim

        minK : int
            Minimum carrying capacity for all species
        maxK : int
            Minimum carrying capacity for all species
        IND_range : tuple
            In-distribution range of starting population numbers. The range is multipled by maxK and randomly sampled from.
            The numbers should be greater than 0 to prevent spawing extinct species.
        OOD_range : tuple
            Out-of-distribution range of starting population numbers. The range is multipled by maxK and randomly sampled from.
            The numbers should be greater than 0 to prevent spawing extinct species.
        R_dist : tuple
            Growth rates. Mu and standard deviation (spread or “width”) for normal distribution. 
        inter_dist : tuple
            Interspecies interaction distribution. Mu and standard deviation (spread or “width”) for normal distribution. 
            Fixed to be all positive.
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """
        super().__init__(latent_dim, embed_dim, seed)

        assert embed_dim == latent_dim

        self.noise_scale = noise_scale
        self.minK = minK,
        self.maxK = maxK,
        self._rng = np.random.default_rng(seed)

        self.R_dist = R_dist
        self.inter_dist = inter_dist

        self.R = self._make_R()  # Growth Rate
        self.K = self._make_K(self.minK, self.maxK)  # Carrying capacity

        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        # r[i] must be positive
        return np.abs(self._rng.normal(self.R_dist[0], self.R_dist[1], (self.latent_dim)))

    def _make_K(self, minK, maxK):
        return self._rng.uniform(minK, maxK, (self.latent_dim))

    def _make_A(self):
        # inter-species is all positive in competitive
        A = np.abs(self._rng.normal(
            self.inter_dist[0], self.inter_dist[1], (self.latent_dim, self.latent_dim)))
        for i in range(self.latent_dim):
            for j in range(self.latent_dim):
                if i == j:
                    A[i][j] = 1  # intra-species is all 1 in competitive

        return A / self.K

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        all = []
        for _ in range(n):
            x0 = []
            for s in range(self.latent_dim):
                if in_dist:
                    number = int(self._rng.uniform(
                        self.IND_range[0] * self.K[s], self.IND_range[1] * self.K[s]))
                else:
                    number = int(self._rng.uniform(
                        self.OOD_range[0] * self.K[s], self.OOD_range[1] * self.K[s]))
                number = np.max([1, number])
                x0.append(number)
            all.append(x0)

        all = np.array(all)
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
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)

    def save_plotted_trajectories( self, 
            y_true:np.ndarray, 
            y_pred: np.ndarray,
            filepath: str = "CompetitiveLVSystem_plotted_trajectories.pdf",
            tag: str = "", 
            savefig: bool = True 
        ):
        """
        Plots the trajectories of the system and the predicted trajectories.

        Parameters
        ----------
        y : np.ndarray
            True trajectories.
        y_pred : np.ndarray
            Predicted trajectories.
        """
        fig, ax = plot([y_true, y_pred], 
                       target_dim=min(self._embed_dim, 3), 
                       labels=["true", "pred"], 
                       max_lines=10,
                       title=f"LV (C) l={self.latent_dim}, e={self._embed_dim} - {tag}")        
        if savefig:
            fig.savefig(filepath, bbox_inches='tight', dpi=300, transparent=True, format='pdf')
            plt.close(fig)
            return None, None
        else:
            return fig, ax