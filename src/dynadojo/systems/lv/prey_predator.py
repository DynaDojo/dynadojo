"""
Prey-Predator Lotka Volterra
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from ...abstractions import AbstractSystem
from ...utils.lv import plot


class PreyPredatorSystem(AbstractSystem):
    """
    Prey-Predator Lotka Volterra, generalized to n-species

    Prey:
    - positive growth rate (grow without predator)
    - negative intraspecies interaction (prevents infinite growth)
    - have 0 interaction with other prey
    - have a negative interaction with predators (-1 * predator's positive interaction)

    Predator:
    - negative growth rate (starve without prey)
    - negative intraspecies interaction (they compete heavily over preys)
    - can interact with other predators (multiple trophic levels)
    - have a positive interaction with preys (-1 * prey's negative interaction)

    Example
    ---------
    >>> from dynadojo.systems.lv import PreyPredatorSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.lv import plot
    >>> latent_dim = 5
    >>> embed_dim = 5
    >>> timesteps = 100
    >>> n = 10
    >>> system = SystemChecker(PreyPredatorSystem(dim, embed_dim))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, specieslabels=["black bear", "salmon", "eagle", "lion", "dolphin", "tiger"], max_lines=100)

    .. image:: ../_images/prey_predator.png
    """

    def __init__(self, latent_dim=2, embed_dim=2,
                 nPrey=None,
                 minK=1,
                 maxK=10,
                 IND_range=(0.1, 0.5),
                 OOD_range=(0.5, 0.9),
                 R_dist=(0.0, 0.5),
                 prey_intra_dist=(0.0,0.1),
                 predator_intra_dist=(0.0,0.1),
                 pZeroInteraction=0.1,
                 noise_scale=0.05,
                 seed=None):
        """
        Initializes a PreyPredatorSystem instance.

        Parameters
        -------------
        latent_dim : int
            The number of species, the rank(A) of interaction matrix
        embed_dim : int
            Must be the same as latent_dim
        nPrey : int,
            Number of prey species. Must be less than or equal to latent_dim. If not assigned, nPrey will be randomly chosen.
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
        prey_intra_dist : tuple
            Prey intraspecies interaction. Mu and standard deviation (spread or “width”) for normal distribution. 
            Intraspecies interactions are fixed to be negative to prevent infinite growth.
        predator_intra_dist : tuple
            Predator intraspecies interaction. Mu and standard deviation (spread or “width”) for normal distribution. 
            Intraspecies interactions are fixed to be negative to prevent infinite growth.
        pZeroInteraction : float
            Probability any given species pair will have no interaction. Creates more interesting food chains. 
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """
        super().__init__(latent_dim, embed_dim, seed)

        assert embed_dim == latent_dim
        if nPrey:
            assert nPrey <= latent_dim

        self.noise_scale = noise_scale
        self.minK = minK
        self.maxK = maxK

        self.R_dist = R_dist
        self.pZeroInteraction = pZeroInteraction
        self.prey_intra_dist = prey_intra_dist
        self.predator_intra_dist = predator_intra_dist

        self._rng = np.random.default_rng(seed)

        self.nPrey = nPrey
        if (not self.nPrey):
            if latent_dim == 1:
                self.nPrey = self._rng.uniform(0, 1)
            else:
                self.nPrey = self._rng.uniform(1, self.latent_dim)

        self.K = self._make_K(self.minK, self.maxK)  # Carrying capacity
        self.R = self._make_R()  # Growth Rate
        self.A = self._make_A()

        self.IND_range = IND_range
        self.OOD_range = OOD_range

    def _make_R(self):
        R = []
        for i in range(self.latent_dim):
            r = self._rng.normal(self.R_dist[0], self.R_dist[1])
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
        for i in range(self.latent_dim):
            if i < self.nPrey:
                k = self._rng.uniform(minK, maxK*2)
            else:
                k = self._rng.uniform(minK, maxK)
            K.append(k)
        return K

    def _make_A(self):
        A = self._rng.normal(0, 1, (self.latent_dim, self.latent_dim))
        for i in range(self.latent_dim):
            for j in range(self.latent_dim):
                if i == j:
                    if i < self.nPrey:
                        # intraspecies prey is not harsh, but needed negative to prevent infinite growth
                        A[i][j] = -1 * np.abs(self._rng.normal(self.prey_intra_dist[0], self.prey_intra_dist[1]))
                    else:
                        A[i][j] = -1 * np.abs(self._rng.normal(self.predator_intra_dist[0], self.predator_intra_dist[1]))
                elif i < self.nPrey:
                    # two preys do not interact
                    if j < self.nPrey:
                        A[i][j] = 0
                        A[j][i] = 0
                    # prey is negative interaction of predator
                    else:
                        # no interaction probability
                        if (self._rng.random() < self.pZeroInteraction):
                            A[i][j] = 0
                            A[j][i] = 0
                        else:
                            A[i][j] = -1 * np.abs(A[j][i])
                            A[j][i] = np.abs(A[j][i])

                elif i >= self.nPrey:
                    if j >= self.nPrey:
                        # no interaction probability
                        if (self._rng.random() < self.pZeroInteraction):
                            A[i][j] = 0
                            A[j][i] = 0
                        # two predators CAN interact
                        else:
                            A[i][j] = -1 * np.abs(A[j][i])
                            A[j][i] = np.abs(A[j][i])

        return A / self.K

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        all = []
        for _ in range(n):
            x0 = []
            for _ in range(self.latent_dim):
                if in_dist:
                    number = int(self._rng.uniform(
                        self.IND_range[0] * self.maxK, self.IND_range[1] * self.maxK))
                else:
                    number = int(self._rng.uniform(
                        self.OOD_range[0] * self.maxK, self.OOD_range[1] * self.maxK))
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
            dX = []
            if noisy:
                noise = self._rng.normal(
                    0, self.noise_scale, (self.latent_dim))
            else:
                noise = np.zeros((self.latent_dim))

            dX = X*(self.R + self.A@X + noise) + u[i]

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
            filepath: str = "PreyPredatorSystem_plotted_trajectories.pdf",
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