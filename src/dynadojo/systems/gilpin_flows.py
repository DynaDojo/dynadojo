import numpy as np
from ..abstractions import AbstractSystem
import importlib
import warnings
import os
import json
import dysts
from dysts.utils import generate_ic_ensemble

class GilpinFlowsSystem(AbstractSystem):
    """
    Gilpin Flows System for modeling chaotic attractors using systems from the dysts library.

    Example
    -------
    >>> from dynadojo.systems.gilpin_flows import GilpinFlowsSystem
    >>> latent_dim = 3
    >>> embed_dim = 3
    >>> system_name = 'Lorenz'
    >>> system = GilpinFlowsSystem(latent_dim, embed_dim, system_name)
    >>> n = 10
    >>> init_conds = system.make_init_conds(n, in_dist=True)
    >>> timesteps = 100
    >>> trajectories = system.make_data(init_conds, timesteps)
    >>> error = system.calc_error(trajectories[0], trajectories[1])
    >>> control_cost = system.calc_control_cost(np.zeros((n, timesteps, embed_dim)))

    Methods
    -------
    __init__(self, latent_dim, embed_dim, system_name: str, seed=None)
        Initializes the system with given dimensions and system name.
    make_init_conds(self, n: int, in_dist=True) -> np.ndarray
        Generates initial conditions for the system.
    make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False) -> np.ndarray
        Generates trajectories from initial conditions.
    calc_error(self, x: np.ndarray, y: np.ndarray) -> float
        Calculates the mean squared error between two arrays.
    calc_control_cost(self, control: np.ndarray) -> float
        Calculates the control cost.
    all_systems(cls) -> list
        Class method that loads systems data and returns the list of available systems, excluding missing systems.
    """
    base_path = os.path.dirname(dysts.__file__)
    json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')

    @classmethod
    def all_systems(cls):
        """Load systems data and return the list of all available systems."""
        with open(cls.json_file_path, 'r') as file:
            systems_data = json.load(file)

        module = importlib.import_module('dysts.flows')
        all_systems = []
        for system_name, attributes in systems_data.items():
            if hasattr(module, system_name):
                if attributes.get('delay') == False:
                    all_systems.append(system_name)
        
        return all_systems

    def __init__(self, latent_dim=3, embed_dim=3, system_name="Lorenz", pts_per_period=100, seed=None):
        """
        Initialize the GilpinFlowsSystem class.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space. Fixed to Gilpin's set dimensionality for the particular system.
        embed_dim : int
            Embedding dimension of the system. Fixed to Gilpin's set dimensionality for the particular system.
        system_name : str
            The name of the system to be used. Defaults to Lorenz.
        pts_per_period: int
            For reasampled trajectories, the number of points per period. Default is 100.
        seed : int or None, optional
            Seed for random number generation. Default is None.
        """
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.system_name = system_name
        self.pts_per_period = pts_per_period
        self._rng = np.random.default_rng(seed)
        
        try:
            module = importlib.import_module('dysts.flows')
            SystemClass = getattr(module, self.system_name)
            self.system = SystemClass()
            self.system.random_state = seed
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Unsupported system: {self.system_name}") from e
        
        data = self.system._load_data()

        data_embed_dim = data.get("embedding_dimension")

        if self._embed_dim != data_embed_dim:
            # print(f"Inputted embedded dimension of {self._embed_dim}, but Gilpin's system has an embedded dimension of {data_embed_dim}. Adjusting the embedded dimension to {data_embed_dim}.")
            self._embed_dim = data_embed_dim

        if self._latent_dim != data_embed_dim:
            # print(f"Inputted latent dimension of {self._latent_dim}, but Gilpin's system has a dimension of {data_embed_dim}. Adjusting the latent dimension to {data_embed_dim}.")
            self._latent_dim = data_embed_dim


        attributes = [
            "embedding_dimension", "bifurcation_parameter", "citation", 
            "correlation_dimension", "delay", "description", "dt", 
            "hamiltonian", "initial_conditions", "kaplan_yorke_dimension", 
            "lyapunov_spectrum_estimated", "maximum_lyapunov_estimated", 
            "multiscale_entropy", "nonautonomous", "parameters", "period", 
            "pesin_entropy", "unbounded_indices"
        ]

        for attr in attributes:
            value = data.get(attr, 'NA')
            setattr(self, attr, value)
            if value == 'NA':
                warnings.warn(
                    f"Attribute '{attr}' not found for system '{self.system_name}'", 
                    UserWarning
                )

        self.reference_traj = self.system.make_trajectory(1000, method="Radau")

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:

        mean = np.mean(self.reference_traj, axis=0)
        variance = np.var(self.reference_traj, axis=0)

        mean_magnitude = np.linalg.norm(mean)
        std = np.sqrt(np.mean(variance))
        
        # Weights for mean and variance contributions
        mean_weight = .25
        variance_weight = .75

        # Calculate scale and frac_perturb based on a weighted sum of mean and std
        scale = std #0.001 * (mean_weight * mean_magnitude + variance_weight * std)
        frac_perturb = 0.1 #0.001 * (mean_weight * mean_magnitude + variance_weight * std)
        points = []

        #print(scale)
        #print(frac_perturb)
        for _ in range(n):
            # Randomly select a point on the reference trajectory
            random_index = self._rng.integers(0, len(self.reference_traj))
            point = self.reference_traj[random_index]

            # Use principal component analysis to generate out-of-distribution points.
            if not in_dist:
                centered = self.reference_traj - mean
                U, s, Vt = np.linalg.svd(centered, full_matrices=False)
                    
                variance_explained = np.cumsum(s**2) / np.sum(s**2)
                min_var_idx = np.argmax(variance_explained >= 0.8) + 1
                min_var_idx = min(min_var_idx, self.reference_traj.shape[1] - 1) # Ensure at least one component remains

                Vt_remaining = Vt[min_var_idx:, :]

                random_projection = self._rng.uniform(-1, 1, Vt_remaining.shape[0]) * scale
                projection = Vt_remaining.T @ random_projection
                point = point + projection
            
            perturbation = 1 + frac_perturb * (2 * self._rng.random(len(point)) - 1)
            point = point * perturbation

            points.append(point)

        return np.array(points)

    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False):
        n = init_conds.shape[0]
        trajectories = np.zeros((n, timesteps, self._embed_dim))

        # Call Gilpin's make_trajectory function for each initial condition. By default, resample trajectories to have dominant Fourier components. If trajectory is cut short, disable resampling.
        for i in range(n):
            self.system.ic = init_conds[i]
            trajectory = self.system.make_trajectory(timesteps, pts_per_period=self.pts_per_period, method="Radau")
            # if trajectory.shape[0] < timesteps:
            #     trajectory = self.system.make_trajectory(timesteps, resample=False)
            while trajectory.shape != (timesteps, self._embed_dim):
                self.system.ic = self.make_init_conds(1)[0]
                trajectory = self.system.make_trajectory(timesteps, pts_per_period=self.pts_per_period, method="Radau")
                #print(trajectory.shape)
                #continue
            #assert trajectory.shape == (timesteps, self._embed_dim)
            trajectories[i] = trajectory

        return trajectories

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self._embed_dim