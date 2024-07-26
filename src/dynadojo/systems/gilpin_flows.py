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

    Attributes
    ----------
    base_path : str
        Base path for the dysts library.
    json_file_path : str
        Path to the JSON file containing chaotic attractors data.

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
    _all_systems(cls) -> list
        Class method that loads systems data and returns the list of available systems, excluding missing systems.
    """
    base_path = os.path.dirname(dysts.__file__)
    json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')

    @classmethod
    def _all_systems(cls):
        """Load systems data and return the list of ll available systems."""
        with open(cls.json_file_path, 'r') as file:
            systems_data = json.load(file)

        system_list = list(systems_data.keys())
        
        module = importlib.import_module('dysts.flows')
        all_systems = [system_name for system_name in system_list if hasattr(module, system_name)]
        
        return all_systems

    def __init__(self, latent_dim, embed_dim, system_name: str, pts_per_period=100, seed=None):
        """
        Initialize the GilpinFlowsSystem class.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space. 
        embed_dim : int
            Embedding dimension of the system. Fixed to Gilpin's set dimensionality for the particular system.
        system_name : str
            The name of the system to be used.
        pts_per_period: int
            For reasampled trajectories, the number of points per period. Default is 100.
        seed : int or None, optional
            Seed for random number generation. Default is None.
        """
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.system_name = system_name
        self.pts_per_period = pts_per_period
        
        try:
            module = importlib.import_module('dysts.flows')
            SystemClass = getattr(module, self.system_name)
            self.system = SystemClass()
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Unsupported system: {self.system_name}") from e
        
        data = self.system._load_data()
        self._embed_dim = data.get("embedding_dimension")
        self._latent_dim = self._embed_dim

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

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        if in_dist:
            tpts0 = np.arange(0, 2, 1)
            trajectories = generate_ic_ensemble(self.system, tpts0, n, random_state=self._seed)
            x0 = trajectories[:, :, 0]
            return x0
        
        # Use principal component analysis to generate out-of-distribution points. Perturbs OOD points with same method as used in generate_ic_ensemble.
        x = self.system.make_trajectory(10000, resample = True)

        mean = np.mean(x, axis=0)
        x_centered = x - mean
        U, s, _ = np.linalg.svd(x_centered, full_matrices=False)
            
        variance_explained = np.cumsum(s**2) / np.sum(s**2)
        num_components = np.argmax(variance_explained >= 0.9) + 1
        num_components = min(num_components, x.shape[1] - 1)

        U_remaining = U[:, num_components:]

        scale = 1
        frac_perturb = 0.1
        ood_points = []

        for _ in range(n):
            random_projection = np.random.uniform(-1, 1, (x.shape[1] - num_components)) * scale
            projection = U_remaining @ random_projection
            ood_point = mean + projection[:mean.shape[0]]
            
            perturbation = 1 + frac_perturb * (2 * np.random.random(len(ood_point)) - 1)
            ood_point = ood_point * perturbation

            ood_points.append(ood_point)

        return np.array(ood_points)

    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False):
        n = init_conds.shape[0]
        trajectories = np.zeros((n, timesteps, self._embed_dim))

        # Call Gilpin's make_trajectory function for each initial condition. By default, resample trajectories to have dominant Fourier components. If trajectory is cut short, disable resampling.
        for i in range(n):
            self.system.ic = init_conds[i]
            trajectory = self.system.make_trajectory(timesteps, resample=True, pts_per_period=self.pts_per_period)
            if trajectory.shape[0] < timesteps:
                trajectory = self.system.make_trajectory(timesteps, resample=False)
            assert trajectory.shape == (timesteps, self._embed_dim)
            trajectories[i] = trajectory

        return trajectories

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self._embed_dim