import numpy as np
from ..abstractions import AbstractSystem
import importlib
import warnings
import os
import json
import dysts
from dysts.utils import generate_ic_ensemble

class GilpinFlowsSystem(AbstractSystem):
    base_path = os.path.dirname(dysts.__file__)
    json_file_path = os.path.join(base_path, 'data', 'chaotic_attractors.json')

    with open(json_file_path, 'r') as file:
        systems_data = json.load(file)

    all_systems = list(systems_data.keys())

    def __init__(self, latent_dim, embed_dim, system_name: str, seed=None):
        super().__init__(latent_dim, embed_dim, seed=seed)
        self.system_name = system_name
        self.trajectories = None
        
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
        
        x = self.system.make_trajectory(10000, resample = True)

        mean = np.mean(x, axis=0)
        x_centered = x - mean
        U, s, _ = np.linalg.svd(x_centered, full_matrices=False)
            
        scale = 10.0
        ood_points = []

        smallest_indices = np.argsort(s)[:n]
        U_least_var = U[:, smallest_indices]

        for _ in range(n):
            random_projection = np.random.randn(len(smallest_indices))
            projection = U_least_var @ random_projection
            ood_point = mean + scale * projection[:mean.shape[0]]
            ood_points.append(ood_point)

        return np.array(ood_points)

    def make_data(self, init_conds: np.ndarray, timesteps: int, control=None, noisy=False):
        n = init_conds.shape[0]
        self.trajectories = np.zeros((n, timesteps, self._embed_dim))

        for i in range(n):
            self.system.ic = init_conds[i]
            trajectory = self.system.make_trajectory(timesteps, resample=True)
            if trajectory.shape[0] < timesteps:
                trajectory = self.system.make_trajectory(timesteps, resample=False)
            assert trajectory.shape == (timesteps, self._embed_dim)
            self.trajectories[i] = trajectory

        return self.trajectories

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2)

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self._embed_dim