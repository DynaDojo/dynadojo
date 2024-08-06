import dysts.maps
import numpy as np
import scipy as sp
import os
import dysts
import importlib
import json
import sys  
from dysts.utils import generate_ic_ensemble
from dysts.analysis import sample_initial_conditions
from dysts.base import DynMap

from dynadojo.abstractions import AbstractSystem

class GilpinMapsSystem(AbstractSystem):

    """
    Gilpin Maps System for modeling discrete maps using systems from the dysts library.
    Attributes
    ----------
    base_path : str
        Base path for the dysts library.
    json_file_path : str
        Path to the JSON file containing chaotic attractors data.
    Example
    -------
    >>> from dynadojo.systems.gilpin_maps import GilpinMapsSystem
    >>> latent_dim = 2
    >>> embed_dim = 2
    >>> system_name = 'Ikeda'
    >>> system = GilpinMapsSystem(latent_dim, embed_dim, system_name)
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
    check_systems_availability(cls) -> list
        Class method that loads systems data and returns the list of available systems, excluding missing systems.
    """

    base_path = os.path.dirname(dysts.__file__)
    json_file_path = os.path.join(base_path, 'data', 'discrete_maps.json')

    with open(json_file_path, 'r') as file:
        system_data = json.load(file)
    all_systems = list(system_data.keys())
    unavailable_systems = ["Tinkerbell"] #any unsupported systems because of implemntation should be added here
    available_systems = []

    def __init__(self, latent_dim, embed_dim, system_name: str, seed=None):  

        """
        Initialize the GilpinMapssSystem class.
        Parameters
        ----------
        system_name : str
            The name of the system to be used.
        latent_dim : int
            Dimension of the latent space. 
        embed_dim : int
            Embedding dimension of the system. Fixed to Gilpin's set dimensionality for the particular system.
        pts_per_period: int
            For reasampled trajectories, the number of points per period. Default is 100.  Do I add this?
        seed : int or None, optional
            Seed for random number generation. Default is None.
        """
        super().__init__(latent_dim, embed_dim, seed=seed)
        self._system_name = system_name 
        self.trajectories = None

        if self._system_name in self.unavailable_systems:
            raise ValueError(f"Unsupported system {self._system_name}")

        try:
            module = importlib.import_module("dysts.maps") 
            systemClass = getattr(module, self._system_name) 
            self.system = systemClass()
        except (ImportError, AttributeError) as e:
            print(f"Error loading system {self._system_name}: {e}")
            return 
        
        data = self.system._load_data()
        self._embed_dim = len(data.get("initial_conditions")) #Fixed deminsion
        self._latent_dim = self._embed_dim

    @classmethod
    def check_systems_availability(cls):

        """
        Check the availability of system names in the `dysts.maps` module.
        """
        
        module = importlib.import_module("dysts.maps")
        for system_name in cls.all_systems:
            try:
                getattr(module, system_name) 
                if (system_name not in cls.available_systems) and (system_name not in cls.unavailable_systems):
                    cls.available_systems.append(system_name)
            except (ImportError, AttributeError):
                if (system_name not in cls.unavailable_systems):
                    cls.unavailable_systems.append(system_name)

    def make_init_conds(self, n: int, in_dist:bool) -> np.ndarray:

        """
        Generate initial conditions for the system.

        This method generates initial conditions for the system based on whether 
        the initial conditions are within distribution (`in_dist`) or out of distribution.

        Parameters:
        ----------
        n : int
            The number of initial conditions to generate.
        in_dist : bool
            Indicates whether to generate initial conditions within the distribution (True) 
            or out of distribution (False).

        Returns:
        -------
        np.ndarray
            An array of initial conditions.

        If `in_dist` is True:
            Generates initial conditions by perturbing the initial conditions of the system's model.
            The perturbations are controlled by a fraction parameter, and the resulting initial 
            conditions are clipped to the range of values in a reference trajectory.

        If `in_dist` is False:
            Generates out-of-distribution initial conditions using Singular Value Decomposition (SVD) 
            on a trajectory of the system, retaining components that explain up to 90% of the variance.
            The remaining components are used to create projections that are scaled and shifted 
            to produce the out-of-distribution initial conditions. (PCA).

        Notes:
        -----
        - The `generate_init_conds` function, defined within this method, is adapted from Gilpin's generate_ic_ensemble method
        - TODO: currently this implemntation avoids generating unstable trajectories(INFs and NaNs) using the reference trajectory 
                to clipping within-distribution initial conditions. Tinkerbell system still unsupported. 
        """

        if(in_dist):
            def generate_init_conds(
                model,
                n_samples,
                frac_perturb_param=0.1,
                random_state=0,
                ):
                np.random.seed(random_state)
                all_samples = []
                ic = model.ic
                for i in range(n_samples):
                    ic_perturb = 1 + frac_perturb_param * (2 * np.random.random(len(ic)) - 1)
                    ic_prime = ic * ic_perturb
                    all_samples.append(ic_prime)
                return all_samples
            
            trajectories = np.array(generate_init_conds(self.system, n, random_state=self._seed))
            ref_traj = self.system.make_trajectory(100)
            min_value = np.min(ref_traj)
            max_value = np.max(ref_traj)
            trajectories= np.clip(trajectories, min_value,max_value)
            return trajectories
        
        x = self.system.make_trajectory(10000, resample = True)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        mean = np.mean(x, axis=0)
        x_centered = x - mean
        if x_centered.ndim == 1:
            x_centered = x_centered.reshape(-1, 1)
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
        
    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False):

        """
        Generate system trajectories based on initial conditions and control inputs.

        Parameters:
        ----------
        init_conds : np.ndarray
            array of initial conditions with shape (n, d), `n`: number of 
            initial conditions, 'd`: dimensionality initial condition space.
        control : np.ndarray
            array representing control inputs (currently not utilized in the method).
        timesteps : int
             number of timesteps to generate the trajectory for.
        noisy : bool, optional
            indicate whether to add noise to the generated trajectories (default is False).

        Returns:
        -------
        np.ndarray
            An array of generated trajectories with shape (n, timesteps, embed_dim).
        """

        n = init_conds.shape[0]
        self.trajectories = np.zeros((n, timesteps, self._embed_dim)) 
        for i in range (n):
                self.system.ic = init_conds[i] 
                trajectory = self.system.make_trajectory(timesteps)
                if trajectory.ndim == 1:
                    trajectory = trajectory[:, np.newaxis]  #reshaping for 1D systems
                self.trajectories[i] = trajectory
        return self.trajectories
    
        
    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)



    