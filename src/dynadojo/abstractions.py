"""
This module contains abstract base classes for systems, algorithms, and challenges.
"""
from abc import ABC, abstractmethod
from functools import cache, cached_property
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging
import itertools

class AbstractAlgorithm(ABC):
    """Base class for all algorithms. Your algorithms should subclass this class."""
    def __init__(self, embed_dim: int, timesteps: int, max_control_cost: float, seed: int | None = None, **kwargs):
        """
        Initialize the class.

        Parameters
        ----------
        embed_dim : int
            Embedded dimension of the dynamics.
        timesteps : int
            Timesteps per training trajectory (per action horizon).
        max_control_cost : float
            Maximum control cost per trajectory (per action horizon).
        **kwargs
            Additional keyword arguments.

        Note
        ------
        Timesteps refer to the timesteps of the training data, NOT the length of the predicted trajectories. The length
        of the predicted trajectories is specified in an argument provided in the :func:`~predict` method.
        """
        self._embed_dim = embed_dim
        # NOTE: this is the timesteps of the training data; NOT the predicted trajectories
        self._timesteps = timesteps
        self._max_control_cost = max_control_cost
        self._seed = seed

    @property
    def embed_dim(self):
        """The embedded dimension of the dynamics."""
        return self._embed_dim

    @property
    def timesteps(self):
        """The timesteps per training trajectory."""
        return self._timesteps

    @property
    def max_control_cost(self):
        """The maximum control cost."""
        return self._max_control_cost

    @property
    def seed(self):
        """The random seed for the algorithm."""
        return self._seed

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs) -> None:
        """
        Fits the algorithm on a tensor of trajectories.

        Parameters
        ----------
        x : np.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        **kwargs
            Additional keyword arguments.
        """
        raise NotImplementedError

    def act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Determines the control for each action horizon.
        control.

        Parameters
        ----------
        x : numpy.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        numpy.ndarray
            (n, timesteps, embed_dim) controls tensor.
        """
        return np.zeros_like(x)

    @abstractmethod
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        """
        Predict how initial conditions matrix evolves over a given number of timesteps.

        Note
        ------
        The timesteps argument can differ from the ._timesteps attribute. This allows algorithms to train on a dataset
        of a given size and then predict trajectories of arbitrary lengths.

        Note
        ------
        The first coordinate of each trajectory should match the initial condition x0.

        Parameters
        -----------
        x0 : np.ndarray
            (n, embed_dim) initial conditions matrix
        timesteps : int
            timesteps per predicted trajectory
        **kwargs
            Additional keyword arguments.

        Returns
        ----------
        np.ndarray
            (n, timesteps, embed_dim) trajectories tensor
        """
        raise NotImplementedError


class AbstractSystem(ABC):
    """Base class for all systems. Your systems should subclass this class."""
    def __init__(self, latent_dim, embed_dim, seed: int | None, **kwargs):
        """
        Initialize the class.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        embed_dim : int
            Embedded dimension of the system.
        seed : int or None, optional
            Seed for random number generation.
        **kwargs
            Additional keyword arguments.
        """
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self._seed = seed

    @property
    def latent_dim(self):
        """The latent dimension for the system."""
        return self._latent_dim

    @property
    def embed_dim(self):
        """The embedded dimension for the system."""
        return self._embed_dim

    @latent_dim.setter
    def latent_dim(self, value):
        """
        Set the latent dimension for the system.

        Notes
        -----
        The reason we use properties and setter methods for ._latent_dim and ._embed_dim is to allow systems to maintain
        information through parameter shifts. See `LDSystem` in `./systems/lds.py` for a principled usage example of the
        setter methods.
        """
        self._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        """
        Set the embedded dimension for the system.

        Notes
        -----
        The reason we use properties and setter methods for ._latent_dim and ._embed_dim is to allow systems to maintain
        information through parameter shifts. See `LDSystem` in `./systems/lds.py` for a principled usage example of the
        setter methods.
        """
        self._embed_dim = value

    @property
    def seed(self):
        """The random seed for the system."""
        return self._seed

    @abstractmethod
    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """
        Generate initial conditions..

        Note
        ------
        Systems developers determine what counts as in vs out-of-distribution. DynaDojo doesn't provide
        any verification that this distinction makes sense or even exists. See :class:`LDSystem` for a principled example.

        Parameters
        ----------
        n : int
            Number of initial conditions.
        in_dist : bool, optional
            If True, generate in-distribution initial conditions. Defaults to True.
            If False, generate out-of-distribution initial conditions.

        Returns
        -------
        numpy.ndarray
            (n, embed_dim) Initial conditions matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        """
        Makes trajectories from initial conditions.

        Parameters
        ----------
        init_conds : numpy.ndarray
            (n, embed_dim) Initial conditions matrix.
        control : numpy.ndarray
            (n, timesteps, embed_dim) Controls tensor.
        timesteps : int
            Timesteps per training trajectory (per action horizon).
        noisy : bool, optional
            If True, add noise to trajectories. Defaults to False. If False, no noise is added.

        Returns
        -------
        numpy.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def calc_error(self, x, y) -> float:
        """
        Calculates the error between two tensors of trajectories.

        Parameters
        ----------
        x : numpy.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        y : numpy.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.

        Returns
        -------
        float
            The error between x and y.
        """
        raise NotImplementedError

    @abstractmethod
    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        """
        Calculate the control cost for each control trajectory (i.e., calculates the costs for every
        control matrix, not for the whole tensor).

        Parameters
        ----------
        control : numpy.ndarray
            (n, timesteps, embed_dim) Controls tensor.

        Returns
        -------
        numpy.ndarray
            (n,) Control costs vector.
        """
        raise NotImplementedError


class AbstractChallenge(ABC):
    """
    Abstract base class for challenges. A run is a setting of parameters defining a system and algorithm. Each trial of the run (specified by system and algorithm seeds) is a separate job. A job is a specific trial of any run, specified by system and algorithm seeds. 


    Parameters
    ----------
    sweep_params : dict[str, List[Any]]
        Lists of parameters to sweep over to create specific runs.
    """

    def __init__(self,
        sweep_params: dict[str, list[Any]],        
    ):
        self.sweep_params = sweep_params
 

    @cached_property
    def run_configs(self):
        """
        Generates a list of configurations for runs by slicing aligned elements of sweep parameters.

        This function creates a list of dictionaries, where each dictionary represents a configuration
        for a run. It pairs the nth elements of each parameter list in `self.sweep_params` to form 
        each configuration. It assumes that all parameter lists in `self.sweep_params` are of the same 
        length.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of configurations for runs.

        Example
        -------
        If `self.sweep_params` is set to
        {
            'a': [1, 2, 3],
            'b': [8, 9, 10]
        }
        `run_configs` will contain configurations for 3 runs:
        [
            {'a': 1, 'b': 8}, #run 1
            {'a': 2, 'b': 9}, #run 2
            {'a': 3, 'b': 10} #run 3
        ]
        """
        param_list_lengths = set(map(len, self.sweep_params.values()))
        assert len(param_list_lengths)==1, "All sweep param lists must be the same length"
        param_list_length = param_list_lengths.pop()
        configs = [ dict([(k, v[i]) for k,v in self.sweep_params.items()]) for i in range(param_list_length)]
        return configs
        

    @cache
    def get_num_jobs(self, trials):
        """
        Calculate the total number of runs based on the number of trials and run configurations.

        Parameters
        ----------
        trials : int
            Number of trials for each run configuration.

        Returns
        -------
        int
            Total number of runs.
        """
        return len(self.run_configs) * trials

    def create_job_configs(
        self,
        trials: int = 1, 
        seed: int|None = None
        ):
        """
        Generates a job configurations by generating seeds for each trial of each run. 

        Parameters
        ----------
        trials : int, optional
            Number of trials for each parameter configuration.
        seed : int | None, optional
            Seed for random number generation, if applicable.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each containing parameters for a run.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        system_seeds = rng.integers(0, 2 ** 32, size = self.get_num_jobs(trials))
        algo_seeds = rng.integers(0, 2 ** 32, size = self.get_num_jobs(trials))

        # Creating all job configs
        job_configs = zip(itertools.product(range(1, trials+1), self.run_configs), system_seeds, algo_seeds)
        # Flattening job configs
        job_configs = [{ 
                        "system_seed": system_seed,
                        "algo_seed" : algo_seed,
                        "trial": trial, 
                        **run_config
                    } for (trial, run_config), system_seed, algo_seed in job_configs]
        # Make ids for each job (for easy filtering)
        job_configs = [{ 
                        "job_id": id,
                        **job_config
                    } for (id, job_config) in enumerate(job_configs)]
        return job_configs
        

    def _filter_jobs(self,
        jobs: list[dict[str, Any]],
        filters: list[dict[str, Any]] | None = None
    ):
        """
        Filters the generated job configurations based on matching any partial job (or run) configuration included in filters.

        Parameters
        ----------
        runs : List[Dict[str, Any]]
            A list of dictionaries, each representing parameters for a specific job.
        filters : List[Dict[str, Any]], optional
            A list of partial job (or run) configurations. Each filter is a dictionary representing a subset of job configuration parameters. A job is included in the final list if its configuration matches any of these partial configurations. The matching is performed such that a job configuration is considered a match if it includes all key-value pairs present in any filter configuration.


        Returns
        -------
        List[Dict[str, Any]]
            A list of job configurations that match at least one filter.
        """
        if not filters:
            return jobs

        def deep_dict_subset_match(d1, d2):
            """Check if d2 is a deep subset of d1 with matching values for shared keys."""
            # Base case: If d2 is empty, it matches d1
            if not d2:
                return True

            # If both d1 and d2 are not dictionaries, compare them directly
            if not isinstance(d1, dict) or not isinstance(d2, dict):
                return d1 == d2

            # Recursive case: check each key in d2
            for key in d2:
                # If key is not in d1 or the values don't match, return False
                if key not in d1 or not deep_dict_subset_match(d1[key], d2[key]):
                    return False

            # If all checks pass, return True
            return True
    

        def filter(job):
            for filter in filters:
                if deep_dict_subset_match(job, filter):
                    return job
            return None
                    
        filtered_jobs = Parallel(n_jobs=-1, timeout=1e6)(delayed(filter)(job) for job in jobs)
        filtered_jobs = [job for job in filtered_jobs if job is not None]
        return filtered_jobs

    

    @abstractmethod
    def execute_job(self, 
                    job_id : int,
                    trial : int = None,
                    system_seed : int = 0, 
                    algo_seed : int = 0,
                    **kwargs
                    ):

        """
        Executes a single, independent trial of a run. (A run is a setting of parameters defining a system and algorithm.)

        This method should be implemented by subclasses to define the specific actions of a single trial of a challenge run. Each execution should be self-contained and should not depend on mutable shared state to allow for parallelization across different CPUs or nodes. For a set of given seeds, this method should be deterministic so that results are reproducible. 

        Parameters
        ----------
        run_id : int
            Unique identifier.
        trial : int | None, optional
            Trial number.
        system_seed : int | None, optional
            Seed for system initialization.
        algo_seed : int | None, optional
            Seed for algorithm initialization.
        **kwargs : dict
            Additional keyword arguments. `AbstractChallenge.evaluate` will supply any parameters defined in `sweep_params` along with additional `kwargs`.

        Returns
        -------
        pandas.DataFrame
            A DataFrame where each row is the result of an algorithm trained and evaluated on a single system for a given training set size.

        Note
        ----
        If the cost of generating data from a system is high, you may not wish to parallelize over training set sizes. See FixedComplexity for an example. In general, you should take caution when sweeping over training set size. 
        """
        raise NotImplementedError
        
    def evaluate(
        self, 
        seed: int|None = None,
        trials: int = 1,
        num_parallel_cpu=-1,
        jobs_filter: list[dict[str, Any]] | None = None,
        **kwargs
        ):
        """
        Orchestrates the evaluation process of the challenge.

        Parameters
        ----------
        seed : int | None, optional
            Seed to initialize random number generation for seeding systems and algorithms.
        trials : int, optional
            Number of trials to run.
        jobs_filter : list[dict[str, Any]] | None, optional
            Specifies which job to evaluate. Defaults to None, which evaluates all jobs.
        **kwargs : dict
            Additional keyword arguments get passed down to self.execute_job
        """
        
        jobs = self.create_job_configs(trials, seed)
        filtered_jobs = self._filter_jobs(jobs, jobs_filter)

        if num_parallel_cpu == 0:
            logging.info(f"Running systems sequentially. {num_parallel_cpu=}")
            data = []
            for job in filtered_jobs:
                data.append(
                    self.execute_job(**kwargs, **job)
                )

        else:
            logging.warning(f"Running systems in parallel. {num_parallel_cpu=}")
            # Run systems in parallel
            data = Parallel(n_jobs=num_parallel_cpu, timeout=1e6)(
                delayed(self.execute_job)(**kwargs, **job) for job in filtered_jobs)

        if data:
            data = pd.concat(data)
        return data



