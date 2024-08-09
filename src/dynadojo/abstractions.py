"""

This module contains abstract base classes for systems, algorithms, and challenges.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cache, cached_property
from multiprocessing import Manager, Process
import os
from typing import Any

import numpy as np

import pandas as pd
from joblib import Parallel, delayed
import logging
import itertools
from .utils.utils import save_to_csv

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
    An abstract base class designed to facilitate the reproducible execution of embarrassingly parallel workloads (particularly when used in conjuction with our Experiment CLI). 
    
    This class serves as a foundation for defining 'challenges', where a challenge is a specific task or set of tasks that can be executed in parallel. Challenges typically involve repeating a task for multiple trials or across a range of parameters (sweep parameters). This class is very abstractly defined in order to support a variety of challenges to be added. If you are interested in creating a challenge, please open an issue and tag @carynbear or @mkanwal for help.
    
    To use this class, subclass it and define the specific behavior in the `execute_job` method. The `evaluate` method orchestrates the execution of the challenge by running each job, which is a single trial of a task for a specific set of parameters.

    Methods
    -------
    __init__(self, sweep_params):
        Initializes the challenge with the given sweep parameters.

    base_configs(self):
        Computes a base set of configurations for jobs by combining elements of sweep parameters. Each configuration represents a unique combination of parameter values. This method is fundamental in generating jobs for each trial.

    get_num_jobs(self, trials):
        

    create_job_configs(self, trials, seed):
        Generates configurations for each job by creating unique seeds for each trial of each base configuration. This method prepares the jobs that will be executed in the challenge.

    execute_job(self, job_id, trial, system_seed, algo_seed, **kwargs):
        Abstract method to be implemented in subclasses. Defines how to execute a single trial for a given job configuration. It should be deterministic and self-contained to facilitate parallel execution.

    evaluate(self, seed, trials, num_parallel_cpu, jobs_filter, csv_output_path, **kwargs):
        Orchestrates the overall evaluation process for the challenge. It manages the creation and execution of jobs and can handle both sequential and parallel processing. Optionally, results can be saved to a CSV file.

    Examples
    --------
    Subclasses of `AbstractChallenge` should implement `execute_job` to define the specific task to be performed. For instance, in a machine learning context, `execute_job` might involve training and evaluating a model with specific hyperparameters and data. The `evaluate` method can then be used to run the challenge across different hyperparameter configurations and trials.

    See the examples in `./challenges/` for guidance on how to implement and use subclasses of `AbstractChallenge`.
    """

    def __init__(self,
        sweep_params: dict[str, list[Any]],        
    ):
        """
        Initializes the challenge with the given sweep parameters.

        Parameters
        ----------
        sweep_params : dict[str, list[Any]]
            A dictionary where each key is a parameter name and each value is a list of values to sweep over. This dictionary defines the parameter space over which the challenge will be executed. All parameter lists in `self.sweep_params` must be the same length.

        """
        param_list_lengths = set(map(len, sweep_params.values()))
        assert len(param_list_lengths)==1, "All sweep param lists must be the same length"
        self.sweep_params = sweep_params
 

    @cached_property
    def base_configs(self):
        """
        Computes a base set of configurations for jobs by iterating over elements of sweep parameters. Each configuration represents a single value for each of the sweep parameters. Later we generate jobs by adding seeds to base_configs for each trial.

        This function creates a list of dictionaries. It matches up the nth elements of each parameter list in `self.sweep_params` to form 
        each configuration. 
        
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
        `base_configs` will contain configurations for 3 runs:
        [
            {'a': 1, 'b': 8},
            {'a': 2, 'b': 9},
            {'a': 3, 'b': 10} 
        ]
        """
        param_list_lengths = set(map(len, self.sweep_params.values()))
        param_list_length = param_list_lengths.pop()
        configs = [ dict([(k, v[i]) for k,v in self.sweep_params.items()]) for i in range(param_list_length)]
        return configs
        

    @cache
    def get_num_jobs(self, trials):
        """
        Calculates the total number of jobs based on the number of trials and the base configurations generated from sweep parameters.

        Parameters
        ----------
        trials : int
            Number of trials for each base configuration.

        Returns
        -------
        int
            Total number of jobs.
        """
        return len(self.base_configs) * trials

    def create_job_configs(
        self,
        trials: int = 1, 
        seed: int|None = None
        ):
        """
        Generates a job configurations by generating seeds for each trial of each base config. 

        Parameters
        ----------
        trials : int, optional
            Number of trials for each parameter configuration.
        seed : int | None, optional
            Seed for random number generation, if applicable.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each containing parameters for a job.
        """
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        system_seeds = rng.integers(0, 2 ** 32, size = self.get_num_jobs(trials))
        algo_seeds = rng.integers(0, 2 ** 32, size = self.get_num_jobs(trials))

        # Creating all job configs
        job_configs = zip(itertools.product(range(1, trials+1), self.base_configs), system_seeds, algo_seeds)
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

    

    @abstractmethod
    def execute_job(self, 
                    job_id : int,
                    trial : int = None,
                    system_seed : int = 0, 
                    algo_seed : int = 0,
                    **kwargs
                    ):

        """
        Abstract method to be implemented in subclasses. Defines how to execute a single trial for a given job configuration. It should be self-contained to facilitate parallel execution and determinister so that results are reproducible.

        This method should return a pandas DataFrame where each row is the result of an algorithm trained and evaluated on a single system for a given training set size.

        Note
        ----
        A job is very abstractly defined, open an issue and tag @carynbear or @mkanwal for help. You can think of it as a trial for a single unit of computation that you wish to distribute. Sometimes we can do a single trial of multiple runs (see FixedComplexity) in special cases where the system is the same for different training set sizes.
        

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
        
        """
        raise NotImplementedError
        
    def evaluate(
        self, 
        seed: int|None = None,
        trials: int = 1,
        num_parallel_cpu=-1,
        jobs_filter: list[int] | None = None,
        csv_output_path: str | None = None,
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
        jobs_filter : list[int] | None, optional
            Specifies which job ids to evaluate. Defaults to None, which evaluates all jobs.
        csv_output_path : str | None, optional
            Path to save results. Will add to file if exists. Defaults to None, which does not save results.
        **kwargs : dict
            Additional keyword arguments get passed down to self.execute_job
        """
        # Add a plots subdirectory to the output directory and to kwargs 
        #TODO HACKY INTERMEDIATE PLOTTING should fix
        if csv_output_path:
            intermediate_dir = os.path.join(os.path.dirname(csv_output_path), "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            kwargs["intermediate_dir"] = intermediate_dir
        
        jobs = self.create_job_configs(trials, seed)
        logging.info(f"Created {len(jobs)} jobs")
        filtered_jobs = [job for job in jobs if job["job_id"] in jobs_filter] if jobs_filter else jobs

        if not num_parallel_cpu or num_parallel_cpu == 0 or num_parallel_cpu == 1:
            logging.info(f"Running systems sequentially. {num_parallel_cpu=}")
            data = []
            for job in filtered_jobs:
                data_job = self.execute_job(**kwargs, **job)
                data.append(data_job)
                if csv_output_path:
                    data_job.to_csv(csv_output_path, mode='a', index=False, header=not os.path.exists(csv_output_path))
        else:
            logging.warning(f"Running systems in parallel. {num_parallel_cpu=}")
            if csv_output_path: # save to csv in parallel
                
                def process(**kwargs):
                    data_job = self.execute_job(**kwargs)
                    q.put(data_job)
                    return data_job
                
                m = Manager()
                q = m.Queue()
                p = Process(target=save_to_csv, args=(q, csv_output_path))
                p.start()
                data = Parallel(n_jobs=num_parallel_cpu, timeout=1e6)(
                    delayed(process)(**kwargs, **job) for job in filtered_jobs)
                q.put(None)
                p.join()
            
            else:
                data = Parallel(n_jobs=num_parallel_cpu, timeout=1e6)(
                    delayed(self.execute_job)(**kwargs, **job) for job in filtered_jobs)

        if data:
            data = pd.concat(data)
        return data
