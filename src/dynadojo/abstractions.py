"""
This module contains abstract base classes for systems and algorithms.
"""
from abc import ABC, abstractmethod

import numpy as np


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
