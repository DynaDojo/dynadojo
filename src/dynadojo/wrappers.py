"""
Wrappers are a convenient way to modify an existing DynaDojo object without having to alter the underlying code directly. Using wrappers will allow you to avoid a lot of boilerplate code and make your project more modular.
"""
from .abstractions import AbstractAlgorithm, AbstractSystem

import numpy as np


class AlgorithmChecker:
    """
    Wrapper class for algorithms that ensures proper input and output handling.

    Example
    --------
    >>> from dynadojo.baselines.lr import LinearRegression
    >>> AlgorithmChecker(LinearRegression(2, 50, 0))
    <AlgorithmChecker<LinearRegression>>
    """

    def __init__(self, alg: AbstractAlgorithm):
        """Initialize the `AlgorithmChecker` object

        Parameters
        -----------
        alg : AbstractAlgorithm
            Underlying algorithm.
        """
        self._alg = alg

    def __repr__(self):
        return f"<{type(self).__name__}<{type(self._alg).__name__}>>"

    @property
    def embed_dim(self):
        """The embedding dimension of the underlying algorithm."""
        return self._alg.embed_dim

    @property
    def timesteps(self):
        """The number of timesteps for the underlying algorithm."""
        return self._alg.timesteps

    @property
    def max_control_cost(self):
        """The max control cost for the underlying algorithm."""
        return self._alg.max_control_cost

    @property
    def seed(self):
        """The random seed for the underlying algorithm."""
        return self._alg.seed

    def act(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Verifies control tensor is the right shape.

        Parameters
        ----------
        x : np.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            (n, timesteps, embed_dim) controls tensor
        """
        control = self._alg.act(x, **kwargs)
        assert control.shape == x.shape
        return control

    def predict(self, x0: np.ndarray, timesteps, **kwargs) -> np.ndarray:
        """
        Verifies predicted trajectories tensor has the right shape.

        Parameters
        ----------
        x0 : np.ndarray
            (n, embed_dim) Initial conditions matrix.
        timesteps : int
            Timesteps per predicted trajectory (most commonly the same as the system timesteps).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            (n, timesteps, embed_dim) Predicted trajectories tensor.
        """
        pred = self._alg.predict(x0, timesteps, **kwargs)
        n = x0.shape[0]
        assert pred.shape == (n, timesteps, self._alg.embed_dim)
        return pred

    def fit(self, x: np.ndarray, **kwargs) -> None:
        """
        Verifies algorithm trains on properly sized data.

        Parameters
        ----------
        x : np.ndarray
            (n, timesteps, embed_dim) Trajectories tensor.
        **kwargs
            Additional keyword arguments.
        """
        assert x.shape == (x.shape[0], self.timesteps, self.embed_dim)
        self._alg.fit(x, **kwargs)


class SystemChecker:
    """Wrapper class for systems that ensures proper input and output handling."""

    def __init__(self, system: AbstractSystem):
        """
        Initialize the SystemChecker object.

        Parameters
        -----------
        system : AbstractSystem
            The underlying system.

        Example
        ---------
        >>> from dynadojo.systems.lds import LDSystem
        >>> SystemChecker(LDSystem())
        <SystemChecker<LDSystem>>
        """
        self._system = system

    def __repr__(self):
        return f"<{type(self).__name__}<{type(self._system).__name__}>>"

    @property
    def seed(self):
        """The seed for the underlying system."""
        return self._system.seed

    @property
    def latent_dim(self):
        """The latent dimension of the underlying system."""
        return self._system.latent_dim

    @property
    def embed_dim(self):
        """The embedded dimension of the underlying system."""
        return self._system.embed_dim

    @latent_dim.setter
    def latent_dim(self, value):
        """Sets the latent dimension of the underlying system."""
        self._system.latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        """Sets the embedded dimension of the underlying system"""
        self._system.embed_dim = value

    def make_init_conds(self, n: int, in_dist=True):
        """
        Verifies initial condition matrix is the right shape.

        Parameters
        -----------
        n: int
            Number of initial conditions.
        in_dist: bool
            If True, generate in-distribution initial conditions. Defaults to True. If False, generate out-of-distribution initial conditions.

        Returns
        -------
        np.ndarray
            (n, embed_dim) Initial conditions matrix.
        """
        init_conds = self._system.make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    def make_data(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1, dt = 0.05,
                  noisy=False) -> np.ndarray:
        """
        Checks that trajectories tensor has the proper shape.

        Parameters
        -----------
        init_conds: np.ndarray
            (n, embed_dim) initial conditions matrix
        control: np.ndarray
            (n, embed_dim) initial conditions matrix
        timesteps: int
            timesteps per training trajectory (per action horizon)
        noisy: bool
            If True, add noise to trajectories. Defaults to False. If False, no noise is added.

        Returns
        --------
        np.ndarray
            (n, timesteps, embed_dim) trajectories tensor
        """
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim), f"control has shape {control.shape}, but it should be ({n}, {timesteps}, {self.embed_dim})"
        data = self._system.make_data(init_conds=init_conds, control=control, timesteps=timesteps, dt=dt, noisy=noisy)
        assert data.shape == (n, timesteps, self.embed_dim), f"data has shape {data.shape}, but it should be ({n}, {timesteps}, {self.embed_dim})"
        return data

    def calc_error(self, x, y) -> float:
        """
        Checks that calc_error is called with properly-shaped x and y.

        Parameters
        -----------
        x: np.ndarray
            (n, timesteps, embed_dim) trajectories tensor
        y: np.ndarray
            (n, timesteps, embed_dim) trajectories tensor

        Returns
        ---------
        float
            Error between x and y.
        """
        assert x.shape == y.shape
        return self._system.calc_error(x, y)

    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        """
        Wraps calc_control_cost.

        Parameters
        -----------
        control: np.ndarray
            (n, timesteps, embed_dim) control tensor

        Returns
        ---------
        np.ndarray
            (n,) control costs vector
        """
        assert control.shape[2] == self.embed_dim and control.ndim == 3
        cost = self._system.calc_control_cost(control)
        assert cost.shape == (len(control),)
        return cost


if __name__ == "__main__":
    pass
