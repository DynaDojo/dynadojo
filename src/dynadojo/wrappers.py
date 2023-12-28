"""
Wrappers
=========
Wrappers are a convenient way to modify an existing DynaDojo object without having to alter the underlying code directly. Using wrappers will allow you to avoid a lot of boilerplate code and make your project more modular. Wrappers can also be chained to combine their effects. To use the wrappers, use dynadojo.wrappers.make_alg or dynadojo.wrappers.make_system.
"""

from .abstractions import AbstractAlgorithm, AbstractSystem

import numpy as np


class AlgorithmChecker:
    """Wrapper class for algorithms that ensures proper input and output handling."""
    def __init__(self, alg: AbstractAlgorithm):
        self._alg = alg

    @property
    def embed_dim(self):
        return self._alg.embed_dim

    @property
    def timesteps(self):
        return self._alg.timesteps

    @property
    def max_control_cost(self):
        return self._alg.max_control_cost

    @property
    def seed(self):
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
            (n, timesteps, embed_dim) Controls tensor.
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

        Returns
        -------
        None
        """
        assert x.shape == (x.shape[0], self.timesteps, self.embed_dim)
        self._alg.fit(x, **kwargs)

def make_alg(alg):
    alg = AlgorithmChecker(alg)
    return alg

class SystemChecker:
    def __init__(self, system: AbstractSystem):
        self._system = system

    @property
    def latent_dim(self):
        return self._system._latent_dim

    @property
    def embed_dim(self):
        return self._system._embed_dim

    @latent_dim.setter
    def latent_dim(self, value):
        self._system._latent_dim = value

    @embed_dim.setter
    def embed_dim(self, value):
        self._system._embed_dim = value

    def make_init_conds(self, n: int, in_dist=True):
        """
        Wrapper for make_init_conds() called in evaluate(). Verifies initial condition matrix is the right shape.
        You should NOT override this.

        :param n: number of initial conditions
        :param in_dist: Boolean. If True, generate in-distribution initial conditions. Defaults to True. If False,
        generate out-of-distribution initial conditions.
        :return: (n, embed_dim) initial conditions matrix
        """
        init_conds = self._system.make_init_conds(n, in_dist)
        assert init_conds.shape == (n, self.embed_dim)
        return init_conds

    def make_data(self, init_conds: np.ndarray, control: np.ndarray = None, timesteps: int = 1,
                  noisy=False) -> np.ndarray:
        """
        Wraps make_data(). Checks that trajectories tensor has the proper shape. You should NOT override this.

        :param init_conds: (n, embed_dim) initial conditions matrix
        :param control: (n, timesteps, embed_dim) controls tensor
        :param timesteps: timesteps per training trajectory (per action horizon)
        :param noisy: Boolean. If True, add noise to trajectories. Defaults to False. If False, no noise is added.
        :return: (n, timesteps, embed_dim) trajectories tensor
        """
        assert timesteps > 0
        assert init_conds.ndim == 2 and init_conds.shape[1] == self.embed_dim
        n = init_conds.shape[0]
        if control is None:
            control = np.zeros((n, timesteps, self.embed_dim))
        assert control.shape == (n, timesteps, self.embed_dim)
        data = self.make_data(init_conds=init_conds,
                               control=control, timesteps=timesteps, noisy=noisy)
        assert data.shape == (n, timesteps, self.embed_dim)
        return data

    def calc_error(self, x, y) -> float:
        """
        Wraps calc_error. Checks that calc_error is called with properly-shaped x and y.
        Your systems should NOT override this.

        :param x: (n, timesteps, embed_dim) trajectories tensor
        :param y: (n, timesteps, embed_dim) trajectories tensor
        :return: Float. The error between x and y.
        """
        assert x.shape == y.shape
        return self._system.calc_error(x, y)

    def calc_control_cost(self, control: np.ndarray) -> np.ndarray:
        """
        Wraps calc_control_cost(). Your systems should NOT override this.

        :param control: (n, timesteps, embed_dim) controls tensor
        :return: (n,) control costs vector
        """
        assert control.shape[2] == self.embed_dim and control.ndim == 3
        cost = self._system.calc_control_cost(control)
        assert cost.shape == (len(control),)
        return cost

def make_system(system):
    system = SystemChecker(system)
    return system
