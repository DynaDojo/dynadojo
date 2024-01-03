"""
Simple System
==============
"""
# import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group

from ...abstractions import AbstractSystem


class SimpleSystem(AbstractSystem):
    """An extension of :class:`AbstractSystem` with some useful methods."""
    def __init__(self,
                 latent_dim=2,
                 embed_dim=2,
                 seed=None,
                 embedder_sv_range=(0.1, 1),
                 controller_sv_range=(0.1, 1),
                 IND_range=(0, 10),
                 OOD_range=(-10, 0),
                 noise_scale=0.01,
                 t_range=(0, 1),
                 ):
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
        embedder_sv_range : tuple
            The singular value range for the embedder matrix. Singular values are non-negative by convention.
            The singular values should exclude 0 to ensure the embedder is invertible.
        controller_sv_range : tuple
            The singular value range for the controller matrix.
        IND_range : tuple
            The in-distribution initial condition range.
        OOD_Range : tuple
            The out-of-distribution initial condition range.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(latent_dim, embed_dim, seed)

        self._t_range = t_range

        self.IND_range = IND_range
        self.OOD_range = OOD_range

        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed=self._seed)

        self._embedder_sv_range = embedder_sv_range
        self._controller_sv_range = controller_sv_range
        self._embedder = None
        self._controller = None
        self._update_embedder_and_controller()

    @property
    def embedder(self):
        """
        The embedder matrix. An invertible map from the latent space to the embedding space.
        """
        return self._embedder

    @property
    def controller(self):
        r"""The controller matrix. For example, in a system :math:`\dot{x} = Ax + Bu`, the controller is :math:`B`."""
        return self._controller

    def _update_embedder_and_controller(self):
        self._embedder = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._embedder_sv_range)
        self._controller = self._sv_to_matrix(self.latent_dim, self.embed_dim, self._controller_sv_range)

    @AbstractSystem.embed_dim.setter
    def embed_dim(self, value):
        self._embed_dim = value
        self._update_embedder_and_controller()

    @AbstractSystem.latent_dim.setter
    def latent_dim(self, value):
        self._latent_dim = value
        self._update_embedder_and_controller()

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        """
        Uniformly samples embedded-dimensional points from an inside or outside distribution

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
        init_cond_range = self.IND_range if in_dist else self.OOD_range
        return self._rng.uniform(*init_cond_range, (n, self.embed_dim))

    def calc_error(self, x, y) -> float:
        """Returns the MSE error normalized by the embedded dimension."""
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        """Calculates the L2 norm / dimension of every vector in the control"""
        return np.linalg.norm(control, axis=(1, 2), ord=2) / self.embed_dim

    def calc_dynamics(self, t, x):
        """
        Calculates the dynamics for the system. Your class must implement this.
        """
        raise NotImplementedError

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        r"""
        Uses the :func:`~calc_dynamics` method to generate data. Mathematically, data is generated like :math:`\dot{x} = f(x) + Bu`.
        Where :math:`f(x)` is given by :func:`~calc_dynamics`.

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
        data = []
        init_conds = init_conds @ np.linalg.pinv(self.embedder)
        time = np.linspace(self._t_range[0], self._t_range[1], num=timesteps)

        def dynamics(t, x, u):
            i = np.argmin(np.abs(t - time))
            dx = self.calc_dynamics(t, x) + self.controller @ u[i]
            if noisy:
                dx += self._rng.normal(scale=self._noise_scale, size=self.latent_dim)
            return dx

        for x0, u in zip(init_conds, control):
            sol = solve_ivp(dynamics, t_span=[self._t_range[0], self._t_range[1]], y0=x0, t_eval=time,
                            dense_output=True, args=(u,))
            data.append(sol.y)

        data = np.transpose(np.array(data), axes=(0, 2, 1)) @ self.embedder
        return data

    def _sv_to_matrix(self, m, n, sv_range):
        U = ortho_group.rvs(m, random_state=self._seed)
        sigma = np.eye(m, n) * self._rng.uniform(*sv_range, size=n)
        V = ortho_group.rvs(n, random_state=self._seed)
        N = U @ sigma @ V
        return N
