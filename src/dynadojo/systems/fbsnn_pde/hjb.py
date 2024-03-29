"""
Hamilton-Jacobi-Bellman PDE
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from ..utils.fbsnn import FBSNNSystem

"""
Adapted from Maziar Raissi, https://github.com/maziarraissi/FBSNNs

Complexity
-------
self.latent_dim, controls dimension of the systems
"""


class HJBSystem(FBSNNSystem):
    """
    Hamilton-Jacobi-Bellman PDE
    
    Adapted from Maziar Raissi, https://github.com/maziarraissi/FBSNNs

    Example
    ---------
    >>> from dynadojo.systems.fbsnn import HJBSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.lv import plot
    >>> latent_dim = 20
    >>> embed_dim = 1
    >>> timesteps = 15
    >>> n = 10
    >>> system = SystemChecker(HJBSystem(dim, embed_dim))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], timesteps=timesteps, target_dim=1, max_lines=100)

    .. image:: ../_images/hjb.png

    """
    def __init__(self, latent_dim=1, embed_dim=1,
                 IND_range=(3.0, 4.0),
                 OOD_range=(4.0, 5.0),
                 layers=None,
                 T=1.0,
                 MC=10**5,
                 noise_scale=0.05,
                 seed=None):
        
        """
        Initializes a HJBSystem instance.

        Parameters
        -------------
        latent_dim : int
            Dimension of the system
        embed_dim : int
            Must be 1
        IND_range : tuple
            In-distribution range of starting trajectory values.
        OOD_range : tuple
            Out-of-distribution range of starting trajectory values.
        layers : [int]
            Neural network configuration architecture
        T : float
            Time to maturity, or the final times to simulate until
        MC : int
            Number of Monte-Carlo simulations to run
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, layers, T, seed)
        self.MC = MC

    def _phi_tf(self, t, X, Y, Z):  # N x 1, N x latent_dim, N x 1, N x latent_dim
        return tf.reduce_sum(Z ** 2, 1, keepdims=True)  # N x 1

    def _g_tf(self, X):  # N x latent_dim
        return tf.math.log(0.5 + 0.5 * tf.reduce_sum(X ** 2, 1, keepdims=True))  # N x 1

    def _mu_tf(self, t, X, Y, Z):  # N x 1, N x latent_dim, N x 1, N x latent_dim
        return super()._mu_tf(t, X, Y, Z)  # N x latent_dim

    def _sigma_tf(self, t, X, Y):  # N x 1, N x latent_dim, N x 1
        return tf.sqrt(2.0) * super()._sigma_tf(t, X, Y)  # N x latent_dim x latent_dim


    def _unsolve_target(self, target, T, new_dim):
        def g(X):  # MC x NC x D
            return np.log(0.5 + 0.5 * np.sum(X ** 2, axis=2, keepdims=True))

        NC = 1
        W = np.random.normal(size=(self.MC, NC, self.latent_dim)) 

        class OptimizationEarlyStop(Exception):
            pass

        def objective(Y):
            self.Y = Y
            sol = -np.log(np.mean(np.exp(-g(Y + np.sqrt(2.0 * np.abs(T)) * W))))
            return np.abs(sol - target)

        # Define a callback function
        def callback(Y):
            # Check if current solution is within an acceptable range
            if objective(Y) < acceptable_threshold:
                raise OptimizationEarlyStop

        # Constraint to ensure all numbers are positive
        constraints = [{'type': 'ineq', 'fun': lambda Y: Y}]

        # Define an acceptable threshold for early stopping
        acceptable_threshold = 0.02  # Adjust this value based on your requirement

        initial_guess = np.random.rand(new_dim)
        self.Y = initial_guess
        try:
            result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints, callback=callback)
            return result.x
        except OptimizationEarlyStop:
            return self.Y


    def _solve(self, t, X, T, U):  # (N) x 1, (N) x latent_dim, T

        def g(X):  # MC x NC x D
            return np.log(0.5 + 0.5 * np.sum(X ** 2, axis=2, keepdims=True))  # MC x N x 1

        NC = t.shape[0]
        W = np.random.normal(size=(self.MC, NC, self.latent_dim))  # MC x NC x D

        return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0 * np.abs(T - t)) * W + U)), axis=0))
