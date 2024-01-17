"""
Black-Scholes-Barenblatt PDE
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

class BSBSystem(FBSNNSystem):
    def __init__(self, latent_dim=1, embed_dim=1,
                 noise_scale=0.001,
                 IND_range=(10, 20),
                 OOD_range=(20, 30),
                 layers=None,
                 T=1.0,
                 r=0.05,
                 sigma_max=0.04,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, layers, T, seed)
        self.r = r
        self.sigma_max = sigma_max

    def _phi_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return 0.05*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # N x 1
    
    def _g_tf(self, X): # N x latent_dim
        return tf.reduce_sum(X**2, 1, keepdims = True) # N x 1

    def _mu_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return super()._mu_tf(t, X, Y, Z) # N x latent_dim
        
    def _sigma_tf(self, t, X, Y): # N x 1, N x latent_dim, N x 1
        return 0.4*tf.linalg.diag(X) # N x latent_dim x latent_dim
    
    def _unsolve_target(self, target, T, new_dim):
        def objective(Y):
            return np.abs((np.exp((self.r + self.sigma_max**2) * T) * np.sum(Y**2)) - target)

        # Constraint to ensure all numbers are positive
        constraints = [{'type': 'ineq', 'fun': lambda Y: Y}]

        initial_guess = np.random.rand(new_dim)
        result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)

        if result.success:
            return result.x
        else:
            raise ValueError(f'Could not expand initial conditions to ${self.latent_dim}')

    def _solve(self, t, X, T, U): # (N) x 1, (N) x latent_dim, T
        
        return (np.exp((self.r + self.sigma_max**2)*(T - t)))*np.sum(X**2+U, 1, keepdims = True) # (N) x 1
            

