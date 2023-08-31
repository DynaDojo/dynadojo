import numpy as np
import tensorflow as tf
from ..utils import FBSNNSystem

'''
Black-Scholes-Barenblatt PDE adapted from Naziar Raissi, https://github.com/maziarraissi/FBSNNs

Complexity
-------
self.latent_dim, controls dimension of the systems

'''
class BSBSystem(FBSNNSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.01,
                 IND_range=(0, 0.5),
                 OOD_range=(0.5, 1),
                 layers=None,
                 T=1.0,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, layers, T, seed)

    def _phi_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return 0.05*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # N x 1
    
    def _g_tf(self, X): # N x latent_dim
        return tf.reduce_sum(X**2, 1, keepdims = True) # N x 1

    def _mu_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return super()._mu_tf(t, X, Y, Z) # N x latent_dim
        
    def _sigma_tf(self, t, X, Y): # N x 1, N x latent_dim, N x 1
        return 0.4*tf.linalg.diag(X) # N x latent_dim x latent_dim
    
    def _u_exact(self, t, X, T): # (N+1) x 1, (N+1) x latent_dim, T
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max**2)*(T - t))*np.sum(X**2, 1, keepdims = True) # (N+1) x 1

