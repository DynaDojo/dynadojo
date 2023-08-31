import numpy as np
import tensorflow as tf
from ..utils import FBSNNSystem

'''
Hamilton-Jacobi-Bellman PDE adapted from Maziar Raissi, https://github.com/maziarraissi/FBSNNs

Complexity
-------
self.latent_dim, controls dimension of the systems

'''
class HJBSystem(FBSNNSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.05,
                 IND_range=(0, 1),
                 OOD_range=(1, 2),
                 layers=None,
                 T=1.0,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, layers, T, seed)

    def _phi_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return tf.reduce_sum(Z**2, 1, keepdims = True) # N x 1
    
    def _g_tf(self, X): # N x latent_dim
        return tf.math.log(0.5 + 0.5*tf.reduce_sum(X**2, 1, keepdims = True)) # N x 1

    def _mu_tf(self, t, X, Y, Z): # N x 1, N x latent_dim, N x 1, N x latent_dim
        return super()._mu_tf(t, X, Y, Z) # N x latent_dim
        
    def _sigma_tf(self, t, X, Y): # N x 1, N x latent_dim, N x 1
        return tf.sqrt(2.0)*super()._sigma_tf(t, X, Y) # N x latent_dim x latent_dim
    

    def _u_exact(self, t, X, T, noisy): # (N+1) x 1, (N+1) x latent_dim, T

        def g(X): # MC x NC x D
            return np.log(0.5 + 0.5*np.sum(X**2, axis=2, keepdims=True)) # MC x N x 1

        MC = 10**5
        NC = t.shape[0]
        W = np.random.normal(size=(MC,NC,self.latent_dim)) # MC x NC x D

        if noisy:
                noise = self._rng.normal(
                    0, self.noise_scale)
        else:
            noise = 0
        
        return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W)),axis=0))+noise

      

