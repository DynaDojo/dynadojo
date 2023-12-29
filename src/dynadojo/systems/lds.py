"""
Linear Dynamical Systems
===========================
"""
import numpy as np
import scipy as sp

from .utils.simple import SimpleSystem


class LDSystem(SimpleSystem):
    """
    Linear dynamical systems. Implements a system of the form :math:`\dot{x} = Ax + Bu`
    where :math:`Ax` is the drift term of :math:`\dot{x}` and :math:`Bu` is input term of :math:`\dot{x}`.
    """
    def __init__(self,
                 latent_dim: int = 2,
                 embed_dim: int = 2,
                 A_eigval_range=(-1, 1),
                 A_eigvec_range=(-1, 1),
                 **kwargs):
        """
        Parameters
        ----------
        latent_dim : int, optional
            Dimension of the latent space. Defaults to 2. Can set to large values.
        embed_dim : int, optional
            Embedded dimension of the system. Defaults to 2. Can set to large values. Should be at least as large as latent_dim.
        A_eigval_range : tuple, optional
            Range for eigenvalues of the matrix A. Defaults to (-1, 1). Recommended to keep this value between -1 and 1
            for stable systems.
        A_eigvec_range : tuple, optional
            Range for eigenvectors of the matrix A. Defaults to (-1, 1).
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(latent_dim, embed_dim, **kwargs)

        self._A_eigval_range = A_eigval_range
        self._A_eigvec_range = A_eigvec_range
        self.A = self._make_A(latent_dim)

    def _eigenvalues_to_matrix(self, dim, eig_range, eigvec_range):
        # use eigendecomposition to produce a real-valued matrix with target eigenvalues
        if self._rng.uniform() <= 0:
            # 50% chance to have all real eigenvalues
            eigenvalues = self._rng.uniform(*eig_range, size=dim)
            Q = self._rng.uniform(*eigvec_range, size=(dim, dim))
            Q = sp.linalg.orth(Q)  # orthogonal matrices preserve operator norms
        else:
            # 50% chance to have some complex eigenvalues
            num_complex = self._rng.integers(0, dim, endpoint=True)
            if num_complex % 2 == 1:
                num_complex -= 1
            num_real = dim - num_complex

            real_eigenvalues = self._rng.uniform(*eig_range, size=num_real)
            real_eigenvectors = self._rng.uniform(*eigvec_range, size=(dim, num_real))

            # complex eigenvectors and eigenvalues must have conjugate pairs in the diagonalization for A to be real
            complex_eigenvalues = self._rng.uniform(*eig_range, size=num_complex // 2) + 1j * self._rng.uniform(
                *eig_range, size=num_complex // 2)
            complex_eigenvectors = self._rng.uniform(*eigvec_range,
                                                     size=(dim, num_complex // 2)) + 1j * self._rng.uniform(
                *eigvec_range, size=(dim, num_complex // 2))

            eigenvalues = np.concatenate((real_eigenvalues, complex_eigenvalues, complex_eigenvalues.conjugate()))
            Q = np.hstack((real_eigenvectors, complex_eigenvectors, complex_eigenvectors.conjugate()))

            # shuffle columns
            pi = self._rng.permutation(dim)
            Q = Q[pi]
            eigenvalues = eigenvalues[pi]

        M = Q @ np.diag(eigenvalues) @ np.linalg.inv(Q)
        return M.real

    def _make_A(self, dim):
        return self._eigenvalues_to_matrix(dim, self._A_eigval_range, self._A_eigvec_range)

    def calc_dynamics(self, t, x):
        return self.A @ x

    @SimpleSystem.latent_dim.setter
    def latent_dim(self, value):
        """
        Only remakes :math:`A` if the latent dimension is changed (it doesn't change if only the embedded dimension
        is changed.
        """
        self._latent_dim = value
        self._update_embedder_and_controller()
        self.A = self._make_A(value)
