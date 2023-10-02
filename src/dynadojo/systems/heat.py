"""
Adapted from: https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
"""
import numpy as np

from .utils import SimpleSystem


class HeatEquation(SimpleSystem):
    """
    Implements the 2D heat equation
    """
    def __init__(self, latent_dim, embed_dim,
                 alpha=2, dx=1,
                 IND_range=(0, 100), OOD_range=(-100, 0),
                 **kwargs):
        assert np.sqrt(latent_dim).is_integer(), "Latent dimension must be a perfect square."
        assert latent_dim == embed_dim
        self.plate_length = int(np.sqrt(latent_dim))
        self.alpha = alpha
        self.dx = dx
        self.dt = (self.dx ** 2) / (4 * self.alpha)
        self.gamma = (self.alpha * self.dt) / (self.dx ** 2)
        super().__init__(latent_dim, embed_dim, IND_range=IND_range, OOD_range=OOD_range, **kwargs)

    def _calculate(self, u, timesteps):
        for k in range(0, timesteps - 1, 1):
            for i in range(1, self.plate_length - 1, self.dx):
                for j in range(1, self.plate_length - 1, self.dx):
                    u[k + 1, i, j] = self.gamma * (
                                u[k][i + 1][j] + u[k][i - 1][j] + u[k][i][j + 1] + u[k][i][j - 1] - 4 * u[k][i][j]) + \
                                     u[k][i][j]

        return u

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        assert not np.any(control), "Control must be zero."

        data = []

        for u0 in init_conds:
            # Initialize solution: the grid of u(k, i, j)
            u = np.empty((timesteps, self.plate_length, self.plate_length))

            # Set the initial condition
            u[0] = u0.reshape((self.plate_length, self.plate_length))

            # Solve the PDE
            u = self._calculate(u, timesteps)

            # Flatten solution
            u = u.reshape((timesteps, -1))
            data.append(u)

        data = np.array(data)

        if noisy:
            data += self._rng.normal(scale=self._noise_scale, size=data.shape)

        return data



