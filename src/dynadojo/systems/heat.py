"""
2D Heat Equation.
"""
import numpy as np

from .utils.simple import SimpleSystem


class HeatEquation(SimpleSystem):
    """
    2D Heat Equation. Adapted from [1]_. Models how heat dissipates across a 2D square plate.

    References
    ------------
    .. [1] https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """

    def __init__(self,
                 latent_dim=4,
                 embed_dim=4,
                 alpha=2, dx=1,
                 IND_range=(0, 100), OOD_range=(-100, 0),
                 seed=None,
                 **kwargs):
        r"""
        Initialize the class.

        Parameters
        ----------
        alpha : float
            Thermal diffusivity. Should be positive.
        dx : float
            :math:`\delta x`.
        latent_dim : int
            The length of the square plate. Must be a perfect square.
        embed_dim : int
            Must be the same as the latent dimension.
        seed : int or None, optional
            Seed for random number generation.
        embedder_sv_range : tuple
            The singular value range for the embedder matrix. Singular values are non-negative by convention.
            The singular values should exclude 0 to ensure the embedder is invertible.
        controller_sv_range : tuple
            The singular value range for the controller matrix.
        IND_range : tuple
            The in-distribution range of possible starting temperatures.
        OOD_Range : tuple
            The out-of-distribution range of possible starting temperatures.
        **kwargs
            Additional keyword arguments.
        """
        assert np.sqrt(latent_dim).is_integer(), "Latent dimension must be a perfect square."
        assert latent_dim == embed_dim
        super().__init__(latent_dim, embed_dim, IND_range=IND_range,
                         OOD_range=OOD_range, seed=seed, **kwargs)
        self.plate_length = int(np.sqrt(latent_dim))
        self.alpha = alpha
        self.dx = dx
        self.dt = (self.dx ** 2) / (4 * self.alpha)
        self.gamma = (self.alpha * self.dt) / (self.dx ** 2)

    def _calculate(self, u, timesteps):
        for k in range(0, timesteps - 1, 1):
            for i in range(1, self.plate_length - 1, self.dx):
                for j in range(1, self.plate_length - 1, self.dx):
                    u[k + 1, i, j] = self.gamma * (
                            u[k][i + 1][j] + u[k][i - 1][j] + u[k][i][j + 1] + u[k][i][j - 1] - 4 * u[k][i][j]) + \
                                     u[k][i][j]

        return u

    def make_data(self,
                  init_conds: np.ndarray,
                  control: np.ndarray,
                  timesteps: int,
                  noisy=False) -> np.ndarray:
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

        # Add noise
        if noisy:
            data += self._rng.normal(scale=self._noise_scale, size=data.shape)

        return data
