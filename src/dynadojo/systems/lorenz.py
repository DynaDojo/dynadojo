"""
Generalized Lorenz System
==========================
"""
import numpy as np

from .utils.simple import SimpleSystem


class LorenzSystem(SimpleSystem):
    """
    Generalized Lorenz system. More details in the DynaDojo paper appendix.

    References
    -----------
    Formulation based on [1]_

    .. [1] https://www.worldscientific.com/doi/epdf/10.1142/S0218127419500378


    Example
    --------
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.lds import plot
    >>> latent_dim = 3
    >>> embed_dim = 3
    >>> n = 10
    >>> timesteps = 50
    >>> system = SystemChecker(LorenzSystem(latent_dim, embed_dim, noise_scale=0, seed=1912))
    >>> x0 = system.make_init_conds(n)
    >>> y0 = system.make_init_conds(n, in_dist=False)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> y = system.make_data(y0, timesteps=timesteps, noisy=True)
    >>> plot([x, y], target_dim=min(latent_dim, 3), labels=["in", "out"], max_lines=15)

    .. image:: ../_images/lorenz.png
    """
    def __init__(self,
                 latent_dim=3,
                 embed_dim=3,
                 sigma=10, r=28, a_squared=1 / 2, b=8 / 3, **kwargs):
        """
        Initialize the class.

        Parameters
        -----------
        latent_dim : int
            Must be an odd number at least 3.
        sigma : int
            the Prandtl number
        r : int
            normalized Rayleigh number (or heating parameter)
        a_squared : float
            :math:`a^2 = 1 / 2` Used default value from [1]_
        b_squared : float
            :math:`b^2 = 8 / 3` Used default value from [1]_
        """
        assert latent_dim % 2 == 1 and latent_dim >= 3, "Latent dimension must be odd number at least 3."
        self._sigma = sigma
        self._r = r
        self._a_squared = a_squared
        self._b = b
        super().__init__(latent_dim, embed_dim, **kwargs)

    def calc_dynamics(self, t, x):
        """See the appendix of the DynaDojo paper for the full details."""
        X, Y, Z = x[:3]
        dX = -self._sigma * X + self._sigma * Y
        dY = -X * Z + self._r * X - Y
        M = self.latent_dim
        N = int((M - 3) / 2)
        pad = -1

        Yj = np.concatenate(([pad], x[3:3 + N], [0]))  # Y_{N+1} = 0
        Zj = np.concatenate(([Z], x[3 + N:]))  # Z_0 = Z

        j = np.arange(1, N + 1)
        beta = np.concatenate(([pad], (j + 1) ** 2 * self._b))
        d = ((2 * j + 1) ** 2 + self._a_squared) / (1 + self._a_squared)

        dZ = X * Y - X * Yj[1] - self._b * Z

        dYj = np.zeros(N + 1)
        dZj = np.zeros(N + 2)
        dYj[j] = j * X * Zj[j - 1] - (j + 1) * X * Zj[j] - d[j - 1] * Yj[j]
        dZj[j] = (j + 1) * X * Yj[j] - (j + 1) * X * Yj[j + 1] - beta[j] * Zj[j]

        dx = np.concatenate(([dX, dY, dZ], dYj[1:], dZj[1:-1]))
        return dx
