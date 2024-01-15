"""
N-Body System.
===============

Note
------
The name `santi` is a reference to the Chinese name for the 3-body problem and the famous eponymous novel from Cixin Liu.
"""
import numpy as np
import rebound

from .utils.simple import SimpleSystem


class NBodySystem(SimpleSystem):
    r"""
    N-Body system.

    Note
    -------
    We implement a very simple version of the n-body problem. We assume a system with :math:`n` planetary bodies and 1 sun.
    All planetary bodies have the same mass. The initial conditions are a matrix with 4 dimensions for every planetary body.
    These dimensions give us the :math:`(x, y, z, vy)` of the planetary body where the first three coordinates are the :math:`xyz`-position
    and the last coordinate :math:`vy` is the :math:`y` velocity. For simplicity, this is the only component of velocity that you can specify.
    So if we wanted 5 planetary bodies, then our latent dimension would be :math:`5 \times 4 = 20`.


    Example
    ---------
    >>> from dynadojo.wrappers import SystemChecker
    >>> n_bodies = 3
    >>> latent_dim = 4 * n_bodies
    >>> embed_dim = latent_dim
    >>> n = 1  # we only want 1 system with several bodies
    >>> timesteps = 10
    >>> system = SystemChecker(NBodySystem(latent_dim, embed_dim, plot_data=True, seed=0))
    >>> x0 = system.make_init_conds(n)
    >>> x = system.make_data(x0, timesteps=timesteps)

    .. image:: ../_images/santi.png

    >>> from dynadojo.challenges import FixedComplexity
    >>> from dynadojo.baselines.dnn import DNN
    >>> challenge = FixedComplexity(l=4, e=None, t=10, N=[3, 5, 10], reps=3, system_cls=NBodySystem, test_examples=1, test_timesteps=5)
    >>> data = challenge.evaluate(algo_cls=DNN)
    >>> challenge.plot(data)

    .. image:: ../_images/nbody_fixed_complexity.png
    """
    def __init__(self, latent_dim=4, embed_dim=4, mass=10e-3, plot_data=False,
                 IND_range=(-1, 1), OOD_range=(-1, 1),
                 **kwargs):
        r"""
        Initialize the class.

        Parameters
        -----------
        latent_dim : int
            The latent dimension should be :math:`4 \times \text{# of desired planetary bodies}`
        embed_dim : int
            Must be the same as the latent dimension.
        mass : float
            The mass for all planetary bodies. In practice, small values between 0 and 1 will work best.
        plot_data : bool
            If True, visualize the results of the system in a Jupyter notebook. Defaults to False.
        """
        assert latent_dim >= 4 and latent_dim % 4 == 0
        assert latent_dim == embed_dim, "Latent dimension and embedding dimension must be the same."
        self._n_bodies = latent_dim // 4
        self._mass = mass
        self._plot_data = plot_data
        super().__init__(latent_dim, embed_dim, IND_range=IND_range, OOD_range=OOD_range, **kwargs)

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        assert not np.any(control), "Control must be zero."

        data = []
        sim = None
        for planets in init_conds:
            sim = rebound.Simulation()
            sim.add(m=1)
            for x, y, z, vy in zip(planets[::4], planets[1::4], planets[2::4], planets[3::4]):
                sim.add(m=self._mass, x=x, y=x, z=x, vy=vy)

            example = []
            for t in np.linspace(self._t_range[0], self._t_range[1], num=timesteps):
                sim.integrate(t)
                positions = []
                for particle in sim.particles:
                    positions += [particle.x, particle.y, particle.z, particle.vy]
                example.append(positions)
            data.append(example)
        data = np.array(data)[:, :, 4:]

        if noisy:
            data += self._rng.normal(scale=self._noise_scale, size=data.shape)

        if self._plot_data:
            rebound.OrbitPlot(sim)

        return data



