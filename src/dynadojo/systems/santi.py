import numpy as np
import rebound

from .utils.simple import SimpleSystem


class NBodySystem(SimpleSystem):
    def __init__(self, latent_dim=4, embed_dim=4, mass=10e-3, plot_data=False,
                 IND_range=(-1, 1), OOD_range=(-1, 1),
                 **kwargs):
        assert latent_dim >= 4 and latent_dim % 4 == 0
        assert latent_dim == embed_dim
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
            for x, y, z, vy in zip(planets[::3], planets[1::3], planets[2::3], planets[3::3]):
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



