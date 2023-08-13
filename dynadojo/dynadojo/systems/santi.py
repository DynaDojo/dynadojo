"""
Generalized Lorenz system formulation based on
paper from Shen: https://www.worldscientific.com/doi/epdf/10.1142/S0218127419500378
"""
import numpy as np
import rebound

from .utils import SimpleSystem


class NBodySystem(SimpleSystem):
    def __init__(self, latent_dim, embed_dim, mass=10, plot_data=False, **kwargs):
        assert latent_dim == embed_dim == 3, "Latent and embedded dimensions must be 3."
        self._mass = mass
        self._plot_data = plot_data
        super().__init__(latent_dim, embed_dim, **kwargs)

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        assert not np.any(control), "Control must be zero."

        sim = rebound.Simulation()
        for x, y, z in init_conds:
            sim.add(m=self._mass, x=x, y=y, z=z)

        data = []
        for t in np.linspace(self._t_range[0], self._t_range[1], num=timesteps):
            sim.integrate(t)
            data.append([[particle.x, particle.y, particle.z] for particle in sim.particles])

        data = np.stack(data)
        data = np.transpose(data, axes=(1, 0, 2))

        if noisy:
            data += self._rng.normal(scale=self._noise_scale, size=data.shape)

        if self._plot_data:
            # Note: cannot plot noisy data with rebound
            rebound.OrbitPlotSet(sim)

        return data



