import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

from dynadojo.abstractions import AbstractSystem

import numpy as np


np.printoptions(suppress=True)


class BiasSystem(AbstractSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.01,
                 IND_range=(-0.5, 0.5),  # prevent spawning extinct species
                 OOD_range=(-1, 1),
                 seed=None):
        super().__init__(latent_dim, embed_dim)

        assert embed_dim == latent_dim

        self.noise_scale = noise_scale

        self.IND_range = IND_range
        self.OOD_range = OOD_range

        self.config = mc.Configuration()
        self.config.add_model_parameter("epsilon", 0.32)
        self.config.add_model_parameter("gamma", 0)

    def create_model(self, x0):
        g = nx.complete_graph(self.latent_dim)
        self.model = op.AlgorithmicBiasModel(g)
        self.model.set_initial_status(x0, self.config)

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):

            x0.append(np.random.uniform(0, 1, (self.latent_dim)))

        return x0

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        if noisy:
            noise = np.random.normal(
                0, self.noise_scale, (timesteps, self.latent_dim))
        else:
            noise = np.zeros((timesteps, self.latent_dim))

        def dynamics(x0):
            self.create_model(x0)

            iterations = self.model.iteration_bunch(timesteps)
            Steps = []
            for iteration in iterations:
                step = []
                for _, val in iteration["status"].items():
                    step.append(val)
                Steps.append(step)
            return Steps

        if control:
            for x0, u in zip(init_conds, control):
                sol = dynamics(x0)
                sol += noise + u
                data.append(sol)

        else:
            for x0 in init_conds:
                sol = dynamics(x0)
                print(sol)
                print(noise)
                sol += noise
                data.append(sol)

        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
