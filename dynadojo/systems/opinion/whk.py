import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

from ...abstractions import AbstractSystem

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

        # mMean field scenario
        g = nx.complete_graph(self.latent_dim)

        # Algorithmic Bias model
        self.model = op.AlgorithmicBiasModel(g)

        # Model configuration
        config = mc.Configuration()
        config.add_model_parameter("epsilon", 0.32)
        config.add_model_parameter("gamma", 0)
        self.model.set_initial_status(config)

        # viz = OpinionEvolution(self.model, iterations)
        # viz.plot("opinion_ev.pdf")

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            temp = []
            for _ in range(self._latent_dim):
                if in_dist:
                    number = np.random.uniform(
                        self.IND_range[0], self.IND_range[1])
                else:
                    number = np.random.uniform(
                        self.OOD_range[0], self.OOD_range[1])
                temp.append(number)
            x0.append(temp)

        return x0

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        def dynamics():
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
                sol = dynamics()
                data.append(sol)

        else:
            for x0 in init_conds:
                u = np.zeros((timesteps, self.latent_dim))
                sol = dynamics()
                data.append(sol)

        data = np.array(data)
        print(data)
        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)


class HKOpinionSystem(AbstractSystem):
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

        # Network topology
        g = nx.erdos_renyi_graph(self.latent_dim, 0.1)

        # Model selection
        self.model = op.WHKModel(g)

        # Model Configuration
        config = mc.Configuration()
        config.add_model_parameter("epsilon", 0.85)
        config.add_model_parameter('perc_stubborness', 0.01)

        # Setting the edge parameters
        weight = 0.2
        if isinstance(g, nx.Graph):
            edges = g.edges
        else:
            edges = [(g.vs[e.tuple[0]]['name'], g.vs[e.tuple[1]]['name'])
                     for e in g.es]

        for e in edges:
            config.add_edge_configuration("weight", e, np.random.random())

        self.model.set_initial_status(config)

    def make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        x0 = []
        for _ in range(n):
            temp = []
            for _ in range(self._latent_dim):
                if in_dist:
                    number = np.random.uniform(
                        self.IND_range[0], self.IND_range[1])
                else:
                    number = np.random.uniform(
                        self.OOD_range[0], self.OOD_range[1])
                temp.append(number)
            x0.append(temp)

        return x0

    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []

        def dynamics():
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
                sol = dynamics()
                data.append(sol)

        else:
            for x0 in init_conds:
                u = np.zeros((timesteps, self.latent_dim))
                sol = dynamics()
                data.append(sol)

        data = np.array(data)
        data = np.transpose(data, axes=(0, 2, 1))
        return data

    def calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.latent_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
