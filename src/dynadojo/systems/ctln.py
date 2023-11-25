import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..abstractions import AbstractSystem


class CTLNSystem(AbstractSystem):
    def __init__(self,
                 latent_dim=2,
                 embed_dim=2,
                 p=0.2,
                 seed=None):
        """
        :param latent_dim: dimension of the latent space
        :param embed_dim: dimension of the embedding space
        :param p: probability of an edge in the graph (more edges = more complex dynamics)
        :param seed: random seed
        """

        super().__init__(latent_dim, embed_dim, seed=seed)
        self._rng = np.random.default_rng(seed=self._seed)
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self.relu = nn.ReLU()
        self._p = p
        self._nodes = latent_dim
        if self._seed is not None:
            torch.manual_seed(self._seed)  # TODO: make sure you relenquish the seed

    def _make_init_conds(self, n: int, in_dist=True) -> np.ndarray:
        self._delta = self._rng.uniform(0, 1)  # what should the upper bound on the randomly generated constants be?
        self._epsilon = self._rng.uniform(0, self._delta / (self._delta + 1))
        graph = self._make_graph(self._nodes, self._p)
        time = torch.tensor([self._rng.uniform(0, 1) for _ in range(self._nodes)])
        x0 = torch.Tensor(self._rng.uniform(0, 1, size=(n, self.embed_dim)))
        b = torch.Tensor(self._rng.uniform(0, 1, size=(self.embed_dim, n)))
        self._state = (x0, time, graph, b)
        return x0

    def _make_graph(self, nodes, p):
        g = nx.erdos_renyi_graph(nodes, p, seed=self._seed)
        edges = nx.adjacency_matrix(g).todense()
        edges_comp = nx.adjacency_matrix(nx.complement(g)).todense()
        graph = (-edges_comp - self._epsilon) + (-edges + self._delta)
        np.fill_diagonal(graph, 0)
        return torch.tensor(graph)

    def _make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        time = torch.linspace(0, 1, timesteps)

        def dynamics(t, state):
            x, T, W, b = state
            y = W @ x.T + b
            dx = (-x.T + self.relu(y)) / T[:, None]
            dT = torch.zeros_like(T)
            dW = torch.zeros_like(W)
            db = torch.zeros_like(b)
            return (dx, dT, dW, db)

        sol = odeint(dynamics, self._state, time, method='rk4')
        data.append(sol[0])
        # result = sol[0].unsqueeze(0).detach().numpy()
        result = np.transpose(sol[0].detach().numpy(), axes=(1, 0, 2))
        return result

    def _calc_error(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def _calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
