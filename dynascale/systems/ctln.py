from dynascale.abstractions import System
import torch
import torch.nn as nn
from torchdiffeq import odeint
import networkx as nx
import numpy as np
import random

RNG = np.random.default_rng()


class CTLNSystem(System):
    def __init__(self, latent_dim, embed_dim):
        super().__init__(latent_dim, embed_dim)
        self._latent_dim = latent_dim
        self._embed_dim = embed_dim
        self.relu = nn.ReLU()


    def make_init_conds(self, p):
        self._nodes = self._latent_dim #latent dim and nodes the same?
        self._p = p

        self._delta = random.uniform(0, 1) #what should the upper bound on the randomly generated constants be?
        self._epsilon = random.uniform(0, self._delta/(self._delta+1))
        self._graph = self._make_graph(self._nodes, self._p)
        self._x0 = torch.tensor([random.uniform(0, 1) for _ in range(self._nodes)])
        self._T = torch.tensor(random.uniform(0, 1))
        self._input = torch.tensor(random.uniform(0, 1))
        self._state = (self._x0, self._T, self._graph, self._input)
        

    def _make_graph(self, nodes, p):
        g = nx.erdos_renyi_graph(nodes, p)
        edges = nx.adjacency_matrix(g).todense()
        edges_comp = nx.adjacency_matrix(nx.complement(g)).todense()
        graph = (-edges_comp - self._epsilon) + (-edges + self._delta)      
        np.fill_diagonal(graph, 0)
        return torch.tensor(graph)


    def make_data(self, init_conds: np.ndarray, control: np.ndarray, timesteps: int, noisy=False) -> np.ndarray:
        data = []
        time = torch.linspace(0, 1, timesteps)

        def dynamics(t, state):
            x, T, W, b = state
            y = torch.matmul(W, x) + b

            dx = (-x + self.relu(y))/T
            dT = torch.zeros_like(T)
            dW = torch.zeros_like(W)
            db = torch.zeros_like(b)

            return (dx, dT, dW, db)

        sol = odeint(dynamics, self._state, time, method='rk4')
        data.append(sol[0])
        return sol[0].unsqueeze(0).detach().numpy()

    def calc_loss(self, x, y) -> float:
        error = x - y
        return np.mean(error ** 2) / self.embed_dim

    def calc_control_cost(self, control: np.ndarray) -> float:
        return np.linalg.norm(control, axis=(1, 2), ord=2)
