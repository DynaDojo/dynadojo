import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

from ..utils import OpinionSystem

import numpy as np

class WHKSystem(OpinionSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.01,
                 IND_range=(-0.5, 0.5),
                 OOD_range=(-1, 1),
                 epsilon=0.32,
                 p_edge=0.1,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, seed)

        assert embed_dim == latent_dim

        self.g = nx.erdos_renyi_graph(self.latent_dim, p_edge)
        
        self.config = mc.Configuration()
        self.config.add_model_parameter("epsilon", epsilon)

        if isinstance(self.g, nx.Graph):
            edges = self.g.edges
        else:
            edges = [(self.g.vs[e.tuple[0]]['name'], g.vs[e.tuple[1]]['name']) for e in self.g.es]

        for e in edges:
            self.config.add_edge_configuration("weight", e, self._rng.random())

    def create_model(self, x0):
        self.model = op.WHKModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0