import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op

from ..utils import OpinionSystem

import numpy as np

''' 
Note: in the current pip release of ndlib, AlgorithmicBiasMediaModel is not implemented. It is coming shortly (https://github.com/GiulioRossetti/ndlib/issues/251).
For now, using this system requires local cloning of ndlib.
'''
class MediaBiasSystem(OpinionSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.01,
                 IND_range=(0, 0.5),
                 OOD_range=(0.5, 1),
                 epsilon=0.32,
                 bias=0,
                 p_edge=1,
                 n_media=3,
                 p_interaction=0.1,
                 bias_media=0.1,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, seed)

        assert embed_dim == latent_dim
        assert latent_dim > 30

        # Network topology
        self.g = nx.erdos_renyi_graph(self.latent_dim, p_edge)

        # Model configuration
        self.config = mc.Configuration()
        self.config.add_model_parameter("epsilon", epsilon)
        self.config.add_model_parameter("gamma", bias)
        self.config.add_model_parameter("k", n_media)
        self.config.add_model_parameter("p", p_interaction)
        self.config.add_model_parameter("gamma_media", bias_media)

    def create_model(self, x0):
        self.model = op.AlgorithmicBiasMediaModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0