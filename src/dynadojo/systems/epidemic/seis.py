import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

from ..utils.epidemic import EpidemicSystem

class SEISSystem(EpidemicSystem):
    def __init__(self, latent_dim, embed_dim,
                 noise_scale=0.01,
                 IND_range=(0, 3),
                 OOD_range=(0, 3),
                 p_infection=0.1,
                 p_recovery=0.5,
                 p_edge=0.1,
                 latency=0.05,
                 fraction_infected=0.1, #initial infections
                 group_status=False,
                 seed=None):

        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, group_status, seed)

        self.g = nx.erdos_renyi_graph(self.latent_dim, p_edge)
        
        self.config = mc.Configuration()
        self.config.add_model_parameter('beta', p_infection)
        self.config.add_model_parameter('lambda', p_recovery)
        self.config.add_model_parameter('alpha', latency)
        self.config.add_model_parameter("fraction_infected", fraction_infected)

    def create_model(self, x0):
        self.model = ep.SEISModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0