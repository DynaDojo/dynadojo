"""
Weighted Hegselmann-Krause
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import networkx as nx

from ..utils.opinion import OpinionSystem


class WHKSystem(OpinionSystem):
    """
    Weighted Hegselmann-Krause (WHK)

    Adapted from Letizia Milli. “Opinion dynamic modeling of news perception”. In: Applied Network Science 6.1 (2021), pp. 1–19.

    Example
    ---------
    >>> from dynadojo.systems.opinion import WHKSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 50
    >>> embed_dim = 50
    >>> timesteps = 20
    >>> n = 1
    >>> system = SystemChecker(WHKSystem(dim, embed_dim, epsilon=0.6, noise_scale=0.0005))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/whk.png

    """
    def __init__(self, latent_dim=2, embed_dim=2,
                 IND_range=(-0.5, 0.5),
                 OOD_range=(-1, 1),
                 epsilon=0.32,
                 p_edge=0.1,
                 noise_scale=0.01,
                 seed=None):

        """
        Initializes a WHKSystem instance.

        Parameters
        -------------
        latent_dim : int
            Number of agents interacting
        embed_dim : int
            Must be the same as latent_dim
        IND_range : tuple
            In-distribution range of initial possible values
        OOD_range : tuple
            Out-of-distribution range of initial possible values
        epsilon : float
            Confidence threshold bound for determining if two agents are similar enough
        p_edge : float
            Probability for every pair of agents that an edge will be created between them
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """
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