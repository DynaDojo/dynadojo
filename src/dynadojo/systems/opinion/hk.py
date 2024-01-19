"""
Hegselmann-Krause
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import networkx as nx

from ..utils.opinion import OpinionSystem


class HKSystem(OpinionSystem):
    """
    Hegselmann-Krause (HK)

    Adapted from Hegselmann Rainer and Ulrich Krause. “Opinion dynamics and bounded confidence: models, analysis and simulation”. In: (2002).

    Example
    ---------
    >>> from dynadojo.systems.opinion import HKSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 50
    >>> embed_dim = 50
    >>> timesteps = 200
    >>> n = 1
    >>> system = SystemChecker(HKSystem(dim, embed_dim, epsilon=0.2, noise_scale=0.0005))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/hk.png

    """
    def __init__(self, latent_dim=2, embed_dim=2,
                 IND_range=(-0.5, 0.5),
                 OOD_range=(-1, 1),
                 epsilon=0.32,
                 p_edge=1.0,
                 noise_scale=0.01,
                 seed=None):

        """
        Initializes a HKSystem instance.

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

    def create_model(self, x0):
        self.model = op.HKModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0