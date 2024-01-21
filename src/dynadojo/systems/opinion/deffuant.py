"""
Deffuant (Algorithmic Bias)
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import networkx as nx

from ..utils.opinion import OpinionSystem

class DeffuantSystem(OpinionSystem):
    """
    Deffuant (Algorithmic Bias)

    Note: With no bias, this is a classic Deffuant system. Bias can be added to simulate the effect of algorithms on shaping opinions.

    Adapted from Guillaume Deffuant et al. “Mixing beliefs among interacting agents”. In: Advances in Complex Systems 3.01n04 (2000), pp. 87–98.

    Example
    ---------
    >>> from dynadojo.systems.opinion import DeffuantSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 50
    >>> embed_dim = 50
    >>> timesteps = 15
    >>> n = 1
    >>> system = SystemChecker(DeffuantSystem(dim, embed_dim, epsilon=0.3, bias=0.8, noise_scale=0.005))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/deffuant.png

    """
    def __init__(self, latent_dim=31, embed_dim=31,
                 IND_range=(0, 0.5),
                 OOD_range=(0.5, 1),
                 epsilon=0.32,
                 bias=0,
                 p_edge=1,
                 noise_scale=0.01,
                 seed=None):

        """
        Initializes a DeffuantSystem instance.

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
        bias : int
            When selecting pairs of agents, higher bias increases the chance the agents will be similar, int in [0,100]
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
        assert latent_dim > 30

        # Network topology
        self.g = nx.erdos_renyi_graph(self.latent_dim, p_edge)

        # Model configuration
        self.config = mc.Configuration()
        self.config.add_model_parameter("epsilon", epsilon)
        self.config.add_model_parameter("gamma", bias)

    def create_model(self, x0):
        self.model = op.AlgorithmicBiasModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0