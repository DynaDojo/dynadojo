"""
Algorithmic Bias with Media
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as op
import networkx as nx

from ..utils.opinion import OpinionSystem


class MediaBiasSystem(OpinionSystem):

    """
    Algorithmic Bias with Media

    Note: In the current pip release of ndlib, AlgorithmicBiasMediaModel is not implemented. It is coming shortly (https://github.com/GiulioRossetti/ndlib/issues/251).
    For now, using this system requires local cloning of ndlib into your repo.

    Example
    ---------
    >>> from dynadojo.systems.opinion import MediaBiasSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 50
    >>> embed_dim = 50
    >>> timesteps = 15
    >>> n = 1
    >>> system = SystemChecker(MediaBiasSystem(dim, embed_dim, epsilon=0.3, bias=0, n_media=4, bias_media=0.2, noise_scale=0.005))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/media_bias.png

    """
    def __init__(self, latent_dim=31, embed_dim=31,
                 IND_range=(0, 0.5),
                 OOD_range=(0.5, 1),
                 epsilon=0.32,
                 bias=0,
                 p_edge=1,
                 n_media=3,
                 p_interaction=0.1,
                 bias_media=0.1,
                 noise_scale=0.01,
                 seed=None):

        """
        Initializes a MediaBiasSystem instance.

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
        n_media : int
            Number of media to simulate
        p_interaction : float
            Probability of a media interaction with an agent
        bias_media : float
            Algorithmic bias for media
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
        self.config.add_model_parameter("k", n_media)
        self.config.add_model_parameter("p", p_interaction)
        self.config.add_model_parameter("gamma_media", bias_media)

    def create_model(self, x0):
        self.model = op.AlgorithmicBiasModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0