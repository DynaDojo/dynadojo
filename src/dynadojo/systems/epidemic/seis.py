"""
Susceptible-Exposed-Infected-Susceptible
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx

from ..utils.epidemic import EpidemicSystem

import numpy as np

class SEISSystem(EpidemicSystem):
    """
    Susceptible-Exposed-Infected-Susceptible

    Example #1 - Individual Trajectories
    ---------
    >>> from dynadojo.systems.epidemic import SEISSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 10
    >>> embed_dim = 10
    >>> timesteps = 10
    >>> n = 1
    >>> system = SystemChecker(SEISSystem(dim, embed_dim, p_recovery=0.1, noise_scale=0))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/seis-traj.png

    Example #2 - Grouped by Status
    ---------
    >>> latent_dim = 5
    >>> embed_dim = 3
    >>> timesteps = 40
    >>> n = 1
    >>> system = SystemChecker(SEISSystem(dim, embed_dim, p_recovery=0.05, noise_scale=0.1, p_initial_infected=0, group_status=True))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, datalabels=["Susceptible", "Exposed", "Infected"], max_lines=100)


    .. image:: ../_images/seis-group.png

    """
    def __init__(self, latent_dim=2, embed_dim=2,
                 IND_range=(0, 3),
                 OOD_range=(2, 3),
                 p_infection=0.1,
                 p_recovery=0.5,
                 latency=0.05,
                 p_edge=0.2,
                 p_initial_infected=0.1,  # initial infections
                 group_status=False,
                 noise_scale=0.01,
                 seed=None):
        """
        Initializes a SEISSystem instance.

        Parameters
        -------------
        latent_dim : int
            Number of agents interacting
        embed_dim : int
            If group_status is False, must be the same as latent_dim; if group_status is True, must be equal to number of statuses for the system (SEIS: 3)
        IND_range : tuple
            In-distribution range of initial possible values, {0: Susceptible, 1: Infected, 2: Exposed}.
            We add +1 to the max range value to use np.floor to ensure all values are equally sampled from.
        OOD_range : tuple
            In-distribution range of initial possible values, {0: Susceptible, 1: Infected, 2: Exposed}.
            We add +1 to the max range value to use np.floor to ensure all values are equally sampled from.
        p_infection : float
            Probability of an Infected agent passing the infection to another agent they interact with, moving them from Susceptible to Exposed
        p_recovery : float
            Probability of an Infected agent recovering back to Susceptible
        latency : float
            Probability an agent moves from Exposed to Infected
        p_edge : float
            Probability for every pair of agents that an edge will be created between them
        p_initial_infected : float
            The fraction of the agent population that starts Infected
        group_status : boolean
            Whether to report trajectories as inidividual agents (False) or as how the aggregated groups of the number of agents in each status, SEI, changes (True)
        noise_scale : float
            Normal noise is added per timestep to a solution. Standard deviation (spread or “width”) of the distribution.
            Must be non-negative.
        seed : int or None
            Seed for random number generation.
        """
        num_statuses = 3
        super().__init__(latent_dim, embed_dim, noise_scale, IND_range, OOD_range, p_initial_infected, group_status, num_statuses, seed)

        self.g = nx.erdos_renyi_graph(self.latent_dim, p_edge)
        
        self.config = mc.Configuration()
        self.config.add_model_parameter('beta', p_infection)
        self.config.add_model_parameter('lambda', p_recovery)
        self.config.add_model_parameter('alpha', latency)
        self.config.add_model_parameter('fraction_infected', 1.0) # done custom in EpidemicSystem 

    def create_model(self, x0):
        self.model = ep.SEISModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0


    