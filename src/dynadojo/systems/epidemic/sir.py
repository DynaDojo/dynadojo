"""
Susceptible-Exposed-Recovered
"""
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import networkx as nx

from ..utils.epidemic import EpidemicSystem
from ...utils.opinion import plot

import numpy as np


class SIRSystem(EpidemicSystem):
    """
    Susceptible-Exposed-Recovered/Removed

    Note: agents are removed from the simulation when they are recovered, representing a terminal state

    Example
    ---------
    >>> from dynadojo.systems.epidemic import SIRSystem
    >>> from dynadojo.wrappers import SystemChecker
    >>> from dynadojo.utils.opinion import plot
    >>> latent_dim = 40
    >>> embed_dim = 40
    >>> timesteps = 50
    >>> n = 1
    >>> system = SystemChecker(SIRSystem(dim, embed_dim,  p_removal=0.1, noise_scale=0.1))
    >>> x0 = system.make_init_conds(n=n)
    >>> x = system.make_data(x0, timesteps=timesteps)
    >>> plot([x], target_dim=2, max_lines=100)

    .. image:: ../_images/sir.png

    """
    def __init__(self, latent_dim=2, embed_dim=2,
                 IND_range=(0, 2),
                 OOD_range=(0, 2),
                 p_infection=0.1,
                 p_removal=0.5,
                 p_edge=0.1,
                 p_initial_infected=0.1,
                 group_status=False,
                 noise_scale=0.01,
                 seed=None):

        """
        Initializes a SIRSystem instance.

        Parameters
        -------------
        latent_dim : int
            Number of agents interacting
        embed_dim : int
            If group_status is False, must be the same as latent_dim; if group_status is True, must be equal to number of statuses for the system (SIR: 3)
        IND_range : tuple
            In-distribution range of initial possible values, {0: Susceptible, 1: Infected, 2: Recovered/Removed}.
            We add +1 to the max range value to use np.floor to ensure all values are equally sampled from.
        OOD_range : tuple
            Out-of-distribution range of initial possible values, {0: Susceptible, 1: Infected, 2: Recovered/Removed}.
            We add +1 to the max range value to use np.floor to ensure all values are equally sampled from.
        p_infection : float
            Probability of an Infected agent passing the infection to another agent they interact with, moving them from Susceptible to Infected
        p_recovery : float
            Probability of an Infected agent recovering to Recovered/Removed from simulation
        p_edge : float
            Probability for every pair of agents that an edge will be created between them
        p_initial_infected : float
            The fraction of the agent population that starts Infected
        group_status : boolean
            Whether to report trajectories as inidividual agents (False) or as how the aggregated groups of the number of agents in each status, SIR, changes (True)
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
        self.config.add_model_parameter('gamma', p_removal)
        self.config.add_model_parameter('fraction_infected', 1.0) # done custom in EpidemicSystem 

    def create_model(self, x0):
        self.model = ep.SIRModel(self.g)
        self.model.set_initial_status(self.config)
        self.model.status = x0
        self.model.initial_status = x0

    def save_plotted_trajectories( self, 
            y_true:np.ndarray, 
            y_pred: np.ndarray,
            filepath: str,
            tag: str = "", 
        ):
        
        fig, ax = plot([y_true, y_pred], 
                       target_dim=min(self._embed_dim, 3), 
                       labels=["true", "pred"], 
                       max_lines=10,
                       title=f"Epidemic l={self.latent_dim}, e={self._embed_dim} - {tag}")
        fig.savefig(filepath, bbox_inches='tight', dpi=300, transparent=True, format='pdf')
        return fig, ax