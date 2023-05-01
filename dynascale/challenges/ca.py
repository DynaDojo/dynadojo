from typing import Callable

import numpy as np
import cellpylib as cpl

from dynascale.abstractions import Factory

RNG = np.random.default_rng()
def _make_data(n: int, timesteps: int, latent_dim: int = 1, embed_dim: int = 64, control: Callable = None) -> np.ndarray:
    u = control or (lambda t: np.zeros(embed_dim, dtype='int32'))

    data = []
    for _ in range(n):
        cellular_automata = cpl.init_random(embed_dim)
        traj = [cellular_automata]
        for t in range(timesteps - 1):
            cellular_automata = cpl.evolve(cellular_automata, timesteps=1, r=latent_dim,
                                           apply_rule=lambda n, c, t: cpl.nks_rule(n,
                                                                                   30))  # TODO: get better random rule discoverer
            cellular_automata += u(t)
            traj.append(cellular_automata)
        traj = np.array(traj).squeeze()
        data.append(traj)
    data = np.array(data)
    return data
