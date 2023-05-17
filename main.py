from dynascale.utils.lds import plot
from dynascale.challenges.snn import SNNChallenge
from dynascale.baselines.simple import Simple
from dynascale.tasks import FixedTrainSize
import numpy as np
import scipy as sp

def main():
    latent_dim = 3
    embed_dim = 6
    n = 3
    timesteps = 50
    challenge = SNNChallenge(latent_dim, embed_dim)
    x0 = challenge.make_init_conds(n)
    y0 = challenge.make_init_conds(30, in_dist=False)
    x = challenge.make_data(x0, timesteps=timesteps)
    y = challenge.make_data(y0, timesteps=timesteps)
    plot([x, y], target_dim=3, labels=["in", "out"], max_lines=30)


if __name__ == '__main__':
    main()
