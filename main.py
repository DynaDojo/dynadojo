from dynascale.utils.lds import plot
from dynascale.challenges.lds import LDSChallenge
from dynascale.challenges.ca import CAChallenge
from dynascale.baselines.new_koopman import Koopman
import time

import numpy as np

def main():
    # challenge = CAChallenge(2, 64, mutation_p=1)
    # x0 = challenge.make_init_conds(n=1)
    # x = challenge.make_data(x0, timesteps=50)
    # x_mut = challenge.make_data(x0, timesteps=50, noisy=True)
    # plot([x, x_mut], labels=["in", "OOD"])

    start = time.time()
    challenge = LDSChallenge(50, 100)
    x0 = challenge.make_init_conds(100)
    y0 = challenge.make_init_conds(100, in_dist=False)
    x = challenge.make_data(x0, timesteps=50)
    y = challenge.make_data(y0, timesteps=50)
    plot([x, y], target_dim=2, labels=["in", "out"])

    print(time.time() - start)


if __name__ == '__main__':
    main()
