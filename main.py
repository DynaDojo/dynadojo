from dynascale.utils.lds import plot
from dynascale.challenges.lds import LDSChallenge
from dynascale.baselines.new_koopman import Koopman

import numpy as np

def main():
    challenge = LDSChallenge(2, 3)
    timesteps = 10
    x = challenge.make_data(timesteps, n=100)
    model = Koopman(challenge.latent_dim, challenge.embed_dim, timesteps)
    model.fit(x)
    # y = model.predict(x[:, 0], timesteps)
    # print(challenge.calc_error(x, y))


if __name__ == '__main__':
    main()
