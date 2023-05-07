from dynascale.challenges.ca import CAChallenge
from dynascale.utils.ca import plot


import numpy as np

def main():
    challenge = CAChallenge(2, 64)
    control = -2 * np.ones((50, challenge.embed_dim))
    x1 = challenge.make_data(50, n=10, control=control)
    x2 = challenge.make_data(50, n=10, in_dist=False)
    plot([x1, x2], labels=["in", "OOD"])


if __name__ == '__main__':
    main()
