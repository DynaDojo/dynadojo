import dynascale as ds
from dynascale.baselines.lr import NaiveLinearRegression
import numpy as np


def main():
    # task = ds.Task(
    #     N=[10],
    #     L=[2],
    #     E=[10],
    #     T=[5],
    #     factory_cls=ds.CAChallenge,
    #     supepochs=2,
    #     test_size=1,
    #     trials=100,
    # )
    # scores = task.evaluate(NaiveLinearRegression)
    # print(scores)
    challenge = ds.CAChallenge(latent_dim=2, embed_dim=64)
    control = np.ones_like((50, challenge.embed_dim))
    x1 = challenge.make_data(50, n=10, control=control)
    x2 = challenge.make_data(50, n=10, in_dist=False)
    ds.utils.ca.plot([x1, x2], labels=["in", "OOD"])


if __name__ == '__main__':
    main()
