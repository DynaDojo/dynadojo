import dynascale as ds
from dynascale.baselines.lr import NaiveLinearRegression


def main():
    task = ds.Task(
        N=[10],
        L=[2],
        E=[10],
        T=[5],
        factory_cls=ds.ca.CAChallenge,
        supepochs=2,
        test_size=1,
        trials=100,
    )
    scores = task.evaluate(NaiveLinearRegression)
    print(scores)


if __name__ == '__main__':
    main()
