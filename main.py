import dynascale as ds


def main():
    task = ds.Task(
        N=[5000],
        L=[2],
        E=[4],
        T=[5],
        factory_cls=ds.lds.LDSFactory,
        supepochs=2,
    )
    scores = task.evaluate(ds.baselines.NaiveLinearRegression)
    print(scores)


if __name__ == '__main__':
    main()
