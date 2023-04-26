import dynascale as ds
import numpy as np


def main():
    model = ds.baselines.NaiveLinearRegression
    # factory = bn.lds.classes.LDSFactory()
    factory = ds.ca.classes.CAFactory()
    task = ds.Task(
        N=[100],
        L=[3, 4, 5, 6],
        E=[12],
        T=[10],
        reps=3,
        factory=factory,
        supepochs=10,
    )
    scores = task.evaluate(model)
    # task = bn.abstractions.Task(N=[100], low=-5, high=5, L=[3], E=[4], T=[10], reps=3, factory=factory, supepochs=10)
    # task = bn.abstractions.FixedExamplesTask(num_ex=100, low=-4, high=3, L=[2, 3, 4, 5], E=[64], T=[10], reps=1, factory=factory, metrics=ca_eval_func)
    # TODO: add flag for in our out of dist (that will be overridden)
    print(scores)


if __name__ == '__main__':
    main()
