import os

import pandas as pd

from dynascale.baselines.lr import MyLinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedTrainSize



def main():
    n = 500
    task = FixedTrainSize(
        n=n,
        L=[5, 10, 50, 100, 500, 1000],
        E=None,
        T=[50],
        max_control_cost_per_dim=0,
        control_horizons=0,
        test_examples=100,
        reps=50,
        test_timesteps=50,
        system_cls=LDSSystem
    )

    file = f"../cache/lr_data_fts_{n}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
            lr_data.to_csv(file)
        else:
            lr_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
        lr_data.to_csv(file)

    task.plot(lr_data)

    file = f"../cache/simple_data_fts_{n}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            simple_data = task.evaluate(model_cls=Simple, model_kwargs={'epochs': 300}, id="Simple")
            simple_data.to_csv(file)
        else:
            simple_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        simple_data = task.evaluate(model_cls=Simple, id="Simple")
        simple_data.to_csv(file)

    task.plot(simple_data)
    task.plot(pd.concat([simple_data, lr_data]))


if __name__ == '__main__':
    main()
