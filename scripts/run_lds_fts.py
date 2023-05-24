import os

import pandas as pd

from dynascale.baselines.lr import MyLinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedTrainSize2



def main():
    task = FixedTrainSize2(
        n=100,
        D=range(2, 50, 2),
        T=[25],
        max_control_cost_per_dim=0,
        control_horizons=1,
        test_examples=10,
        reps=10,
        test_timesteps=25,
        system_cls=LDSSystem
    )

    file = "../cache/lr_data_fts.csv"
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

    file = "../cache/simple_data_fts.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            simple_data = task.evaluate(model_cls=Simple, model_kwargs={'epochs': 100}, id="Simple")
            simple_data.to_csv(file)
        else:
            simple_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        simple_data = task.evaluate(model_cls=Simple, id="Simple")
        simple_data.to_csv(file)

    data = pd.concat([simple_data, lr_data])
    task.plot(data)


if __name__ == '__main__':
    main()
