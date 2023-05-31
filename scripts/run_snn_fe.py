import os

import pandas as pd

from dynascale.baselines.simple import Simple
from dynascale.baselines.lr import ManualLinearRegression
from dynascale.systems.snn import SNNSystem
from dynascale.tasks import FixedError



def main():
    target_loss = 0.5
    task = FixedError(
        target_loss=target_loss,
        L=[10, 100, 1000, 10000],
        E=[20, 200, 2000, 20000],
        t=50,
        max_control_cost_per_dim=0,
        control_horizons=0,
        test_examples=100,
        reps=50,
        test_timesteps=50,
        system_cls=SNNSystem,
    )

    file = f"../cache/SNN/lr_data_fe_{target_loss}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            lr_data = task.evaluate(model_cls=ManualLinearRegression, id="LR")
            lr_data.to_csv(file)
        else:
            lr_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        lr_data = task.evaluate(model_cls=ManualLinearRegression, id="LR")
        lr_data.to_csv(file)

    task.plot(lr_data)

    file = f"../cache/CA/cnn_data_fe_{target_loss}.csv"
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


if __name__ == '__main__':
    main()
