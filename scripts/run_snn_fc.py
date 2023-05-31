from dynascale.systems.lds import LDSSystem
import pandas as pd

import os
from dynascale.baselines.lr import ManualLinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.snn import SNNSystem
from dynascale.tasks import FixedComplexity


def main():
    l = 30
    noisy = True
    task = FixedComplexity(N=[1, 10, 100, 1000, 10000],
                           l=l,
                           e=2 * l,
                           t=50,
                           max_control_cost_per_dim=0,
                           control_horizons=0,
                           test_examples=100,
                           reps=50,
                           test_timesteps=50,
                           system_cls=SNNSystem
                           )

    file = f"../cache/SNN/lr_data_fc_{l}_{noisy}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            lr_data = task.evaluate(model_cls=ManualLinearRegression, id="LR", noisy=noisy)
            lr_data.to_csv(file)
        else:
            lr_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        lr_data = task.evaluate(model_cls=ManualLinearRegression, id="LR")
        lr_data.to_csv(file)

    file = f"../cache/SNN/simple_data_{l}_{noisy}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            simple_data = task.evaluate(model_cls=Simple, model_kwargs={'epochs': 1000}, id="Simple", noisy=noisy)
            simple_data.to_csv(file)
        else:
            simple_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        simple_data = task.evaluate(model_cls=Simple, id="Simple")
        simple_data.to_csv(file)

    task.plot(lr_data)
    task.plot(simple_data)
    task.plot(pd.concat([lr_data, simple_data]))


if __name__ == '__main__':
    main()
