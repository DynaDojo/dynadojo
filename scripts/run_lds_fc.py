from dynascale.systems.lds import LDSSystem
import pandas as pd

import os
from dynascale.baselines.lr import MyLinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity


def main():
    task = FixedComplexity(N=range(100, 1010, 10),
                           l=2,
                           e=2,
                           t=50,
                           max_control_cost_per_dim=0,
                           control_horizons=1,
                           test_examples=100,
                           reps=10,
                           test_timesteps=50,
                           system_cls=LDSSystem
                           )

    file = "../cache/lr_data_fc.csv"
    if os.path.exists(file):
        if input("Do you want to overwrite the existing file? [y]") == "y":
            lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
            lr_data.to_csv(file)
        else:
            lr_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
        lr_data.to_csv(file)

    # file = "../cache/simple_data_fc.csv"
    # if os.path.exists(file):
    #     if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
    #         simple_data = task.evaluate(model_cls=Simple, model_kwargs={'epochs': 100}, id="Simple")
    #         simple_data.to_csv(file)
    #     else:
    #         simple_data = pd.read_csv(file)
    #         print("Nothing was overwritten.")
    # else:
    #     simple_data = task.evaluate(model_cls=Simple, id="Simple")
    #     simple_data.to_csv(file)

    # data = pd.concat([simple_data, lr_data])
    task.plot(lr_data)


if __name__ == '__main__':
    main()
