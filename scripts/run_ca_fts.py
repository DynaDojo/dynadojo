import os

import pandas as pd

from dynascale.baselines.lpr import LowestPossibleRadius
from dynascale.baselines.cnn import CNN
from dynascale.systems.ca import CASystem
from dynascale.tasks import FixedTrainSize



def main():
    n = 100
    mutation_p = 0
    task = FixedTrainSize(
        n=n,
        L=range(1, 11),
        E=64,
        T=[5],
        max_control_cost_per_dim=0,
        control_horizons=0,
        test_examples=10,
        reps=50,
        test_timesteps=5,
        system_cls=CASystem,
        system_kwargs={"mutation_p": mutation_p},
    )

    file = f"../cache/CA/lpr_data_fts_{n}_{mutation_p}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            lpr_data = task.evaluate(model_cls=LowestPossibleRadius, id="LPR", noisy=True)
            lpr_data.to_csv(file)
        else:
            lpr_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        lpr_data = task.evaluate(model_cls=LowestPossibleRadius, id="LPR")
        lpr_data.to_csv(file)

    task.plot(lpr_data)

    file = f"../cache/CA/cnn_data_fts_{n}_{mutation_p}.csv"
    if os.path.exists(file):
        if input(f"'{file}' already exists. Do you want to overwrite it? [y]") == "y":
            cnn_data = task.evaluate(model_cls=CNN, model_kwargs={'epochs': 300}, id="CNN", noisy=True)
            cnn_data.to_csv(file)
        else:
            cnn_data = pd.read_csv(file)
            print("Nothing was overwritten.")
    else:
        cnn_data = task.evaluate(model_cls=CNN, id="CNN")
        cnn_data.to_csv(file)

    task.plot(cnn_data)


if __name__ == '__main__':
    main()
