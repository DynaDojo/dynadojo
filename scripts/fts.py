import pandas as pd
import os

from dynascale.baselines.lr import LinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    n = 100
    add_on = True
    task = FixedTrainSize(n=n,
                          L=[10, 50, 100, 400, 600, 800, 1000],
                          E=None,
                          T=[50],
                          max_control_cost_per_dim=0,
                          control_horizons=0,
                          test_examples=100,
                          reps=8,
                          test_timesteps=50,
                          system_cls=LDSSystem
                          )

    # for in_dist in [True, False]:
    for in_dist in [True]:
        # file = f"../cache/LDS/simple_{in_dist=}.csv"
        file = f"../cache/LDS/simple/fts_{n=}_{in_dist=}.csv"
        if os.path.exists(file):
            data = pd.read_csv(file)
            if add_on:
                new_data = task.evaluate(model_cls=Simple, id=f"DNN ({in_dist})", fit_kwargs={'epochs': 300}, noisy=True, in_dist=in_dist)
                data = pd.concat((data, new_data))
                data.to_csv(file)
        else:
            # data = task.evaluate(model_cls=Simple, id=f"DNN ({in_dist})", noisy=True, in_dist=in_dist)
            data = task.evaluate(model_cls=Simple, id=f"DNN ({in_dist})", fit_kwargs={'epochs': 300}, noisy=True, in_dist=in_dist)
            data.to_csv(file)
        task.plot(data)


if __name__ == '__main__':
    main()
