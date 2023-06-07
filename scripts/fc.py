import pandas as pd
import os

from dynascale.baselines.lr import LinearRegression
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    add_more = True
    for l in [10, 20, 30, 40, 50, 100]:
        task = FixedComplexity(N=[20, 30, 40],
                               l=l,
                               e=l,
                               t=50,
                               max_control_cost_per_dim=0,
                               control_horizons=0,
                               test_examples=100,
                               reps=30,
                               test_timesteps=50,
                               system_cls=LDSSystem
                               )

    # for in_dist in [True, False]:
        in_dist = True
        file = f"../cache/LDS/contour/{l=}_{in_dist=}.csv"
        if os.path.exists(file):
            data = pd.read_csv(file)
            if add_more:
                new_data = task.evaluate(model_cls=LinearRegression, id=f"LR ({l}, {in_dist})", noisy=True, in_dist=in_dist)
                data = pd.concat((data, new_data))
                data.to_csv(file)
        else:
            data = task.evaluate(model_cls=LinearRegression, id=f"LR ({l}, {in_dist})", noisy=True, in_dist=in_dist)
            data.to_csv(file)
        task.plot(data)


if __name__ == '__main__':
    main()
