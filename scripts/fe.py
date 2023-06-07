import pandas as pd
import os

from dynascale.baselines.lr import LinearRegression
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    task = FixedError(
        target_loss=10e2,
        L=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80],
        E=None,
        t=50,
        max_control_cost_per_dim=0,
        control_horizons=0,
        test_examples=100,
        reps=30,
        max_samples=100000,
        test_timesteps=50,
        system_cls=LDSSystem
    )

    # for in_dist in [True, False]:
    in_dist = True
    file = f"../cache/LDS/LR/target_loss_new.csv"
    # if os.path.exists(file):
    #     data = pd.read_csv(file)
    # else:
    data = task.evaluate(model_cls=LinearRegression, id=f"LR ({in_dist})", noisy=False, in_dist=in_dist)
    data.to_csv(file)
    task.plot(data)


if __name__ == '__main__':
    main()
