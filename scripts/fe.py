import pandas as pd
import os

from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    target_loss = 10e2
    task = FixedError(
        target_loss=target_loss,
        L=[100, 50, 2],
        E=None,
        t=50,
        max_control_cost_per_dim=0,
        control_horizons=0,
        test_examples=100,
        reps=4,
        max_samples=10000,
        test_timesteps=50,
        system_cls=LDSSystem
    )

    for in_dist in [True, False]:
        file = f"../cache/LDS/simple/{target_loss=}_{in_dist=}.csv"
        if os.path.exists(file):
            data = pd.read_csv(file)
        else:
            # data = task.evaluate(model_cls=LinearRegression, id=f"LR ({in_dist})", noisy=False, in_dist=in_dist)
            data = task.evaluate(model_cls=Simple, id=f"DNN ({in_dist})", fit_kwargs={"epochs": 400}, noisy=False, in_dist=in_dist)
            data.to_csv(file)
        task.plot(data)


if __name__ == '__main__':
    main()
