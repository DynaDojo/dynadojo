import pandas as pd
import os

from dynascale.baselines.lr import LinearRegression
from dynascale.systems.ctln import CTLNSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    l = 5
    add_more = False
    overwrite = False
    task = FixedComplexity(N=[200],
                           l=l,
                           e=None,
                           t=50,
                           max_control_cost_per_dim=0,
                           control_horizons=0,
                           test_examples=100,
                           reps=100,
                           test_timesteps=50,
                           system_cls=CTLNSystem,
                           system_kwargs={"p": 0.5}
                           )

    file = f"../cache/TLN/fc_{l=}.csv"
    if os.path.exists(file) and not overwrite:
        data = pd.read_csv(file)
        if add_more:
            new_data = task.evaluate(model_cls=LinearRegression, id=f"LR (TLN)", noisy=False, in_dist=True)
            data = pd.concat((data, new_data))
            data.to_csv(file)
    else:
        data = task.evaluate(model_cls=LinearRegression, id=f"LR (TLN)", noisy=False, in_dist=True)
        data.to_csv(file)
    task.plot(data)


if __name__ == '__main__':
    main()
