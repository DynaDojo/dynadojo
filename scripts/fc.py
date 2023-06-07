import pandas as pd
import os

from dynascale.baselines.lr import LinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity, FixedError, FixedTrainSize


def main():
    add_more = True
    l = 50
    task = FixedComplexity(N=[10, 50, 100, 400, 600, 1000],
                           l=l,
                           e=l,
                           t=50,
                           max_control_cost_per_dim=0,
                           control_horizons=0,
                           test_examples=100,
                           reps=4,
                           test_timesteps=50,
                           system_cls=LDSSystem
                           )

    for in_dist in [True, False]:
        file = f"../cache/LDS/simple/{l}_{in_dist}.csv"
        if os.path.exists(file):
            data = pd.read_csv(file)
            if add_more:
                new_data = task.evaluate(model_cls=Simple, id=f"DNN ({l}, {in_dist})", fit_kwargs={"epochs": 25}, noisy=True, in_dist=in_dist)
                data = pd.concat((data, new_data))
                data.to_csv(file)
        else:
            data = task.evaluate(model_cls=Simple, id=f"DNN ({l}, {in_dist})", fit_kwargs={"epochs": 25},
                                     noisy=True, in_dist=in_dist)
            data.to_csv(file)
        task.plot(data)


if __name__ == '__main__':
    main()
