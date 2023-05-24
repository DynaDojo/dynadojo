from dynascale.systems.lds import LDSSystem
import pandas as pd

from dynascale.baselines.lr import MyLinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity


def main():
    task = FixedComplexity(N=[10, 100], l=2, e=2, t=50, max_control_cost_per_dim=0,
                           control_horizons=1,
                           test_examples=10, reps=1,
                           test_timesteps=50, system_cls=LDSSystem)
    # simple_data = task.evaluate(model_cls=Simple, fit_kwargs={"epochs": 10}, id="simple")
    lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
    lr_data.to_csv("lr_data.csv")
    print(len(lr_data))
    # data = pd.concat(lr_data])
    task.plot(lr_data)


if __name__ == '__main__':
    main()
