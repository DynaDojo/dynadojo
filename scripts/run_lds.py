from dynascale.utils.lds import plot
import dynascale as ds
from dynascale.challenges.lds import LDSChallenge
from dynascale.tasks import FixedTrainSize, FixedComplexity
from dynascale.baselines.simple import Simple
from dynascale.baselines.lr import MyLinearRegression
import numpy as np
import scipy as sp
import pandas as pd

def main():
    task = FixedComplexity(N=[10, 5000], l=50, e=100, t=50, max_control_cost_per_dim=0,
                           control_horizons=1,
                           test_examples=10, reps=1,
                           test_timesteps=50, challenge_cls=LDSChallenge)
    simple_data = task.evaluate(model_cls=Simple, fit_kwargs={"epochs": 10}, id="simple")
    lr_data = task.evaluate(model_cls=MyLinearRegression, id="LR")
    data = pd.concat([simple_data, lr_data])
    task.plot(data)


if __name__ == '__main__':
    main()
