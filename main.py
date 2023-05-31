from dynascale.utils.lds import plot
from dynascale.systems.snn import SNNSystem
from dynascale.baselines.simple import Simple
from dynascale.tasks import FixedTrainSize
import numpy as np
import scipy as sp

from dynascale.baselines.lr import LinearRegression
from dynascale.baselines.simple import Simple
from dynascale.systems.lds import LDSSystem
from dynascale.tasks import FixedComplexity

def main():
    l = 5
    task = FixedComplexity(N=[10, 100, 200],
                           l=l,
                           e=l,
                           t=50,
                           max_control_cost_per_dim=0,
                           control_horizons=0,
                           test_examples=100,
                           reps=10,
                           test_timesteps=50,
                           system_cls=LDSSystem
                           )

    lr_data = task.evaluate(model_cls=LinearRegression, id="LR")

    task.plot(lr_data)



if __name__ == '__main__':
    main()
