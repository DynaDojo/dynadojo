from dynascale.utils.lds import plot
import dynascale as ds
from dynascale.systems.ca import CASystem
from dynascale.tasks import FixedTrainSize, FixedComplexity
from dynascale.baselines.basic import Basic
from dynascale.baselines.lpr import LowestPossibleRadius
from dynascale.baselines.cnn import CNN
import numpy as np
import scipy as sp
import pandas as pd

def main():
    # TODO: parallelize task after decision
    task = FixedTrainSize(n=500, L=[2, 3, 10], E=[64], T=[5], max_control_cost_per_dim=0, control_horizons=1, test_examples=10,
                          reps=1,
                          system_kwargs={'mutation_p': 0},
                          test_timesteps=5, challenge_cls=CASystem)
    basic_data = task.evaluate(model_cls=Basic, fit_kwargs={"epochs": 100}, id="basic")
    cnn_data = task.evaluate(model_cls=CNN, fit_kwargs={"epochs": 100}, id="cnn")
    lpr_data = task.evaluate(model_cls=LowestPossibleRadius, id="lpr")
    task.plot(pd.concat([basic_data, cnn_data, lpr_data]))

    # TODO: save plot output somewhere


if __name__ == '__main__':
    main()
