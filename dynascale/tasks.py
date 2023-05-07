from .abstractions import Task, Challenge
import pandas as pd

from dynascale.utils.plotting import plotMetric, plotFixedError, plot_metric
class TargetError(Task):
    def __init__(self, N: list[int], L: list[int], E: list[int], T: list[int], supepochs: int,
                 factory_cls: type[Challenge], trials: int, test_size: int, target_err: float):
        self._target_err = target_err
        # TODO: Tommy transfer summer code by overloading evaluate if necessary
        super().__init__(N, L, E, T, supepochs, factory_cls, trials, test_size)

    def plot(self, data: pd.DataFrame):
        plotFixedError(data, "latent_dim", "n", self._target_err)




class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, E: list[int], T: list[int], supepochs: int,
                 factory_cls: type[Challenge], trials: int, test_size: int):
        L = [l] * len(N)
        super().__init__(N, L, E, T, supepochs, factory_cls, trials, test_size)

    def plot(self, *data):
        plot_metric(data, x_col="n", y_col="error")


class FixedTrainSize(Task):
    def __init__(self, n: int, L: list[int], E: list[int], T: list[int], supepochs: int,
                 factory_cls: type[Challenge], trials: int, test_size: int):
        N = [n] * len(L)
        super().__init__(N, L, E, T, supepochs, factory_cls, trials, test_size)

    def plot(self, data: pd.DataFrame):
        plotMetric(data, "latent_dim", "error")