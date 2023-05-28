import itertools

import seaborn as sns
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from dynascale.utils.plotting import plot_metric
from abstractions import Model, Task, System


class FixedComplexity(Task):
    def __init__(self, N: list[int], l: int, e: int, t: int, max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        L = [l]
        E = e
        T = [t]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data):
        g = sns.lmplot(
            data=data,
            x="n", y="loss", hue="id",
            height=5,
            ci=50,
            robust=True,
            facet_kws=dict(sharey=False),
            scatter=False,
            fit_reg=True
        )
        g.set_axis_labels(x_var="$n$", y_var="Loss")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()

class FixedTrainSize(Task):
    def __init__(self, n: int, L: list[int], E: list[int] | int | None, T: list[int], max_control_cost_per_dim: int, control_horizons: int,
                 system_cls: type[System], reps: int, test_examples: int, test_timesteps: int, system_kwargs: dict = None):
        N = [n]
        super().__init__(N, L, E, T, max_control_cost_per_dim, control_horizons, system_cls, reps, test_examples, test_timesteps, system_kwargs=system_kwargs)

    @staticmethod
    def plot(data: pd.DataFrame):
        g = sns.lmplot(
            data=data,
            x="latent_dim", y="loss", hue="id",
            height=5,
            ci=50,
            robust=True,
            facet_kws=dict(sharey=False),
            scatter=False,
            fit_reg=True
        )
        g.set_axis_labels(x_var="Latent Dimension", y_var="Loss")
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
